"""
09_tune_component2.py - Optuna TPE hyperparameter tuning for Component 2.

Runs a Bayesian (TPE) hyperparameter search over the Component-2
PRAME-conditioned MIL classifier defined in 10_train_component2.py. Each
trial runs N-fold (default 10) StratifiedGroupKFold CV on a stratified
random patient-level subsample of the diagnostic manifest. Per-fold val
AUCs are reported back to Optuna for median-pruning of weak trials.

Optimizations for GPU-bound execution (defaults are CPU-safe):
  --vram-cache       Preload subsample features to VRAM once at study
                     start. ~12 to 15 GB for 200 slides at fp16. After
                     that every epoch / fold / trial reads zero bytes
                     from disk.
  --n-jobs N         Run N Optuna trials concurrently in Python threads
                     sharing one CUDA context. Component-2 is tiny
                     (~200K params) so kernel-launch latency dominates;
                     parallel trials interleave on the GPU command
                     queue and give near-linear speedup until compute
                     saturates.
  --amp              bf16 autocast for the forward pass (CUDA only).

The script writes the following under results/{model}/component2_tune/:
  trials.csv          one row per trial (state, value, hyperparameters,
                      per-fold AUCs).
  best_config.json    winning hyperparameters + subsample metadata.
                      Consumed by 10_train_component2.py --config.
  tune_dashboard.png  2x2: optimization history, fANOVA importance,
                      top-5 parallel coordinates, best-trial per-fold
                      AUC.
  subsample.csv       the 200-slide subsample actually used.
  study.db            present only if --storage sqlite:///path.db.

Tuning is locked to mode='full' (the production target). The winning
config is reused unchanged when running mode='no_predicted' or
mode='no_prame' in 10_train_component2.py --compare; tuning per mode
would fit each mode's noise floor and stop measuring the PRAME signal.

Usage:
    # 5-minute CPU smoke test
    python 09_tune_component2.py --trials 3 --max-slides 30 --folds 3 \\
        --epochs 10 --n-startup-trials 2 --pruner-warmup 1

    # Production Colab run on L4 (about 12 to 25 min for 30 trials)
    python 09_tune_component2.py --trials 30 --vram-cache --amp \\
        --n-jobs 8 --vram-budget-gb 18 \\
        --storage sqlite:////content/study.db

    # Resume an interrupted Colab study
    python 09_tune_component2.py --trials 30 --vram-cache --amp \\
        --n-jobs 8 --storage sqlite:////content/study.db --resume
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    from optuna.samplers import TPESampler
    from optuna.exceptions import TrialPruned
except ImportError as exc:
    raise ImportError(
        "09_tune_component2.py requires optuna. "
        "Install with `pip install optuna`."
    ) from exc


# ---------------------------------------------------------------------------
# Shared code is loaded from 10_train_component2.py because numeric prefixes
# are not valid Python module names; spec_from_file_location is the project
# convention (same trick used in 05, 06, and 10 itself for 04).
# ---------------------------------------------------------------------------

def _load_component2():
    here = Path(__file__).resolve().parent
    spec = spec_from_file_location(
        "_c2_10", str(here / "10_train_component2.py")
    )
    mod = module_from_spec(spec)
    sys.modules["_c2_10"] = mod
    spec.loader.exec_module(mod)
    return mod


_C2 = _load_component2()
Component2MIL = _C2.Component2MIL
SlideDataset = _C2.SlideDataset
collate_bag = _C2.collate_bag
load_manifest = _C2.load_manifest
_preprocess_prame = _C2._preprocess_prame
train_one_epoch = _C2.train_one_epoch
evaluate = _C2.evaluate
_safe_auc = _C2._safe_auc
FEAT_DIMS = _C2.FEAT_DIMS


# ---------------------------------------------------------------------------
# VRAM feature cache
# ---------------------------------------------------------------------------

def build_feature_cache(df, device, dtype, budget_gb=0.0, verbose=True):
    """Pre-load patch features for every unique slide in df to device memory.

    Returns:
        cache: dict[h5_path -> Tensor on device with dtype]
        fallback: list of h5_paths that did not fit in the budget
    """
    cache = {}
    fallback = []
    elem_bytes = torch.zeros(1, dtype=dtype).element_size()
    budget_bytes = (int(budget_gb * (1024 ** 3))
                    if budget_gb and budget_gb > 0 else None)
    running = 0
    unique_paths = df["h5_path"].drop_duplicates().tolist()
    if verbose:
        dtype_name = "fp16" if dtype == torch.float16 else "fp32"
        print(f"  Pre-loading {len(unique_paths)} slides to {device} "
              f"({dtype_name})")
    for p in unique_paths:
        with h5py.File(p, "r") as f:
            arr = f["features"][:]
        size_bytes = arr.size * elem_bytes
        if budget_bytes is not None and running + size_bytes > budget_bytes:
            fallback.append(p)
            continue
        tensor = torch.from_numpy(arr.astype(np.float32)).to(
            device=device, dtype=dtype,
        )
        cache[p] = tensor
        running += size_bytes
    if verbose:
        print(f"  VRAM cache: {len(cache)} cached "
              f"({running / 1024 ** 3:.2f} GB), "
              f"{len(fallback)} disk fallback")
    return cache, fallback


class CachedSlideDataset(SlideDataset):
    """SlideDataset subclass that pulls features from a pre-loaded cache.

    Falls back to h5 read on cache miss so partial caching still works.
    PRAME / has_prame / cohort_idx / label come from the parent class's
    arrays. cohort_idxs is required by the parent's 5-tuple contract; the
    tuner does not engage the DANN adversary so the value is functionally
    inert (a zero placeholder is sufficient).
    """

    def __init__(self, slide_paths, prames, has_prames, cohort_idxs, labels,
                 feature_cache):
        super().__init__(slide_paths, prames, has_prames, cohort_idxs, labels)
        self.feature_cache = feature_cache

    def __getitem__(self, idx):
        path = self.slide_paths[idx]
        feat = self.feature_cache.get(path)
        if feat is None:
            with h5py.File(path, "r") as f:
                feat = torch.from_numpy(f["features"][:].astype(np.float32))
        return (
            feat,
            torch.tensor(self.prames[idx], dtype=torch.float32),
            bool(self.has_prames[idx]),
            torch.tensor(self.cohort_idxs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Stratified patient-level subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(df, max_slides, min_per_cohort, seed):
    """Patient-level stratified random subsample of the manifest.

    Stratify on (source_group, label) tuple. Sample whole patients per
    stratum so groups aren't split across in/out (that would break the
    StratifiedGroupKFold invariant in the inner CV). Per-stratum target
    is proportional to cohort size with a floor of min_per_cohort capped
    by the cohort size. Allow up to ~10% overshoot of max_slides as a
    side-effect of whole-patient sampling.
    """
    rng = np.random.RandomState(seed)
    total = len(df)
    picked = []
    print(f"\nStratified subsample (target {max_slides}, "
          f"floor {min_per_cohort}, seed {seed}):")
    for (source, label), grp in df.groupby(["source_group", "label"]):
        n_in_stratum = len(grp)
        proportional = int(round(max_slides * n_in_stratum / total))
        target = max(min(min_per_cohort, n_in_stratum), proportional)
        target = min(target, n_in_stratum)
        patients = grp["patient"].drop_duplicates().tolist()
        rng.shuffle(patients)
        chosen = []
        cum = 0
        for pat in patients:
            n_pat_slides = int((grp["patient"] == pat).sum())
            chosen.append(pat)
            cum += n_pat_slides
            if cum >= target:
                break
        sub = grp[grp["patient"].isin(chosen)]
        picked.append(sub)
        print(f"  ({source!s:14s}, label={label}): "
              f"{len(sub):3d} slides from {len(chosen):3d} patients "
              f"(target ~{target})")
    out = pd.concat(picked, ignore_index=True)
    overshoot_pct = max(0, len(out) - max_slides) / max(1, max_slides) * 100
    print(f"  Total: {len(out)} slides "
          f"(target {max_slides}, overshoot {overshoot_pct:.1f}%)")
    return out


# ---------------------------------------------------------------------------
# Per-trial training (one fold)
# ---------------------------------------------------------------------------

def _make_loaders(df, train_idx, val_idx, feature_cache):
    # Tuner doesn't engage DANN; pass placeholder cohort_idx=0 for every row.
    # Backbone Component2MIL has no .grl, so the cohort_idx is never read
    # downstream — but the 5-tuple SlideDataset contract still requires it.
    def _ds(idx):
        sub = df.iloc[idx]
        cohort_col = sub.get("cohort_idx", pd.Series([0] * len(sub)))
        cohort_list = cohort_col.tolist() if hasattr(cohort_col, "tolist") else list(cohort_col)
        if len(cohort_list) != len(sub):
            cohort_list = [0] * len(sub)
        return CachedSlideDataset(
            sub["h5_path"].tolist(),
            sub["prame"].tolist(),
            sub["has_prame"].tolist(),
            cohort_list,
            sub["label"].tolist(),
            feature_cache,
        )
    return (
        DataLoader(_ds(train_idx), batch_size=1, shuffle=True,
                   collate_fn=collate_bag, num_workers=0),
        DataLoader(_ds(val_idx), batch_size=1, shuffle=False,
                   collate_fn=collate_bag, num_workers=0),
    )


def _train_one_fold(df, train_idx, val_idx, feat_dim, hp, device,
                    feature_cache, epochs, amp):
    """Train Component2MIL on one CV fold; return (best_val_auc, n_epochs_run)."""
    train_loader, val_loader = _make_loaders(
        df, train_idx, val_idx, feature_cache,
    )
    model = Component2MIL(
        feat_dim,
        hidden_dim=hp["hidden_dim"],
        attn_dim=hp["attn_dim"],
        dropout=hp["dropout"],
        use_prame=True,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )

    best_val_auc = -np.inf
    patience_counter = 0
    epochs_run = 0
    for epoch in range(epochs):
        train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            entropy_lambda=hp["entropy_lambda"],
            grad_clip=hp["grad_clip"],
            label_smoothing=hp["label_smoothing"],
            amp=amp,
        )
        _, val_auc, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, amp=amp,
        )
        scheduler.step()
        epochs_run += 1
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= hp["patience"]:
            break

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_val_auc <= -np.inf:
        return float("nan"), epochs_run
    return float(best_val_auc), epochs_run


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _suggest_hyperparams(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True,
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", [128, 256, 384, 512],
        ),
        "attn_dim": trial.suggest_categorical(
            "attn_dim", [64, 128, 192, 256],
        ),
        "patience": trial.suggest_int("patience", 5, 20),
        "grad_clip": trial.suggest_categorical(
            "grad_clip", [0.0, 0.5, 1.0, 2.0],
        ),
        "label_smoothing": trial.suggest_float(
            "label_smoothing", 0.0, 0.10,
        ),
        "entropy_lambda": trial.suggest_categorical(
            "entropy_lambda", [0.0, 1e-4, 1e-3],
        ),
        "prame_norm": trial.suggest_categorical(
            "prame_norm", ["log", "raw", "zscore_per_source"],
        ),
    }


def make_objective(df_sub, feat_dim, device, feature_cache, args):
    """Closure capturing the subsample + feature cache + run-level args."""

    def objective(trial):
        hp = _suggest_hyperparams(trial)
        # Apply trial's prame_norm choice. df_sub keeps prame_raw + source_group.
        df = df_sub.copy()
        df["prame"] = _preprocess_prame(
            df["prame_raw"].values, df["source_group"].values, hp["prame_norm"],
        )

        skf = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed,
        )
        splits = list(skf.split(
            df["h5_path"].values, df["label"].values,
            groups=df["patient"].values,
        ))

        per_fold_aucs = []
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            best_auc, n_epochs = _train_one_fold(
                df, train_idx, val_idx, feat_dim, hp, device,
                feature_cache, args.epochs, args.amp,
            )
            per_fold_aucs.append(best_auc)
            running_mean = float(np.nanmean(per_fold_aucs))
            trial.report(running_mean, step=fold_i)
            trial.set_user_attr("per_fold_aucs", per_fold_aucs)
            trial.set_user_attr("n_folds_completed", len(per_fold_aucs))
            if trial.should_prune():
                raise TrialPruned()

        return float(np.nanmean(per_fold_aucs))

    return objective


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dashboard(study, out_path, model_name):
    """2x2 dashboard at dpi=150, matching 04_train_mil_reg.py's style."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    complete = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,),
    )

    # (0,0) Optimization history with running best
    ax = axes[0, 0]
    if complete:
        nums = [t.number for t in complete]
        vals = [t.value for t in complete]
        ax.scatter(nums, vals, color="steelblue", s=42, alpha=0.7,
                   label="trial")
        running = np.maximum.accumulate(vals)
        ax.plot(nums, running, color="darkred", lw=2, label="running best")
        ax.legend()
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Objective (mean val AUC)")
    ax.set_title("Optimization History")
    ax.grid(alpha=0.25)

    # (0,1) fANOVA hyperparameter importance
    ax = axes[0, 1]
    try:
        imp = optuna.importance.get_param_importances(study)
        names = list(imp.keys())[::-1]
        vals = [imp[n] for n in names]
        ax.barh(names, vals, color="seagreen", alpha=0.85)
        ax.set_xlabel("Importance (fANOVA)")
        ax.set_title("Hyperparameter Importance")
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"Importance unavailable\n({exc})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10)
        ax.set_title("Hyperparameter Importance")

    # (1,0) Top-5 parallel coordinates
    ax = axes[1, 0]
    top5 = sorted(complete, key=lambda t: t.value, reverse=True)[:5]
    if top5:
        keys = list(top5[0].params.keys())
        meta = {}
        for k in keys:
            seen = [t.params.get(k) for t in complete if t.params.get(k) is not None]
            if seen and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in seen):
                meta[k] = ("num", min(seen), max(seen))
            else:
                cats = sorted({str(v) for v in seen})
                meta[k] = ("cat", cats)
        x = np.arange(len(keys))
        cmap = plt.get_cmap("viridis")
        for i, t in enumerate(top5):
            ys = []
            for k in keys:
                m = meta[k]
                v = t.params[k]
                if m[0] == "num":
                    lo, hi = m[1], m[2]
                    ys.append((v - lo) / max(hi - lo, 1e-12))
                else:
                    cats = m[1]
                    ys.append(cats.index(str(v))
                              / max(len(cats) - 1, 1))
            ax.plot(x, ys, color=cmap(i / max(len(top5) - 1, 1)),
                    lw=2, marker="o", alpha=0.85,
                    label=f"#{t.number} AUC={t.value:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Normalized value")
        ax.set_title("Top-5 Trials (Parallel Coordinates)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    # (1,1) Per-fold val AUC of the best trial
    ax = axes[1, 1]
    if top5:
        best = top5[0]
        per_fold = best.user_attrs.get("per_fold_aucs", [])
        if per_fold:
            xs = list(range(1, len(per_fold) + 1))
            ax.bar(xs, per_fold, color="steelblue", alpha=0.85)
            mean_auc = float(np.nanmean(per_fold))
            ax.axhline(mean_auc, color="darkred", linestyle="--",
                       label=f"Mean: {mean_auc:.3f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Val AUC")
            ax.set_title(f"Per-Fold AUC (Best Trial #{best.number})")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.25)

    plt.suptitle(
        f"{model_name.upper()} - Component 2 Hyperparameter Tuning",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def write_trials_csv(study, out_path):
    rows = []
    for t in study.get_trials(deepcopy=False):
        row = {
            "trial": t.number,
            "state": t.state.name,
            "value": t.value if t.value is not None else float("nan"),
            "n_folds_completed": t.user_attrs.get(
                "n_folds_completed",
                len(t.user_attrs.get("per_fold_aucs", [])),
            ),
            "datetime_start": (t.datetime_start.isoformat()
                               if t.datetime_start else ""),
            "datetime_complete": (t.datetime_complete.isoformat()
                                  if t.datetime_complete else ""),
        }
        for i, auc in enumerate(t.user_attrs.get("per_fold_aucs", [])):
            row[f"fold{i + 1}_auc"] = auc
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_best_config(study, args, subsample_df, out_path):
    cohort_counts = {
        f"{src}|label={lab}": int(n)
        for (src, lab), n in subsample_df.groupby(
            ["source_group", "label"]
        ).size().items()
    }
    best = study.best_trial
    cfg = {
        "model": args.model,
        "hyperparameters": dict(best.params),
        "best_value": float(best.value) if best.value is not None else None,
        "n_trials": len(study.trials),
        "n_complete": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ),
        "subsample": {
            "seed": args.seed,
            "max_slides": args.max_slides,
            "min_per_cohort": args.min_per_cohort,
            "actual_slides": int(len(subsample_df)),
            "actual_patients": int(subsample_df["patient"].nunique()),
            "cohort_counts": cohort_counts,
        },
        "study_name": args.study_name,
        "folds": args.folds,
        "epochs": args.epochs,
        "datetime": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("  [warn] --device cuda requested but CUDA not available; "
              "using CPU")
        return torch.device("cpu")
    return torch.device(name)


def _make_pruner(name, n_startup_trials, warmup):
    if name == "median":
        return MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=warmup,
        )
    if name == "hyperband":
        return HyperbandPruner(
            min_resource=max(1, warmup),
            max_resource="auto",
        )
    if name == "none":
        return NopPruner()
    raise ValueError(f"unknown --pruner: {name!r}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter tuning for Component 2.",
    )
    p.add_argument("--model", choices=list(FEAT_DIMS.keys()), default="uni")
    p.add_argument("--manifest",
                   default="data/expression/diagnostic_manifest.csv")
    p.add_argument("--emb-dir", default="embeddings")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--trials", type=int, default=30,
                   help="Optuna trial budget (default: 30)")
    p.add_argument("--folds", type=int, default=10,
                   help="Inner CV fold count (default: 10)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-slides", type=int, default=200,
                   help="Stratified subsample target slide count")
    p.add_argument("--min-per-cohort", type=int, default=20,
                   help="Floor per source_group; capped by cohort size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-startup-trials", type=int, default=10,
                   help="TPE random-warmup trials before Bayesian phase")
    p.add_argument("--pruner", choices=("median", "hyperband", "none"),
                   default="median")
    p.add_argument("--pruner-warmup", type=int, default=2,
                   help="Folds completed before pruning may kick in")
    p.add_argument("--study-name", default="component2_full")
    p.add_argument("--storage", default="",
                   help="Optuna RDB URL (sqlite:///path.db); "
                        "blank = in-memory")
    p.add_argument("--resume", action="store_true",
                   help="Resume existing study; requires --storage")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"),
                   default="auto")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Concurrent Optuna trials (GPU only)")
    p.add_argument("--vram-cache", action="store_true",
                   help="Preload subsample features to VRAM once")
    p.add_argument("--vram-budget-gb", type=float, default=0.0,
                   help="VRAM cap for the cache in GB; 0 = no cap")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast for forward pass (CUDA only)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _resolve_device(args.device)
    feat_dim = FEAT_DIMS[args.model]

    if device.type == "cuda":
        # Brutal GPU defaults: TF32 matmul + cuDNN benchmark. TF32 is ~1.5-2x
        # faster on Ampere+ and the precision drop is well below AUC noise for
        # this task. cuDNN benchmark keys on input shape; per-slide patch counts
        # vary so cache reuse is partial, but the worst case is no speedup.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 64)
    print("Component 2 Hyperparameter Tuning (Optuna TPE)")
    print("=" * 64)
    print(f"Device:       {device}")
    print(f"Model:        {args.model.upper()} (feat_dim={feat_dim})")
    print(f"Trials:       {args.trials}  "
          f"(n_startup={args.n_startup_trials})")
    print(f"Folds:        {args.folds}")
    print(f"Max epochs:   {args.epochs}")
    print(f"Subsample:    target {args.max_slides}, "
          f"min/cohort {args.min_per_cohort}, seed {args.seed}")
    print(f"Concurrency:  n_jobs={args.n_jobs}  "
          f"vram_cache={'yes' if args.vram_cache else 'no'}  "
          f"amp={'bf16' if args.amp else 'fp32'}")
    print(f"Pruner:       {args.pruner} (warmup={args.pruner_warmup})")
    if args.storage:
        print(f"Storage:      {args.storage}"
              f"{' (resume)' if args.resume else ''}")

    df_full = load_manifest(
        args.manifest, args.emb_dir, args.model, "full", "raw",
    )
    # load_manifest produced a 'prame' column equal to prame_raw (norm=raw);
    # per-trial we re-apply the trial's prame_norm to prame_raw.

    df_sub = stratified_subsample(
        df_full, args.max_slides, args.min_per_cohort, args.seed,
    ).reset_index(drop=True)

    results_dir = Path(args.results_dir) / args.model / "component2_tune"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(results_dir / "subsample.csv", index=False)

    feature_cache = {}
    if args.vram_cache:
        if device.type == "cuda":
            cache_dtype = torch.float16 if args.amp else torch.float32
            feature_cache, fallback = build_feature_cache(
                df_sub, device, cache_dtype, args.vram_budget_gb,
            )
            if fallback:
                print(f"  [warn] {len(fallback)} slides exceeded VRAM "
                      f"budget; falling back to disk")
        else:
            print("  [warn] --vram-cache requested but device is CPU; "
                  "ignoring")

    sampler = TPESampler(
        n_startup_trials=args.n_startup_trials, seed=args.seed,
    )
    pruner = _make_pruner(
        args.pruner, args.n_startup_trials, args.pruner_warmup,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage or None,
        load_if_exists=bool(args.resume),
    )

    objective = make_objective(df_sub, feat_dim, device, feature_cache, args)

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=max(1, args.n_jobs),
        catch=(RuntimeError,),
    )
    elapsed = time.time() - t0
    print(f"\nStudy finished in {elapsed / 60:.1f} min")

    print(f"\nWriting results to {results_dir}/")
    write_trials_csv(study, results_dir / "trials.csv")
    print("  trials.csv")
    write_best_config(study, args, df_sub, results_dir / "best_config.json")
    print("  best_config.json")
    plot_dashboard(study, results_dir / "tune_dashboard.png", args.model)
    print("  tune_dashboard.png")
    print("  subsample.csv")

    if study.best_trial is None:
        print("\n[warn] No completed trials. Increase --trials or "
              "reduce pruning.")
        return

    best = study.best_trial
    print("\n" + "=" * 64)
    print(f"BEST TRIAL #{best.number}: val AUC = {best.value:.4f}")
    print("=" * 64)
    for k, v in best.params.items():
        print(f"  {k:18s} = {v}")
    print()
    print(f"Next: python 10_train_component2.py --compare "
          f"--config {results_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
