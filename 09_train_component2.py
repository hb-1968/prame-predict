"""
Train the Component-2 PRAME-conditioned diagnostic MIL classifier.

Component 2 fuses H&E patch features and per-slide PRAME (measured for
SKCM/GTEx, pseudobulked for HEST, predicted by Component 1 for COBRA)
into a single multimodal MIL classifier. The PRAME scalar is projected
to the hidden representation space and appended to the patch bag as
one extra MIL instance; gated-attention pooling then co-attends over
the patch instances and the PRAME instance.

Three modes selected via --mode let the user ablate the PRAME signal:

    full          : PRAME projection + PRAME instance for has_prame=True rows
    no_predicted  : same as full, but rows with prame_source==
                    "component1_predicted" (COBRA) are silenced
                    (has_prame forced to False)
    no_prame      : PRAME projection branch is not built; pure visual MIL
                    baseline. Architecturally identical to Component 1's
                    AttentionMIL.

--compare runs all three modes through the full N-fold CV (--folds,
default 5) with the same deterministic patient-level split and emits
a bundled comparison plot + JSON summary on top of the per-mode CV
artifacts. Use `--compare --folds 1` for a single-fold sanity check.

Usage:
    python 09_train_component2.py --mode full
    python 09_train_component2.py --mode no_predicted
    python 09_train_component2.py --mode no_prame
    python 09_train_component2.py --compare
    python 09_train_component2.py --mode full --epochs 100 --folds 5
"""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# AttentionMIL is loaded from 04_train_mil.py (numeric prefix is not a valid
# Python module name; project convention is SourceFileLoader / spec_from_file).
# ---------------------------------------------------------------------------

def _load_attention_mil():
    here = Path(__file__).resolve().parent
    spec = spec_from_file_location("_mil_04", str(here / "04_train_mil.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AttentionMIL


AttentionMIL = _load_attention_mil()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAT_DIMS = {"uni": 1024, "conch": 768}

# Per-cohort embedding subdirectory under --emb-dir. Component-1 SKCM lives
# at {emb_root}/{model}; the Component-2 cohorts get their own subdirs so
# they don't collide with the SKCM run.
SOURCE_EMB_SUBDIR = {
    "skcm_melanoma": "{model}",
    "skcm_normal":   "{model}",
    "gtex_normal":   "{model}_gtex",
    "cobra_bcc":     "{model}_cobra",
    "hest_visium":   "{model}_hest",
}

PREDICTED_PRAME_SOURCE = "component1_predicted"

MODES = ("full", "no_predicted", "no_prame")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Component2MIL(AttentionMIL):
    """PRAME-as-MIL-instance classifier.

    Inherits AttentionMIL's feature_net, gated attention, and classifier
    head. When use_prame=True, an additional 1->hidden_dim projection
    encodes the per-slide PRAME scalar; the projected vector is appended
    as one extra instance to the post-feature_net bag, so gated-attention
    pooling co-attends over patch instances and the PRAME instance.

    When use_prame=False, the prame_proj layer is not constructed and the
    model is architecturally identical to AttentionMIL.
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim=256,
        attn_dim=128,
        dropout=0.25,
        use_prame=True,
    ):
        super().__init__(feat_dim, hidden_dim, attn_dim, dropout)
        self.use_prame = use_prame
        if use_prame:
            # Tanh-bounded so the PRAME instance can't dominate the bag at init.
            self.prame_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
            )

    def forward(self, x, prame=None, has_prame=False):
        h = self.feature_net(x)  # (N, hidden_dim)

        if self.use_prame and bool(has_prame):
            prame_scalar = prame.view(1, 1).to(h.dtype)
            prame_h = self.prame_proj(prame_scalar)  # (1, hidden_dim)
            h = torch.cat([h, prame_h], dim=0)        # (N+1, hidden_dim)

        a_V = self.attention_V(h)
        a_U = self.attention_U(h)
        a = self.attention_w(a_V * a_U)
        attention = torch.softmax(a, dim=0).squeeze(-1)

        slide_repr = (attention.unsqueeze(-1) * h).sum(dim=0)
        logit = self.classifier(slide_repr).squeeze()
        return logit, attention


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlideDataset(Dataset):
    """Per-slide patch features + PRAME scalar + has_prame flag + label."""

    def __init__(self, slide_paths, prames, has_prames, labels):
        self.slide_paths = slide_paths
        self.prames = prames
        self.has_prames = has_prames
        self.labels = labels

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        with h5py.File(self.slide_paths[idx], "r") as f:
            features = f["features"][:].astype(np.float32)
        return (
            torch.from_numpy(features),
            torch.tensor(self.prames[idx], dtype=torch.float32),
            bool(self.has_prames[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def collate_bag(batch):
    # Batch size is always 1 (variable-size bags); skip the default stack.
    return batch[0]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _resolve_emb_path(row, emb_root, model_name):
    """Find the .h5 embedding for a manifest row.

    Tries cohort-specific subdirs (per SOURCE_EMB_SUBDIR) before falling
    back to Component-1's flat SKCM location. Tries multiple filename
    conventions because the SKCM run saves embeddings by .svs basename
    while the COBRA Colab notebook saves them by file_id stem.
    """
    file_id = str(row.get("file_id", "") or "")
    file_name = str(row.get("file_name", "") or "")
    source = str(row.get("source_group", "") or "")

    subdirs = []
    if source in SOURCE_EMB_SUBDIR:
        subdirs.append(SOURCE_EMB_SUBDIR[source].format(model=model_name))
    subdirs.append(model_name)

    names = []
    if file_name.endswith(".svs"):
        names.append(file_name.replace(".svs", ".h5"))
    if file_id:
        names.append(f"{file_id}.h5")
    if file_name and not file_name.endswith(".h5"):
        names.append(f"{Path(file_name).stem}.h5")
    if file_name.endswith(".h5"):
        names.append(file_name)

    for sd in subdirs:
        for n in names:
            p = Path(emb_root) / sd / n
            if p.exists():
                return p
    return None


def _preprocess_prame(prames, sources, norm):
    prames = np.asarray(prames, dtype=np.float32)
    sources = np.asarray(sources)
    if norm == "raw":
        return prames
    if norm == "log":
        return np.log1p(np.maximum(prames, 0.0))
    if norm == "zscore_per_source":
        out = np.zeros_like(prames)
        for s in np.unique(sources):
            mask = sources == s
            if mask.sum() > 1:
                vals = prames[mask]
                mu = float(vals.mean())
                sigma = float(vals.std() + 1e-8)
                out[mask] = (vals - mu) / sigma
        return out
    raise ValueError(f"unknown --prame-norm: {norm!r}")


def load_manifest(manifest_path, emb_root, model_name, mode, prame_norm):
    """Read manifest, resolve embeddings, apply mode-specific has_prame masking.

    Returns a DataFrame with columns: h5_path, prame, has_prame, label,
    patient, source_group, prame_source. PRAME has already been preprocessed
    via --prame-norm at this point.
    """
    raw = pd.read_csv(manifest_path)
    print(f"Manifest: {len(raw)} rows from {manifest_path}")

    # Defensive: older manifests may lack has_prame / prame_source.
    if "has_prame" not in raw.columns:
        raw["has_prame"] = raw["prame_tpm"].notna()
    if "prame_source" not in raw.columns:
        raw["prame_source"] = ""
    if "source_group" not in raw.columns:
        raw["source_group"] = ""

    rows = []
    missing = 0
    for _, r in raw.iterrows():
        p = _resolve_emb_path(r, emb_root, model_name)
        if p is None:
            missing += 1
            continue
        has_prame = bool(r["has_prame"]) and pd.notna(r.get("prame_tpm"))
        prame_source = str(r.get("prame_source", "") or "")

        if mode == "no_predicted" and prame_source == PREDICTED_PRAME_SOURCE:
            has_prame = False

        rows.append({
            "h5_path": str(p),
            "prame_raw": float(r["prame_tpm"]) if pd.notna(r.get("prame_tpm")) else 0.0,
            "has_prame": has_prame,
            "label": int(r["melanoma_label"]),
            "patient": str(r["submitter_id"]),
            "source_group": str(r.get("source_group", "") or ""),
            "prame_source": prame_source,
        })

    df = pd.DataFrame(rows)
    if missing:
        print(f"  [warn] {missing} rows skipped (no .h5 embedding found)")
    print(f"  {len(df)} slides resolved")

    if len(df) == 0:
        raise RuntimeError("No embeddings resolved; check --emb-dir and manifest paths.")

    # Per-source breakdown
    for s in sorted(df["source_group"].unique()):
        sub = df[df["source_group"] == s]
        n_pos = int(sub["label"].sum())
        n_prame = int(sub["has_prame"].sum())
        print(f"    {s:18s}  n={len(sub):4d}  pos={n_pos:4d}  has_prame={n_prame:4d}")

    df["prame"] = _preprocess_prame(
        df["prame_raw"].values, df["source_group"].values, prame_norm,
    )
    return df


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds, truths = [], []
    for features, prame, has_prame, label in loader:
        features = features.to(device)
        prame = prame.to(device)
        label = label.to(device)

        logit, _ = model(features, prame, has_prame)
        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.append(torch.sigmoid(logit).item())
        truths.append(label.item())

    auc = _safe_auc(truths, preds)
    return total_loss / max(1, len(loader)), auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, truths = [], []
    with torch.inference_mode():
        for features, prame, has_prame, label in loader:
            features = features.to(device)
            prame = prame.to(device)
            label = label.to(device)

            logit, _ = model(features, prame, has_prame)
            loss = criterion(logit, label)

            total_loss += loss.item()
            preds.append(torch.sigmoid(logit).item())
            truths.append(label.item())

    auc = _safe_auc(truths, preds)
    acc = accuracy_score(truths, [int(p > 0.5) for p in preds])
    return total_loss / max(1, len(loader)), auc, acc, preds, truths


def _safe_auc(truths, preds):
    if len(set(truths)) < 2:
        return float("nan")
    return roc_auc_score(truths, preds)


def _build_model(feat_dim, args, mode, device):
    use_prame = mode in ("full", "no_predicted")
    model = Component2MIL(
        feat_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
        use_prame=use_prame,
    ).to(device)
    return model


def train_one_fold(
    fold_idx,
    train_idx,
    val_idx,
    df,
    feat_dim,
    args,
    mode,
    device,
):
    print(f"\n{'=' * 50}")
    print(f"Fold {fold_idx + 1}/{args.folds}  (mode={mode})")
    print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

    train_patients = set(df.iloc[train_idx]["patient"])
    val_patients = set(df.iloc[val_idx]["patient"])
    assert train_patients.isdisjoint(val_patients), "Patient leakage!"

    train_ds = SlideDataset(
        df.iloc[train_idx]["h5_path"].tolist(),
        df.iloc[train_idx]["prame"].tolist(),
        df.iloc[train_idx]["has_prame"].tolist(),
        df.iloc[train_idx]["label"].tolist(),
    )
    val_ds = SlideDataset(
        df.iloc[val_idx]["h5_path"].tolist(),
        df.iloc[val_idx]["prame"].tolist(),
        df.iloc[val_idx]["has_prame"].tolist(),
        df.iloc[val_idx]["label"].tolist(),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_bag, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_bag, num_workers=0)

    model = _build_model(feat_dim, args, mode, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_auc": [], "val_auc": [],
    }

    for epoch in range(args.epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_loss, val_auc, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d} | "
                  f"Train Loss {train_loss:.4f} AUC {train_auc:.3f} | "
                  f"Val Loss {val_loss:.4f} AUC {val_auc:.3f} Acc {val_acc:.3f}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch + 1} (best: epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_auc, val_acc, val_preds, val_truths = evaluate(
        model, val_loader, criterion, device,
    )

    val_binary = [int(p > 0.5) for p in val_preds]
    if len(set(val_truths)) >= 2:
        tn, fp, fn, tp = confusion_matrix(val_truths, val_binary).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens = spec = float("nan")

    fold_result = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "val_auc": val_auc,
        "val_acc": val_acc,
        "sensitivity": sens,
        "specificity": spec,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
    }
    return fold_result, history, best_state, val_preds, val_truths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cv_results(fold_results, pool_truths, pool_preds, model_name, mode, out_dir):
    aucs = [r["val_auc"] for r in fold_results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(range(1, len(aucs) + 1), aucs, color="steelblue", alpha=0.8)
    ax.axhline(np.nanmean(aucs), color="red", linestyle="--",
               label=f"Mean: {np.nanmean(aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title(f"{model_name.upper()} {mode} - Val AUC per Fold")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1]
    if len(set(pool_truths)) >= 2:
        fpr, tpr, _ = roc_curve(pool_truths, pool_preds)
        pooled_auc = roc_auc_score(pool_truths, pool_preds)
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"Pooled AUC = {pooled_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name.upper()} {mode} - Pooled ROC")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"cv_results_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(histories_per_fold, model_name, mode, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, h in enumerate(histories_per_fold):
        axes[0].plot(h["train_loss"], alpha=0.5, label=f"Fold {i+1} train")
        axes[0].plot(h["val_loss"], alpha=0.5, linestyle="--", label=f"Fold {i+1} val")
        axes[1].plot(h["train_auc"], alpha=0.5, label=f"Fold {i+1} train")
        axes[1].plot(h["val_auc"], alpha=0.5, linestyle="--", label=f"Fold {i+1} val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} {mode} - Loss")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{model_name.upper()} {mode} - AUC")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / f"training_curves_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_compare(per_mode, model_name, n_folds, out_dir):
    """2x2 grid summarizing an N-fold CV ablation across three PRAME modes.

    (0,0) Val AUC per epoch: every fold as a thin trace + bold mean line per mode.
    (0,1) Pooled ROC across all folds, one curve per mode.
    (1,0) Per-fold val AUC grouped bar chart.
    (1,1) Aggregate metrics (mean across folds) with std error bars.
    """
    colors = {
        "full":         "steelblue",
        "no_predicted": "coral",
        "no_prame":     "seagreen",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Per-fold val AUC traces + mean line per mode
    ax = axes[0, 0]
    for mode in MODES:
        c = colors[mode]
        histories = per_mode[mode]["histories"]
        for h in histories:
            ax.plot(h["val_auc"], color=c, alpha=0.25, lw=1)
        # Mean line — pad with NaN where folds end early due to early stopping
        max_ep = max(len(h["val_auc"]) for h in histories) if histories else 0
        means = []
        for ep in range(max_ep):
            vals = [h["val_auc"][ep] for h in histories if ep < len(h["val_auc"])]
            means.append(float(np.nanmean(vals)) if vals else np.nan)
        ax.plot(means, color=c, lw=2.5, label=mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Val AUC per Fold (light) + Mean (bold)")
    ax.legend()
    ax.set_ylim(0, 1)

    # (0,1) Pooled ROC across folds per mode
    ax = axes[0, 1]
    for mode in MODES:
        c = colors[mode]
        truths = per_mode[mode]["pool_truths"]
        preds = per_mode[mode]["pool_preds"]
        if len(set(truths)) >= 2:
            fpr, tpr, _ = roc_curve(truths, preds)
            auc = roc_auc_score(truths, preds)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{mode} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC (held-out predictions, all folds)")
    ax.legend()

    # (1,0) Per-fold val AUC grouped bar chart
    ax = axes[1, 0]
    x = np.arange(n_folds)
    width = 0.8 / len(MODES)
    for i, mode in enumerate(MODES):
        aucs = [r["val_auc"] for r in per_mode[mode]["fold_results"]]
        offset = (i - (len(MODES) - 1) / 2) * width
        ax.bar(x + offset, aucs, width, label=mode,
               color=colors[mode], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {k + 1}" for k in range(n_folds)])
    ax.set_ylabel("Val AUC")
    ax.set_title("Per-Fold Val AUC by Mode")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # (1,1) Aggregate metrics with std error bars
    ax = axes[1, 1]
    metric_names = ["Val AUC", "Val Acc", "Sensitivity", "Specificity"]
    mean_keys = ["mean_auc", "mean_acc", "mean_sensitivity", "mean_specificity"]
    std_keys =  ["std_auc",  "std_acc",  "std_sensitivity",  "std_specificity"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(MODES)
    for i, mode in enumerate(MODES):
        s = per_mode[mode]["summary"]
        means = [s[k] for k in mean_keys]
        stds  = [s[k] for k in std_keys]
        offset = (i - (len(MODES) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=mode, color=colors[mode], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Aggregate Metrics ({n_folds}-fold mean +/- std)")
    ax.legend(fontsize=8)

    plt.suptitle(
        f"{model_name.upper()} - PRAME Ablation ({n_folds}-fold CV)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "compare_variants.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _cv_split(df, args):
    skf = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed,
    )
    return list(skf.split(
        df["h5_path"].values,
        df["label"].values,
        groups=df["patient"].values,
    ))


def run_full_cv(df, feat_dim, args, mode, device, results_dir):
    splits = _cv_split(df, args)
    fold_results = []
    histories = []
    all_val_preds = np.zeros(len(df))
    all_val_labels = np.zeros(len(df))
    val_indices_all = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        result, history, best_state, val_preds, val_truths = train_one_fold(
            fold_i, train_idx, val_idx, df, feat_dim, args, mode, device,
        )
        fold_results.append(result)
        histories.append(history)

        if best_state is not None:
            torch.save(
                best_state,
                results_dir / f"fold{fold_i + 1}_{mode}_model.pt",
            )

        for i, idx in enumerate(val_idx):
            all_val_preds[idx] = val_preds[i]
            all_val_labels[idx] = val_truths[i]
        val_indices_all.extend(int(x) for x in val_idx.tolist())

        print(f"  Best epoch {result['best_epoch']} | "
              f"Val AUC {result['val_auc']:.3f} Acc {result['val_acc']:.3f} | "
              f"Sens {result['sensitivity']:.3f} Spec {result['specificity']:.3f}")

    # Aggregate
    print(f"\n{'=' * 50}\nAGGREGATE - {args.model.upper()} mode={mode}\n{'=' * 50}")
    aucs = [r["val_auc"] for r in fold_results]
    accs = [r["val_acc"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]

    val_idx_arr = np.array(val_indices_all)
    if len(set(all_val_labels[val_idx_arr].tolist())) >= 2:
        pooled_auc = float(roc_auc_score(
            all_val_labels[val_idx_arr], all_val_preds[val_idx_arr],
        ))
    else:
        pooled_auc = float("nan")

    print(f"AUC         {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"Acc         {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    print(f"Sensitivity {np.nanmean(sens):.3f} +/- {np.nanstd(sens):.3f}")
    print(f"Specificity {np.nanmean(specs):.3f} +/- {np.nanstd(specs):.3f}")
    print(f"Pooled AUC  {pooled_auc:.3f}")

    # Save per-fold CSV
    pd.DataFrame(fold_results).to_csv(
        results_dir / f"cv_results_{mode}.csv", index=False,
    )

    # Summary JSON
    summary = {
        "model": args.model,
        "mode": mode,
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "prame_norm": args.prame_norm,
        "num_slides": len(df),
        "num_patients": int(df["patient"].nunique()),
        "mean_auc": float(np.nanmean(aucs)),
        "std_auc": float(np.nanstd(aucs)),
        "mean_acc": float(np.nanmean(accs)),
        "std_acc": float(np.nanstd(accs)),
        "mean_sensitivity": float(np.nanmean(sens)),
        "std_sensitivity": float(np.nanstd(sens)),
        "mean_specificity": float(np.nanmean(specs)),
        "std_specificity": float(np.nanstd(specs)),
        "pooled_auc": pooled_auc,
    }
    with open(results_dir / f"summary_{mode}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_cv_results(
        fold_results,
        all_val_labels[val_idx_arr].tolist(),
        all_val_preds[val_idx_arr].tolist(),
        args.model, mode, results_dir,
    )
    plot_training_curves(histories, args.model, mode, results_dir)

    print(f"\nSaved to {results_dir}/")
    print(f"  cv_results_{mode}.csv")
    print(f"  summary_{mode}.json")
    print(f"  cv_results_{mode}.png")
    print(f"  training_curves_{mode}.png")
    print(f"  fold*_{mode}_model.pt")

    return {
        "fold_results": fold_results,
        "histories": histories,
        "summary": summary,
        "pool_truths": all_val_labels[val_idx_arr].tolist() if len(val_idx_arr) else [],
        "pool_preds":  all_val_preds[val_idx_arr].tolist()  if len(val_idx_arr) else [],
    }


def _apply_mode_masking(df, mode):
    """Return a (copy of) df with mode-specific has_prame masking applied."""
    if mode == "no_predicted":
        out = df.copy()
        out.loc[out["prame_source"] == PREDICTED_PRAME_SOURCE, "has_prame"] = False
        return out
    return df


def run_compare(df, feat_dim, args, device, results_dir):
    """Run a full N-fold CV (--folds, default 5) for each of the three modes
    and emit a bundled comparison plot + JSON on top of the per-mode CV
    artifacts that run_full_cv already writes.

    The same deterministic patient-level split (seed=42) is used across all
    three modes so per-fold metrics are directly comparable.
    """
    out_dir = results_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[compare] Running {args.folds}-fold CV for each of {len(MODES)} modes")

    per_mode = {}
    for mode in MODES:
        df_mode = _apply_mode_masking(df, mode)
        print(f"\n{'=' * 60}")
        print(f"[compare] mode = {mode}")
        print(f"{'=' * 60}")
        per_mode[mode] = run_full_cv(df_mode, feat_dim, args, mode, device, results_dir)

    # Bundled artifacts on top of the per-mode CV outputs
    plot_compare(per_mode, args.model, args.folds, out_dir)

    summary_json = {}
    for mode in MODES:
        s = per_mode[mode]["summary"]
        per_fold = per_mode[mode]["fold_results"]
        summary_json[mode] = {
            "val_auc_per_fold":     [r["val_auc"] for r in per_fold],
            "val_acc_per_fold":     [r["val_acc"] for r in per_fold],
            "sensitivity_per_fold": [r["sensitivity"] for r in per_fold],
            "specificity_per_fold": [r["specificity"] for r in per_fold],
            "mean_val_auc":     s["mean_auc"],
            "std_val_auc":      s["std_auc"],
            "mean_val_acc":     s["mean_acc"],
            "std_val_acc":      s["std_acc"],
            "mean_sensitivity": s["mean_sensitivity"],
            "std_sensitivity":  s["std_sensitivity"],
            "mean_specificity": s["mean_specificity"],
            "std_specificity":  s["std_specificity"],
            "pooled_auc":       s["pooled_auc"],
        }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Headline table
    print(f"\n{'=' * 60}")
    print(f"COMPARISON SUMMARY ({args.folds}-fold CV per mode)")
    print(f"{'=' * 60}")
    print(f"  {'mode':14s} {'val_auc':>17s} {'val_acc':>17s} "
          f"{'sens':>17s} {'spec':>17s} {'pooled':>8s}")
    for mode in MODES:
        s = summary_json[mode]
        def _fmt(m, sd):
            return f"{m:.3f} +/- {sd:.3f}"
        print(f"  {mode:14s} "
              f"{_fmt(s['mean_val_auc'], s['std_val_auc']):>17s} "
              f"{_fmt(s['mean_val_acc'], s['std_val_acc']):>17s} "
              f"{_fmt(s['mean_sensitivity'], s['std_sensitivity']):>17s} "
              f"{_fmt(s['mean_specificity'], s['std_specificity']):>17s} "
              f"{s['pooled_auc']:>8.3f}")
    print(f"\n[compare] Saved to {out_dir}/")
    print(f"  comparison.json")
    print(f"  compare_variants.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Component-2 PRAME-conditioned diagnostic MIL.",
    )
    p.add_argument("--model", choices=list(FEAT_DIMS.keys()), default="uni",
                   help="Foundation model embeddings (default: uni)")
    p.add_argument("--mode", choices=list(MODES), default="full",
                   help="PRAME ablation mode (default: full)")
    p.add_argument("--manifest",
                   default="data/expression/diagnostic_manifest.csv")
    p.add_argument("--emb-dir", default="embeddings",
                   help="Root for embeddings; cohort subdirs resolved per source_group")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prame-norm",
                   choices=("log", "raw", "zscore_per_source"), default="log",
                   help="PRAME preprocessing (default: log = log1p)")
    p.add_argument("--compare", action="store_true",
                   help="Run all three modes through the full --folds CV and "
                        "emit bundled comparison artifacts on top of per-mode CV")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model.upper()}  (feat_dim={FEAT_DIMS[args.model]})")
    print(f"Mode:   {args.mode if not args.compare else f'compare (all three modes, {args.folds}-fold)'}")
    print(f"PRAME norm: {args.prame_norm}")

    feat_dim = FEAT_DIMS[args.model]

    # For --compare, load manifest in "full" mode (no_predicted masking is applied
    # per-mode inside run_compare).
    load_mode = args.mode if not args.compare else "full"
    df = load_manifest(args.manifest, args.emb_dir, args.model, load_mode, args.prame_norm)

    results_dir = Path(args.results_dir) / args.model / "component2"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        run_compare(df, feat_dim, args, device, results_dir)
    else:
        run_full_cv(df, feat_dim, args, args.mode, device, results_dir)


if __name__ == "__main__":
    main()
