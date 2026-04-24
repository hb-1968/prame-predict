"""
Predict PRAME for COBRA BCC slides using Component 1's trained UNI
fold ensemble.

COBRA slides carry no paired RNA-Seq, so `prame_tpm` cannot be
measured from tissue. Instead we run each slide through all five of
Component 1's trained MIL checkpoints (results/uni/fold{1..5}_model.pt)
and ensemble the probabilities. The resulting `prob_mean` is the
predicted PRAME signal consumed by `08_build_diagnostic_manifest.py`.

Output: data/expression/cobra_prame_predictions.csv, with one row
per predicted slide and columns:

    file_id, file_name, source_group,
    prob_fold1..prob_fold5, prob_mean, prob_std, prob_median,
    pred_label, confidence, n_patches,
    prame_source = "component1_predicted",
    component1_model

Unlike `05_generate_heatmaps.py` (which does held-out fold inference
because its inputs were in the training set), this script applies
all five folds to every slide because COBRA slides were never seen
during Component-1 training.

Usage:
    # Primary: predict on every COBRA embedding in embeddings/uni/
    python 06_predict_cobra_prame.py --no-manifest-filter

    # Filter against an existing manifest (after 08 has run once)
    python 06_predict_cobra_prame.py \\
        --manifest data/expression/diagnostic_manifest.csv \\
        --source-group cobra_bcc

    # Sanity-check the ensemble on 200 SKCM (has measured PRAME)
    python 06_predict_cobra_prame.py \\
        --manifest data/expression/slide_manifest.csv \\
        --source-group-col prame_group \\
        --source-group high \\
        --output data/expression/prame_predictions_skcm_sanity.csv

    # Use CONCH embeddings (not recommended — near-chance in Component 1)
    python 06_predict_cobra_prame.py --model conch --no-manifest-filter
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_FEAT_DIMS = {
    "uni": 1024,
    "conch": 768,
}


# ---------- sibling module import (CLAUDE.md pattern) ----------

def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------- inference helpers ----------

def load_fold_model(weights_path: Path, feat_dim: int, AttentionMIL, device: torch.device):
    model = AttentionMIL(feat_dim=feat_dim)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def read_features(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["features"][:].astype(np.float32)


def predict_slide(models: list, features: np.ndarray, device: torch.device) -> np.ndarray:
    """One probability per fold model."""
    x = torch.from_numpy(features).to(device)
    probs = []
    with torch.inference_mode():
        for m in models:
            logit, _ = m(x)
            probs.append(float(torch.sigmoid(logit).item()))
    return np.asarray(probs, dtype=np.float64)


# ---------- input selection ----------

def _stem(file_name: str) -> str:
    return Path(str(file_name)).stem


def select_slides(
    manifest_path: Path | None,
    source_group_col: str,
    source_group: str | None,
    embeddings_dir: Path,
    no_manifest_filter: bool,
) -> pd.DataFrame:
    """
    Return a DataFrame of (file_id, file_name, source_group, h5_path).

    If --no-manifest-filter is set, iterates every .h5 in the embeddings
    dir with source_group="unknown". Otherwise filters the manifest to
    rows matching `source_group` in `source_group_col`.
    """
    if no_manifest_filter:
        rows = []
        for h5 in sorted(embeddings_dir.glob("*.h5")):
            rows.append({
                "file_id": h5.stem,
                "file_name": h5.stem,
                "source_group": "unknown",
                "h5_path": str(h5),
            })
        return pd.DataFrame(rows)

    if manifest_path is None or not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Pass --no-manifest-filter to predict on every .h5 in the embeddings dir.")
        sys.exit(1)

    mani = pd.read_csv(manifest_path)
    if source_group:
        if source_group_col not in mani.columns:
            print(f"ERROR: manifest has no column '{source_group_col}'. "
                  f"Available: {mani.columns.tolist()}")
            sys.exit(1)
        mani = mani[mani[source_group_col].astype(str) == source_group]

    rows = []
    for _, r in mani.iterrows():
        stem = _stem(r["file_name"])
        h5 = embeddings_dir / f"{stem}.h5"
        if h5.exists():
            rows.append({
                "file_id": r["file_id"],
                "file_name": r["file_name"],
                "source_group": r.get("source_group", source_group or "unknown"),
                "h5_path": str(h5),
            })
    return pd.DataFrame(rows)


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict PRAME for COBRA (or other unlabeled) slides via Component-1 fold ensemble.",
    )
    parser.add_argument("--model", choices=["uni", "conch"], default="uni",
                        help="Foundation model (default: uni; Component 1 showed CONCH at chance)")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="Default: embeddings/<model>/")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Default: results/<model>/  (contains fold*_model.pt)")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/expression/diagnostic_manifest.csv"))
    parser.add_argument("--source-group-col", type=str, default="source_group",
                        help="Manifest column to match (default: source_group)")
    parser.add_argument("--source-group", type=str, default="cobra_bcc",
                        help="Value to match in --source-group-col (default: cobra_bcc)")
    parser.add_argument("--no-manifest-filter", action="store_true",
                        help="Predict on every .h5 in --embeddings-dir (skips manifest)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/expression/cobra_prame_predictions.csv"))
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device (cpu / cuda / mps)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    feat_dim = MODEL_FEAT_DIMS[model_name]
    embeddings_dir = args.embeddings_dir or Path("embeddings") / model_name
    model_dir = args.model_dir or Path("results") / model_name

    if not embeddings_dir.exists():
        print(f"ERROR: embeddings dir not found at {embeddings_dir}")
        return 1
    if not model_dir.exists():
        print(f"ERROR: model dir not found at {model_dir}")
        print("Component-1 training must land first (04_train_mil.py / 04_train_mil_reg.py).")
        return 1

    fold_paths = sorted(model_dir.glob("fold*_model.pt"))
    if len(fold_paths) == 0:
        print(f"ERROR: no fold*_model.pt found in {model_dir}")
        return 1
    print(f"Found {len(fold_paths)} fold checkpoints in {model_dir}")
    for p in fold_paths:
        print(f"  {p.name}")

    print("\nLoading AttentionMIL from 04_train_mil.py...")
    mil_mod = _load_sibling("train_mil", "04_train_mil.py")
    AttentionMIL = mil_mod.AttentionMIL

    slides = select_slides(
        args.manifest, args.source_group_col, args.source_group or None,
        embeddings_dir, args.no_manifest_filter,
    )
    print(f"\nSelected {len(slides)} slides "
          f"(model={model_name}, filter={args.source_group or 'ANY'})")
    if len(slides) == 0:
        print("No slides to predict on. Check:")
        print(f"  - embeddings dir: {embeddings_dir}")
        print(f"  - manifest: {args.manifest}")
        print(f"  - filter: {args.source_group_col}={args.source_group}")
        return 1

    if args.dry_run:
        print("\n[dry-run] No predictions computed.")
        return 0

    device = torch.device(args.device)
    print(f"\nLoading fold models onto {device}...")
    models = [
        load_fold_model(p, feat_dim, AttentionMIL, device)
        for p in fold_paths
    ]
    n_folds = len(models)

    out_rows = []
    for _, s in tqdm(slides.iterrows(), total=len(slides), desc="Predicting"):
        try:
            features = read_features(Path(s["h5_path"]))
        except Exception as e:
            print(f"  [warn] failed to read {s['h5_path']}: {e}")
            continue
        probs = predict_slide(models, features, device)
        mean = float(np.mean(probs))
        std = float(np.std(probs, ddof=0))
        median = float(np.median(probs))
        out_rows.append({
            "file_id": s["file_id"],
            "file_name": s["file_name"],
            "source_group": s["source_group"],
            **{f"prob_fold{i + 1}": float(probs[i]) for i in range(n_folds)},
            "prob_mean": mean,
            "prob_std": std,
            "prob_median": median,
            "pred_label": int(mean >= 0.5),
            "confidence": abs(mean - 0.5) * 2.0,
            "n_patches": int(features.shape[0]),
            "prame_source": "component1_predicted",
            "component1_model": model_name,
        })

    out = pd.DataFrame(out_rows)

    print()
    print("=== Summary ===")
    print(f"Predicted {len(out)} slides using {model_name.upper()} x {n_folds} folds")
    if len(out):
        print(f"  prob_mean     min={out['prob_mean'].min():.3f}  "
              f"median={out['prob_mean'].median():.3f}  "
              f"max={out['prob_mean'].max():.3f}")
        print(f"  confidence    median={out['confidence'].median():.3f}  "
              f"q25={out['confidence'].quantile(0.25):.3f}  "
              f"q75={out['confidence'].quantile(0.75):.3f}")
        hi = int((out["pred_label"] == 1).sum())
        lo = int((out["pred_label"] == 0).sum())
        print(f"  pred_label    high:{hi}  low:{lo}")
        ambiguous = int((out["confidence"] < 0.2).sum())
        print(f"  indeterminate (confidence<0.2): {ambiguous} ({ambiguous / len(out):.1%})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
