"""
Render attention-weighted heatmaps over WSIs for the trained MIL models.

For each target slide, the script:
1. Identifies the held-out CV fold by replaying StratifiedGroupKFold(seed=42)
   so attention is only ever produced by a model that did NOT train on that slide.
2. Loads patch-level features + level-0 coordinates from the slide's .h5 embedding.
3. Runs the held-out fold's AttentionMIL to capture (logit, attention).
4. Re-downloads the .svs from GDC (deleted after feature extraction), reads a
   thumbnail, paints each patch's bounding box with its attention weight, and
   saves a PNG to results/{model}/heatmaps/.

Default mode (no --slide) computes out-of-fold predictions for every slide,
classifies each as TP/TN/FP/FN at threshold 0.5, and renders the top-N per
class by |logit| (default N=2 -> 8 figures).

Usage:
    python 05_generate_heatmaps.py                                   # 8 curated heatmaps
    python 05_generate_heatmaps.py --slide TCGA-HR-A5NC-01Z-00-DX1.F8577EB3-C670-4DDB-9EDD-B8681CE2D664
    python 05_generate_heatmaps.py --n-per-class 3                   # 12 curated heatmaps
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle


GDC_API = "https://api.gdc.cancer.gov"
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Import helpers from numerically-prefixed sibling scripts.
# Numeric prefixes (e.g. "04_train_mil") aren't valid Python identifiers, so
# the standard `import 04_train_mil` syntax cannot be used. We load the modules
# from their file paths via importlib instead.
# ---------------------------------------------------------------------------

def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_openslide():
    try:
        import openslide  # noqa: F401
        return openslide
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Failed to import openslide. On Windows, ensure the OpenSlide DLLs "
            "are on PATH (see CLAUDE.md Environment section)."
        ) from exc


# ---------------------------------------------------------------------------
# CV fold replay
# ---------------------------------------------------------------------------

def stem_from_filename(file_name: str) -> str:
    """e.g. 'TCGA-XX-XXXX-...DX1.UUID.svs' -> stem without extension."""
    return Path(file_name).stem


def build_cv_fold_map(manifest: pd.DataFrame, emb_dir: Path,
                      seed: int = 42, n_splits: int = 5):
    """
    Replay the exact StratifiedGroupKFold split used by 04_train_mil.py to
    map each slide stem -> the fold index where it was held out for validation.

    Returns:
        fold_map: dict[stem] = fold_idx in [0, n_splits-1]
        slides_in_order: list[str] of h5 paths (ordered as 04 sees them)
        labels_in_order: np.ndarray of binary labels
        patients_in_order: np.ndarray of submitter_ids
    """
    slides, labels, patients, stems = [], [], [], []
    for _, row in manifest.iterrows():
        h5_path = emb_dir / row["file_name"].replace(".svs", ".h5")
        if h5_path.exists():
            slides.append(str(h5_path))
            labels.append(int(row["prame_label"]))
            patients.append(row["submitter_id"])
            stems.append(stem_from_filename(row["file_name"]))

    slides_arr = np.array(slides)
    labels_arr = np.array(labels)
    patients_arr = np.array(patients)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(
            skf.split(slides_arr, labels_arr, groups=patients_arr)):
        for i in val_idx:
            fold_map[stems[i]] = fold_idx

    return fold_map, slides_arr, labels_arr, patients_arr, stems


# ---------------------------------------------------------------------------
# Inference: load fold model, compute attention for one slide
# ---------------------------------------------------------------------------

def load_fold_model(weights_path: Path, feat_dim: int, AttentionMIL, device):
    model = AttentionMIL(feat_dim=feat_dim)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def compute_attention(h5_path: Path, model, device):
    """
    Returns:
        coords: (N, 2) int array of level-0 top-left pixel coords
        attention: (N,) float softmax-normalized attention weights
        prob: float sigmoid(logit)
    """
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:].astype(np.float32)
        coords = f["coords"][:]

    x = torch.from_numpy(features).to(device)
    with torch.no_grad():
        logit, attention = model(x)

    return coords, attention.detach().cpu().numpy(), float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# OOF predictions across all slides, then curated TP/TN/FP/FN selection
# ---------------------------------------------------------------------------

def compute_oof_predictions(manifest, fold_map, emb_dir, results_dir,
                            feat_dim, AttentionMIL, device):
    """
    For each fold, load that fold's checkpoint and run inference on the slides
    held out for that fold. Returns a DataFrame with one row per slide:
      stem, file_id, file_name, label, prob, logit_signed, outcome.
    """
    rows_by_fold = {fold: [] for fold in range(5)}
    for _, row in manifest.iterrows():
        stem = stem_from_filename(row["file_name"])
        fold = fold_map.get(stem)
        if fold is None:
            continue
        h5_path = emb_dir / (stem + ".h5")
        if not h5_path.exists():
            continue
        rows_by_fold[fold].append({
            "stem": stem,
            "file_id": row["file_id"],
            "file_name": row["file_name"],
            "label": int(row["prame_label"]),
            "h5_path": h5_path,
        })

    out_rows = []
    for fold, slides in rows_by_fold.items():
        if not slides:
            continue
        weights_path = results_dir / f"fold{fold + 1}_model.pt"
        if not weights_path.exists():
            print(f"  [warn] missing {weights_path}, skipping fold")
            continue
        model = load_fold_model(weights_path, feat_dim, AttentionMIL, device)
        for entry in tqdm(slides, desc=f"OOF inference fold {fold + 1}", leave=False):
            with h5py.File(entry["h5_path"], "r") as f:
                features = f["features"][:].astype(np.float32)
            x = torch.from_numpy(features).to(device)
            with torch.no_grad():
                logit, _ = model(x)
            logit_val = float(logit.item())
            prob = float(torch.sigmoid(logit).item())
            pred = 1 if prob >= 0.5 else 0
            label = entry["label"]
            if pred == 1 and label == 1:
                outcome = "TP"
            elif pred == 0 and label == 0:
                outcome = "TN"
            elif pred == 1 and label == 0:
                outcome = "FP"
            else:
                outcome = "FN"
            out_rows.append({
                "stem": entry["stem"],
                "file_id": entry["file_id"],
                "file_name": entry["file_name"],
                "fold": fold,
                "label": label,
                "logit": logit_val,
                "prob": prob,
                "outcome": outcome,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(out_rows)


def select_curated_slides(oof_df: pd.DataFrame, n_per_class: int = 2):
    """Return up to n_per_class slides per outcome class, ranked by |logit|."""
    picks = []
    for outcome in ("TP", "TN", "FP", "FN"):
        subset = oof_df[oof_df["outcome"] == outcome].copy()
        if subset.empty:
            continue
        subset["abs_logit"] = subset["logit"].abs()
        subset = subset.sort_values("abs_logit", ascending=False).head(n_per_class)
        picks.append(subset)
    if not picks:
        return pd.DataFrame(columns=oof_df.columns)
    return pd.concat(picks, ignore_index=True)


# ---------------------------------------------------------------------------
# WSI re-download from GDC
# ---------------------------------------------------------------------------

def download_wsi(file_id: str, out_path: Path, session: requests.Session,
                 chunk_size: int = 1 << 20) -> Path:
    """Stream a single .svs from GDC. Raises HTTPError on failure."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GDC_API}/data/{file_id}"
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"  download {out_path.name[:40]}", leave=False) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def pick_thumbnail_level(slide, max_px: int):
    """Return (level, downsample) for the smallest pyramid level whose
    largest dimension is <= max_px. Falls back to the highest available level."""
    for level in range(slide.level_count):
        w, h = slide.level_dimensions[level]
        if max(w, h) <= max_px:
            return level, slide.level_downsamples[level]
    last = slide.level_count - 1
    return last, slide.level_downsamples[last]


def render_heatmap(svs_path: Path, coords: np.ndarray, attention: np.ndarray,
                   out_png: Path, meta: dict, get_20x_level,
                   thumbnail_max_px: int = 4096, alpha: float = 0.5,
                   cmap_name: str = "viridis"):
    """Composite an attention overlay on a slide thumbnail and save as PNG."""
    openslide = _import_openslide()
    slide = openslide.OpenSlide(str(svs_path))
    try:
        _, level_scale = get_20x_level(slide)
        box_lvl0 = int(round(224 * level_scale))

        level, downsample = pick_thumbnail_level(slide, thumbnail_max_px)
        thumb_w, thumb_h = slide.level_dimensions[level]
        thumb = slide.read_region((0, 0), level, (thumb_w, thumb_h)).convert("RGB")
        thumb_arr = np.asarray(thumb)

        # Scale level-0 coords/box into the thumbnail level coordinate system.
        coords_scaled = coords / downsample
        box_scaled = box_lvl0 / downsample

        # Rescale attention for visual contrast: softmax attention over ~30k
        # patches is ~1/N (~3e-5), so percentile-normalize to [0, 1].
        attn = np.asarray(attention, dtype=np.float64)
        if attn.size > 1 and attn.max() > attn.min():
            lo, hi = np.percentile(attn, [1, 99])
            if hi <= lo:
                lo, hi = float(attn.min()), float(attn.max())
            attn_norm = np.clip((attn - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        else:
            attn_norm = np.zeros_like(attn)

        cmap = cm.get_cmap(cmap_name)

        fig_w = max(8.0, thumb_w / 250)
        fig_h = max(8.0, thumb_h / 250)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(thumb_arr)
        ax.set_xlim(0, thumb_w)
        ax.set_ylim(thumb_h, 0)
        ax.set_axis_off()

        for (x, y), a in zip(coords_scaled, attn_norm):
            color = cmap(float(a))
            rect = Rectangle(
                (float(x), float(y)),
                float(box_scaled), float(box_scaled),
                linewidth=0, facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)

        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("attention (1/99 percentile-normalized)")

        title = (
            f"{meta['stem'][:60]}\n"
            f"label={'high' if meta['label'] == 1 else 'low'}  "
            f"prob={meta['prob']:.3f}  outcome={meta['outcome']}  "
            f"fold={meta['fold'] + 1}  N={len(coords)} patches"
        )
        ax.set_title(title, fontsize=10)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    finally:
        slide.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["uni", "conch"], default="uni")
    parser.add_argument("--manifest", default="data/expression/slide_manifest.csv")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None,
                        help="default: results/{model}/heatmaps")
    parser.add_argument("--wsi-cache-dir", default="data/wsi")
    parser.add_argument("--slide", default=None,
                        help="render a single slide (stem or .svs file_name); "
                             "skips OOF + curated selection")
    parser.add_argument("--n-per-class", type=int, default=2,
                        help="curated mode: top-N per TP/TN/FP/FN class")
    parser.add_argument("--keep-wsi", action="store_true",
                        help="do not delete the .svs after rendering")
    parser.add_argument("--thumbnail-max-px", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train_mil = _load_sibling("train_mil_helpers", "04_train_mil.py")
    tile_wsi = _load_sibling("tile_wsi_helpers", "02_tile_wsi.py")
    AttentionMIL = train_mil.AttentionMIL
    get_20x_level = tile_wsi.get_20x_level

    feat_dim = {"uni": 1024, "conch": 768}[args.model]
    device = resolve_device(args.device)
    manifest_path = Path(args.manifest)
    emb_dir = Path(args.emb_dir) / args.model
    results_dir = Path(args.results_dir) / args.model
    wsi_cache_dir = Path(args.wsi_cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    fold_map, _, _, _, _ = build_cv_fold_map(
        manifest, emb_dir, seed=args.seed, n_splits=args.folds,
    )

    # --- Decide which slides to render -----------------------------------
    if args.slide:
        target_stem = stem_from_filename(args.slide) if args.slide.endswith(".svs") \
            else args.slide
        match = manifest[manifest["file_name"].apply(stem_from_filename) == target_stem]
        if match.empty:
            raise SystemExit(f"slide '{args.slide}' not found in manifest")
        row = match.iloc[0]
        fold = fold_map.get(target_stem)
        if fold is None:
            raise SystemExit(f"slide '{target_stem}' not in any held-out fold "
                             f"(missing embedding?)")
        weights_path = results_dir / f"fold{fold + 1}_model.pt"
        model = load_fold_model(weights_path, feat_dim, AttentionMIL, device)
        h5_path = emb_dir / (target_stem + ".h5")
        coords, attention, prob = compute_attention(h5_path, model, device)
        label = int(row["prame_label"])
        pred = 1 if prob >= 0.5 else 0
        outcome = ["TN", "FN", "FP", "TP"][2 * label + pred]
        targets = [{
            "stem": target_stem,
            "file_id": row["file_id"],
            "label": label,
            "fold": fold,
            "prob": prob,
            "outcome": outcome,
            "coords": coords,
            "attention": attention,
        }]
    else:
        print("Computing out-of-fold predictions for curated selection...")
        oof_df = compute_oof_predictions(
            manifest, fold_map, emb_dir, results_dir, feat_dim,
            AttentionMIL, device,
        )
        oof_csv = out_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_csv, index=False)
        print(f"OOF predictions: {oof_csv}")
        curated = select_curated_slides(oof_df, n_per_class=args.n_per_class)
        print(f"Curated selection ({len(curated)} slides):")
        for _, r in curated.iterrows():
            print(f"  {r['outcome']:>2}  prob={r['prob']:.3f}  "
                  f"label={r['label']}  fold={r['fold'] + 1}  {r['stem']}")
        targets = []
        for _, r in curated.iterrows():
            weights_path = results_dir / f"fold{int(r['fold']) + 1}_model.pt"
            model = load_fold_model(weights_path, feat_dim, AttentionMIL, device)
            h5_path = emb_dir / (r["stem"] + ".h5")
            coords, attention, _ = compute_attention(h5_path, model, device)
            targets.append({
                "stem": r["stem"],
                "file_id": r["file_id"],
                "label": int(r["label"]),
                "fold": int(r["fold"]),
                "prob": float(r["prob"]),
                "outcome": r["outcome"],
                "coords": coords,
                "attention": attention,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- Download, render, cleanup ---------------------------------------
    session = requests.Session()
    for t in targets:
        svs_path = wsi_cache_dir / (t["stem"] + ".svs")
        try:
            download_wsi(t["file_id"], svs_path, session)
        except requests.HTTPError as exc:
            print(f"  [skip] GDC download failed for {t['stem']}: {exc}")
            continue

        out_png = out_dir / f"{t['outcome']}_{t['stem']}.png"
        try:
            render_heatmap(
                svs_path, t["coords"], t["attention"], out_png,
                meta={
                    "stem": t["stem"], "label": t["label"], "prob": t["prob"],
                    "outcome": t["outcome"], "fold": t["fold"],
                },
                get_20x_level=get_20x_level,
                thumbnail_max_px=args.thumbnail_max_px,
                alpha=args.alpha, cmap_name=args.cmap,
            )
            print(f"  wrote {out_png}")
        finally:
            if not args.keep_wsi and svs_path.exists():
                try:
                    svs_path.unlink()
                except OSError:
                    pass

    if not args.keep_wsi:
        # Best-effort cleanup of empty cache directory.
        try:
            if wsi_cache_dir.exists() and not any(wsi_cache_dir.iterdir()):
                shutil.rmtree(wsi_cache_dir)
        except OSError:
            pass

    print(f"\nDone. Heatmaps in {out_dir}")


if __name__ == "__main__":
    main()
