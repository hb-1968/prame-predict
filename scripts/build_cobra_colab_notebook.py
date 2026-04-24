"""One-shot builder for notebooks/cobra_predict_colab.ipynb.

Kept as a tracked script so the notebook can be regenerated / diffed
without hand-editing JSON.
"""

import json
from pathlib import Path

cells = []


def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    })


def md(src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    })


md("""# COBRA PRAME Prediction Pipeline (Colab)

End-to-end workflow for producing `cobra_prame_predictions.csv`:

1. **Inventory** — list `s3://cobra-pathology/` (unsigned), filter to BCC, stratified pick of N slides.
2. **Per-slide** — download WSI from S3, tile in-memory at 20x (via `02_tile_wsi.py`), extract UNI features, save `.h5` to Drive, delete local files.
3. **Predict** — run Component 1's 5 UNI fold checkpoints as an ensemble over every saved embedding (via `06_predict_cobra_prame.py` helpers). Output predictions CSV to Drive.

This notebook mirrors `notebooks/prame_predict.ipynb` (the Component-1 feature-extraction pipeline) but retargets the download source (S3 instead of GDC), adds the prediction stage, and runs UNI only — Component 1 showed CONCH near-chance on PRAME, so CONCH is excluded here.

**Prerequisites**
- Hugging Face access to `MahmoodLab/uni` (for the foundation model).
- Component 1's 5 fold UNI checkpoints on Drive at `prame-predict/results/uni/fold{1..5}_model.pt`.
- Google Drive mounted with write access.

**Runtime** — recommend T4 or L4 GPU. ~5-8 min per slide for tile + extract (depends on patch count); prediction is fast once embeddings exist.
""")

code("""# Cell 1: Install dependencies, mount Drive, clone repo
!pip install -q timm huggingface_hub openslide-python h5py opencv-python-headless boto3
!apt-get install -qq -y openslide-tools

from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('prame-predict'):
    !git clone https://github.com/hb-1968/prame-predict.git
else:
    !cd prame-predict && git pull --ff-only""")

code("""# Cell 2: HuggingFace login (UNI access)
from huggingface_hub import login
login()""")

md("""## Imports + Module Loading

Load helpers from the repo's numbered scripts via `SourceFileLoader`. `02_tile_wsi.py` supplies the in-memory tiler; `04_train_mil.py` supplies the `AttentionMIL` architecture; `06_predict_cobra_prame.py` supplies the fold-ensemble inference primitives.""")

code("""# Cell 3: Imports and sibling module loading
import shutil
import threading
import queue
import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')
import timm
import h5py
import openslide
from pathlib import Path
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

import boto3
from botocore import UNSIGNED
from botocore.config import Config

tile_mod = SourceFileLoader("tile", "prame-predict/02_tile_wsi.py").load_module()
tile_slide = tile_mod.tile_slide
_close_thread_handles = tile_mod._close_thread_handles

mil_mod = SourceFileLoader("mil", "prame-predict/04_train_mil.py").load_module()
AttentionMIL = mil_mod.AttentionMIL

cobra_mod = SourceFileLoader("cobra", "prame-predict/06_predict_cobra_prame.py").load_module()
load_fold_model = cobra_mod.load_fold_model
predict_slide = cobra_mod.predict_slide
read_features = cobra_mod.read_features
MODEL_FEAT_DIMS = cobra_mod.MODEL_FEAT_DIMS""")

md("""## Configuration""")

code("""# Cell 4: Paths, constants, device
DRIVE_ROOT = Path("/content/drive/MyDrive/prame-predict")
EMB_DIR    = DRIVE_ROOT / "embeddings" / "uni_cobra"      # keep COBRA separate from SKCM embeddings
FOLD_DIR   = DRIVE_ROOT / "results" / "uni"               # Component-1 fold checkpoints
OUT_CSV    = DRIVE_ROOT / "data" / "expression" / "cobra_prame_predictions.csv"

LOCAL_WSI   = Path("/content/wsi_batch")
LOCAL_TILES = Path("/content/tiles_tmp")
for d in (EMB_DIR, FOLD_DIR, OUT_CSV.parent, LOCAL_WSI, LOCAL_TILES):
    d.mkdir(parents=True, exist_ok=True)

# Inventory / tiling / extraction
N_COBRA_SLIDES   = 115          # Plan E target; raise if you want more
MAX_PATCHES      = 80000        # random sample cap per slide
BATCH_SIZE_GPU   = 512          # patches per forward pass
TILE_WORKERS     = 16
SEED             = 42

# S3
COBRA_BUCKET = "cobra-pathology"
COBRA_REGION = "us-west-2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Drive embeddings dir: {EMB_DIR}")
print(f"Fold checkpoints dir: {FOLD_DIR}")""")

md("""## Stage 1 - COBRA Inventory from S3

List the open bucket, filter to WSIs, classify by BCC subtype via path/name heuristic, reject the risky-cancer and SCC buckets (to avoid melanoma contamination or SCC leaking into BCC negatives), then stratified-pick across subtypes.""")

code("""# Cell 5: S3 inventory + BCC subtype filter
s3 = boto3.client(
    "s3",
    region_name=COBRA_REGION,
    config=Config(signature_version=UNSIGNED),
)

print(f"Listing s3://{COBRA_BUCKET}/ (unsigned)...")
paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket=COBRA_BUCKET):
    for obj in page.get("Contents", []):
        keys.append({"key": obj["Key"], "size_gb": obj["Size"] / (1024 ** 3)})
print(f"  {len(keys)} objects")

WSI_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif")
wsis = [k for k in keys if k["key"].lower().endswith(WSI_EXTS)]
print(f"  {len(wsis)} WSI files")

BCC_SUBTYPES = (
    "ia", "ib", "ii", "iii",
    "nodular", "superficial", "medium", "high", "aggressive",
)
RISKY_KEYWORDS = ("risky", "melanoma", "merkel")
SCC_KEYWORDS   = ("scc", "squamous")

def classify(key):
    k = key.lower()
    if any(r in k for r in RISKY_KEYWORDS + SCC_KEYWORDS):
        return None
    for sub in BCC_SUBTYPES:
        if f"/{sub}/" in k or f"_{sub}_" in k or f"_{sub}." in k or f"-{sub}-" in k:
            return sub
    if "bcc" in k:
        return "bcc_unspecified"
    return None

classified = [{**w, "subtype": s} for w in wsis
              if (s := classify(w["key"])) is not None]
print(f"  {len(classified)} BCC WSIs after risky + SCC filter")

if not classified:
    raise RuntimeError(
        "Heuristic classifier matched nothing. Inspect the bucket layout:\\n"
        f"  aws s3 ls --no-sign-request --recursive s3://{COBRA_BUCKET}/ | head -50"
    )

_all = pd.DataFrame(classified)
rng = np.random.default_rng(SEED)
per_sub = max(1, N_COBRA_SLIDES // _all["subtype"].nunique())
picks = []
for sub, group in _all.groupby("subtype"):
    if len(group) <= per_sub:
        picks.append(group)
    else:
        idx = rng.choice(len(group), size=per_sub, replace=False)
        picks.append(group.iloc[idx])
inventory = pd.concat(picks).head(N_COBRA_SLIDES).reset_index(drop=True)

print()
print(f"Selected {len(inventory)} BCC slides:")
print(inventory["subtype"].value_counts().to_string())
print(f"Total download size: {inventory['size_gb'].sum():.1f} GB")
inventory.head()""")

md("""## Stage 2 - Load UNI + Define Extractor

Load UNI at float16 with `torch.compile`. `extract_features` auto-switches between GPU-resident mode (all patches preloaded to VRAM, fastest) and a CPU-side prefetcher fallback when the slide doesn't fit.""")

code("""# Cell 6: UNI loader + extract_features + save_h5
def load_uni():
    from huggingface_hub import hf_hub_download
    model = timm.create_model(
        "vit_large_patch16_224", init_values=1e-5,
        num_classes=0, pretrained=False,
    )
    ckpt = hf_hub_download(repo_id="MahmoodLab/uni", filename="pytorch_model.bin")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def preprocess_batch(batch_np):
    batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    batch = batch.to(dtype=torch.float16).div_(127.5).sub_(1.0)
    return batch.to(device, non_blocking=True)


class BatchPrefetcher:
    def __init__(self, patches, batch_size):
        self._patches = patches
        self._bs = batch_size
        self._n = len(patches)
        self._q = queue.Queue(maxsize=2)
        threading.Thread(target=self._produce, daemon=True).start()

    def _produce(self):
        for i in range(0, self._n, self._bs):
            self._q.put(preprocess_batch(np.array(self._patches[i:i + self._bs])))
        self._q.put(None)

    def __iter__(self):
        while (b := self._q.get()) is not None:
            yield b


def extract_features(patches, model):
    n = len(patches)
    num_batches = (n + BATCH_SIZE_GPU - 1) // BATCH_SIZE_GPU
    features = None
    write_idx = 0

    torch.cuda.empty_cache()
    patch_bytes = n * 3 * 224 * 224 * 2
    gpu_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    use_gpu_resident = (patch_bytes * 1.5) < (gpu_free - 2 * 1024 ** 3)
    mode = "GPU-resident" if use_gpu_resident else "prefetcher"
    print(f"    {mode}: {patch_bytes / 1024 ** 3:.1f} GB patches / {gpu_free / 1024 ** 3:.1f} GB free")

    with torch.inference_mode():
        if use_gpu_resident:
            all_patches = torch.from_numpy(np.array(patches)).permute(0, 3, 1, 2)
            all_patches = all_patches.to(device=device, dtype=torch.float16).div_(127.5).sub_(1.0)
            for i in tqdm(range(0, n, BATCH_SIZE_GPU),
                          total=num_batches, desc="    uni", leave=False):
                with torch.amp.autocast("cuda"):
                    feats = model(all_patches[i:i + BATCH_SIZE_GPU].float())
                feats_np = feats.half().cpu().numpy()
                if features is None:
                    features = np.empty((n, feats_np.shape[1]), dtype=np.float16)
                features[write_idx:write_idx + len(feats_np)] = feats_np
                write_idx += len(feats_np)
            del all_patches
            torch.cuda.empty_cache()
        else:
            for batch in tqdm(BatchPrefetcher(patches, BATCH_SIZE_GPU),
                              total=num_batches, desc="    uni", leave=False):
                with torch.amp.autocast("cuda"):
                    feats = model(batch.float())
                feats_np = feats.half().cpu().numpy()
                if features is None:
                    features = np.empty((n, feats_np.shape[1]), dtype=np.float16)
                features[write_idx:write_idx + len(feats_np)] = feats_np
                write_idx += len(feats_np)
    return features


def save_h5(path, features, coords, slide_name):
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        f.create_dataset("coords",   data=coords,   compression="gzip", compression_opts=4)
        f.attrs["model"] = "uni"
        f.attrs["slide_name"] = slide_name
        f.attrs["num_patches"] = features.shape[0]
        f.attrs["feature_dim"] = features.shape[1]


print("Loading UNI...")
uni_model = load_uni()
uni_model.eval().to(device)
uni_model = torch.compile(uni_model)
print("UNI ready.")""")

md("""## Stage 3 - Download + Tile + Extract per Slide

One slide at a time keeps `/content` disk bounded. Embeddings accumulate on Drive under `embeddings/uni_cobra/`. Resumable - skips any slide that already has a `.h5` on Drive.""")

code("""# Cell 7: Per-slide pipeline
def download_s3_slide(key, local_path):
    if local_path.exists():
        return local_path
    s3.download_file(COBRA_BUCKET, key, str(local_path))
    return local_path


# Filter inventory to slides we haven't already embedded
remaining = []
for _, row in inventory.iterrows():
    slide_name = Path(row["key"]).name
    emb_path = EMB_DIR / (Path(slide_name).stem + ".h5")
    if not emb_path.exists():
        remaining.append(row)
remaining = pd.DataFrame(remaining).reset_index(drop=True) if remaining else pd.DataFrame(columns=inventory.columns)
print(f"Slides to process: {len(remaining)} / {len(inventory)}")

for _, row in remaining.iterrows():
    key = row["key"]
    subtype = row["subtype"]
    slide_name = Path(key).name
    local_path = LOCAL_WSI / slide_name
    emb_path = EMB_DIR / (Path(slide_name).stem + ".h5")
    slide_out = LOCAL_TILES / Path(slide_name).stem

    if emb_path.exists():
        continue

    print(f"\\n[{subtype}] {slide_name}  ({row['size_gb']:.2f} GB)")
    try:
        download_s3_slide(key, local_path)
    except Exception as e:
        print(f"  download failed: {type(e).__name__}: {e}")
        continue

    try:
        num_patches, coords, patches = tile_slide(
            local_path, slide_out,
            workers=TILE_WORKERS, max_patches=MAX_PATCHES, in_memory=True,
        )
        _close_thread_handles()
    except Exception as e:
        print(f"  tile failed: {type(e).__name__}: {e}")
        local_path.unlink(missing_ok=True)
        continue

    if num_patches == 0:
        print("  0 patches; skipping")
        local_path.unlink(missing_ok=True)
        if slide_out.exists():
            shutil.rmtree(slide_out)
        continue

    try:
        features = extract_features(patches, uni_model)
        del patches
    except Exception as e:
        print(f"  extract failed: {type(e).__name__}: {e}")
        local_path.unlink(missing_ok=True)
        if slide_out.exists():
            shutil.rmtree(slide_out)
        continue

    save_h5(emb_path, features, np.array(coords), slide_name)
    print(f"  saved {features.shape}")
    del features

    local_path.unlink(missing_ok=True)
    if slide_out.exists():
        shutil.rmtree(slide_out)

# Final cleanup
for f in LOCAL_WSI.glob("*"):
    if f.is_file():
        f.unlink()
for d in LOCAL_TILES.iterdir():
    if d.is_dir():
        shutil.rmtree(d)

# Free the extractor before the prediction stage
del uni_model
torch.cuda.empty_cache()
print("\\nTile + extract stage complete.")""")

md("""## Stage 4 - PRAME Prediction via Component-1 Ensemble

Load the 5 UNI fold checkpoints from Drive (they're gitignored so they don't live in the cloned repo). Ensemble all five over every `.h5` now in `EMB_DIR`. Mirrors `06_predict_cobra_prame.py --no-manifest-filter`.""")

code("""# Cell 8: Run 5-fold ensemble predictions
fold_paths = sorted(FOLD_DIR.glob("fold*_model.pt"))
assert len(fold_paths) == 5, (
    f"Expected 5 fold checkpoints at {FOLD_DIR}, got {len(fold_paths)}. "
    "Re-run train_mil_colab.ipynb or copy results/uni/fold*_model.pt to Drive."
)
print(f"Loading {len(fold_paths)} fold checkpoints from {FOLD_DIR}")

feat_dim = MODEL_FEAT_DIMS["uni"]
fold_models = [load_fold_model(p, feat_dim, AttentionMIL, device) for p in fold_paths]
n_folds = len(fold_models)

embeddings = sorted(EMB_DIR.glob("*.h5"))
print(f"Predicting on {len(embeddings)} slides")

rows = []
for h5_path in tqdm(embeddings, desc="Predicting"):
    try:
        features = read_features(h5_path)
    except Exception as e:
        print(f"  [warn] {h5_path.name}: {e}")
        continue
    probs = predict_slide(fold_models, features, device)
    mean = float(np.mean(probs))
    std = float(np.std(probs, ddof=0))
    rows.append({
        "file_id":    h5_path.stem,
        "file_name":  h5_path.stem + ".svs",
        "source_group": "cobra_bcc",
        **{f"prob_fold{i+1}": float(probs[i]) for i in range(n_folds)},
        "prob_mean":    mean,
        "prob_std":     std,
        "prob_median":  float(np.median(probs)),
        "pred_label":   int(mean >= 0.5),
        "confidence":   abs(mean - 0.5) * 2.0,
        "n_patches":    int(features.shape[0]),
        "prame_source": "component1_predicted",
        "component1_model": "uni",
    })

predictions = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUT_CSV, index=False)
print(f"\\nWrote {len(predictions)} rows to {OUT_CSV}")""")

md("""## Summary""")

code("""# Cell 9: Distribution stats for downstream QC
if len(predictions) == 0:
    print("No predictions produced.")
else:
    print(f"Predicted {len(predictions)} COBRA slides.")
    print()
    print("prob_mean distribution:")
    print(f"  min    = {predictions['prob_mean'].min():.3f}")
    print(f"  q25    = {predictions['prob_mean'].quantile(0.25):.3f}")
    print(f"  median = {predictions['prob_mean'].median():.3f}")
    print(f"  q75    = {predictions['prob_mean'].quantile(0.75):.3f}")
    print(f"  max    = {predictions['prob_mean'].max():.3f}")
    print()
    print("confidence distribution:")
    print(f"  q25    = {predictions['confidence'].quantile(0.25):.3f}")
    print(f"  median = {predictions['confidence'].median():.3f}")
    print(f"  q75    = {predictions['confidence'].quantile(0.75):.3f}")
    ambig = int((predictions['confidence'] < 0.2).sum())
    print(f"  indeterminate (<0.2): {ambig} ({ambig / len(predictions):.1%})")
    print()
    print("pred_label:")
    print(f"  high (>=0.5): {int((predictions['pred_label'] == 1).sum())}")
    print(f"  low  (< 0.5): {int((predictions['pred_label'] == 0).sum())}")
    print()
    print("Feed to 08_build_diagnostic_manifest.py via")
    print(f"  --cobra-predictions {OUT_CSV}")""")


nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("notebooks/cobra_predict_colab.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size} bytes, {len(cells)} cells)")
