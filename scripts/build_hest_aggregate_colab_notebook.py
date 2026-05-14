"""One-shot builder for notebooks/hest_aggregate_colab.ipynb.

Kept as a tracked script so the notebook can be regenerated / diffed
without hand-editing JSON. Mirrors the structure of
`build_cobra_colab_notebook.py`.
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


md("""# HEST-1k PRAME Pseudobulk Aggregation (Colab)

End-to-end workflow for producing `hest_prame_aggregate.csv` (the
per-slide pseudobulk PRAME table that feeds Component-2 manifest
assembly in `08_build_diagnostic_manifest.py`).

1. **Metadata** - fetch the HEST-1k metadata CSV from HuggingFace, filter to non-melanoma skin slides.
2. **Per-slide** - download each slide's `.h5ad` from `MahmoodLab/hest`, restrict to in-tissue spots, sum raw counts across spots, normalize to per-million (pseudobulk CPM), extract the PRAME row, delete the local `.h5ad`.
3. **Output** - write `hest_prame_aggregate.csv` to Drive at the path 08 expects.

This notebook wraps `07_aggregate_hest_prame.py` for Colab execution. It reuses the same private helpers (`_download_metadata`, `_filter_skin_nonmelanoma`, `_pseudobulk_prame`) via `SourceFileLoader` so the logic stays in lockstep with the script. The Colab-specific additions are:

- **Disk hygiene** - each `.h5ad` is deleted from the local HF cache after pseudobulking. Visium objects can be hundreds of MB to several GB; without cleanup, Colab `/content` fills up after a few dozen slides.
- **Resumability** - the output CSV is rewritten after every slide. On reconnect, already-processed `file_id`s are skipped.
- **CPU pinning** - pseudobulk is pure CPU work (sparse matrix sum + normalize). No compute units are burned even if a GPU runtime is attached.

**Prerequisites**
- HuggingFace access to `MahmoodLab/hest` (gated dataset). Request at https://huggingface.co/datasets/MahmoodLab/hest and wait for approval before running cell 5. Without approval, the metadata fetch will fail and the notebook will hard-stop with the access URL.
- Google Drive mounted with write access.

**Runtime** - CPU runtime is fine (and cheaper). Wall-clock is dominated by HF download bandwidth, not compute; expect roughly 1-3 min per slide depending on Visium object size and Colab's link to HF.
""")

code("""# Cell 1: Install dependencies, mount Drive, clone repo
!pip install -q scanpy anndata huggingface_hub

from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('prame-predict'):
    !git clone https://github.com/hb-1968/prame-predict.git
else:
    !cd prame-predict && git pull --ff-only""")

code("""# Cell 2: HuggingFace login (HEST access required)
# If you have not yet been approved for MahmoodLab/hest, request access at:
#   https://huggingface.co/datasets/MahmoodLab/hest
# Then run this cell with a token that has read access to the gated dataset.
from huggingface_hub import login
login()""")

md("""## Imports + Module Loading

Load the helpers from `07_aggregate_hest_prame.py` via `SourceFileLoader`. The script defines: `_download_metadata` (tries the known HEST metadata CSV filenames in order), `_filter_skin_nonmelanoma` (organ + disease column heuristics), `_find_prame_row` (resolves PRAME via gene symbol or Ensembl ID), and `_pseudobulk_prame` (in-tissue mask + per-gene sum + CPM normalization).""")

code("""# Cell 3: Imports and sibling module loading
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

import anndata
from huggingface_hub import hf_hub_download

hest_mod = SourceFileLoader("hest_aggregate", "prame-predict/07_aggregate_hest_prame.py").load_module()
_download_metadata      = hest_mod._download_metadata
_filter_skin_nonmelanoma = hest_mod._filter_skin_nonmelanoma
_pseudobulk_prame       = hest_mod._pseudobulk_prame
HEST_REPO_ID            = hest_mod.HEST_REPO_ID
HEST_ST_SUBDIR          = hest_mod.HEST_ST_SUBDIR""")

md("""## Configuration

`META_CACHE` is kept across the run (metadata is reused). `SLIDE_CACHE` is wiped after each slide so per-slide `.h5ad` blobs do not accumulate. `OUT_CSV` lands at the same path `08_build_diagnostic_manifest.py` reads by default (`--hest-aggregate` flag).""")

code("""# Cell 4: Paths, constants
DRIVE_ROOT = Path("/content/drive/MyDrive/prame-predict")
OUT_CSV    = DRIVE_ROOT / "data" / "expression" / "hest_prame_aggregate.csv"

META_CACHE  = Path("/content/hf_cache/meta")    # kept across slides
SLIDE_CACHE = Path("/content/hf_cache/slide")   # wiped per slide

for d in (OUT_CSV.parent, META_CACHE, SLIDE_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# Filtering / processing
INCLUDE_MELANOMA = False    # mirror 07's default: exclude SKCM samples
LIMIT            = None     # cap processed slides; None = all non-melanoma skin

print(f"Output CSV:   {OUT_CSV}")
print(f"Meta cache:   {META_CACHE}")
print(f"Slide cache:  {SLIDE_CACHE}  (cleared per slide)")""")

md("""## Stage 1 - HEST Metadata + Skin Filter

Fetch the HEST-1k metadata CSV from HuggingFace (07 tries several known filenames in order to survive minor HEST version bumps). Filter rows whose organ contains "skin"; if a disease column is available, drop melanoma/SKCM rows.

**If this cell raises**, the most likely cause is that HuggingFace gating on `MahmoodLab/hest` has not yet been approved for your account. Request access and re-run.""")

code("""# Cell 5: Fetch metadata, filter to non-melanoma skin
print(f"Downloading HEST metadata from {HEST_REPO_ID}...")
meta = _download_metadata(META_CACHE)
if meta is None:
    raise RuntimeError(
        f"Could not fetch any HEST metadata CSV from {HEST_REPO_ID}. "
        f"Likely cause: HuggingFace gating not yet approved. "
        f"Request access at https://huggingface.co/datasets/{HEST_REPO_ID}"
    )
print(f"  {len(meta)} total HEST rows")

skin = _filter_skin_nonmelanoma(meta, include_melanoma=INCLUDE_MELANOMA)
if len(skin) == 0:
    raise RuntimeError("No non-melanoma skin slides found in HEST metadata.")

# Column detection (mirrors 07's main())
id_col = next((c for c in skin.columns if c.lower() in ("id", "sample_id", "slide_id")), None)
if id_col is None:
    raise RuntimeError(f"No id column in metadata. Columns: {skin.columns.tolist()}")

cohort_col = next(
    (c for c in skin.columns
     if any(k in c.lower() for k in ("cohort", "dataset", "study"))),
    None,
)
disease_col = next(
    (c for c in skin.columns
     if any(k in c.lower() for k in ("disease", "oncotree", "cancer", "diagnosis"))),
    None,
)
tech_col = next(
    (c for c in skin.columns
     if any(k in c.lower() for k in ("technology", "platform", "st_technology"))),
    None,
)
print(f"  columns: id={id_col!r} cohort={cohort_col!r} disease={disease_col!r} tech={tech_col!r}")

if LIMIT is not None:
    skin = skin.head(LIMIT)
    print(f"  LIMIT applied: processing first {len(skin)} slides")

if cohort_col:
    print()
    print("Cohorts (top 10 by count):")
    print(skin[cohort_col].value_counts().head(10).to_string())

print(f"\\nWill aggregate {len(skin)} HEST slides.")
skin.head()""")

md("""## Stage 2 - Per-slide Pseudobulk with Cleanup

For each non-melanoma skin slide: download the `.h5ad` from `MahmoodLab/hest/st/<slide>.h5ad`, read with anndata, call 07's `_pseudobulk_prame`, append the row to the output CSV, then wipe `SLIDE_CACHE` so the blob is freed before the next slide. Resumable - already-processed `file_id`s in `OUT_CSV` are skipped on re-run.""")

code("""# Cell 6: Per-slide loop with incremental save + disk cleanup
# Load any prior progress
if OUT_CSV.exists():
    prev = pd.read_csv(OUT_CSV)
    done = set(prev["file_id"].astype(str))
    rows = prev.to_dict("records")
    print(f"Resuming: {len(done)} slides already in {OUT_CSV.name}")
else:
    done = set()
    rows = []

remaining = skin[~skin[id_col].astype(str).isin(done)]
print(f"Slides to process: {len(remaining)} / {len(skin)}")


def _clear_slide_cache():
    shutil.rmtree(SLIDE_CACHE, ignore_errors=True)
    SLIDE_CACHE.mkdir(parents=True, exist_ok=True)


for _, r in tqdm(remaining.iterrows(), total=len(remaining), desc="HEST slides"):
    slide_id = str(r[id_col])
    remote = f"{HEST_ST_SUBDIR}/{slide_id}.h5ad"

    # Download
    try:
        local = hf_hub_download(
            repo_id=HEST_REPO_ID, filename=remote,
            repo_type="dataset",
            cache_dir=str(SLIDE_CACHE),
        )
    except Exception as e:
        print(f"  [skip {slide_id}] download failed: {type(e).__name__}: {e}")
        _clear_slide_cache()
        continue

    # Read + pseudobulk
    try:
        adata = anndata.read_h5ad(local)
        n_in_tissue, total_counts, prame_count, prame_cpm = _pseudobulk_prame(adata)
        del adata
    except Exception as e:
        print(f"  [skip {slide_id}] read/pseudobulk failed: {type(e).__name__}: {e}")
        _clear_slide_cache()
        continue

    rows.append({
        "file_id": slide_id,
        "hest_cohort": str(r[cohort_col]) if cohort_col else "",
        "disease":     str(r[disease_col]) if disease_col else "",
        "platform":    str(r[tech_col]) if tech_col else "Visium",
        "n_in_tissue_spots": n_in_tissue,
        "total_raw_counts":  total_counts,
        "prame_raw_count":   prame_count,
        "prame_tpm_pseudobulk": prame_cpm,
    })

    # Incremental save - protects against Colab disconnect mid-run
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # Free the .h5ad blob before the next slide
    _clear_slide_cache()

print(f"\\nAggregation complete. {len(rows)} rows in {OUT_CSV}")""")

md("""## Stage 3 - Summary + Handoff

Distribution stats over the pseudobulk PRAME column. The output CSV is what `08_build_diagnostic_manifest.py` consumes via its `--hest-aggregate` flag (default path matches `OUT_CSV`).""")

code("""# Cell 7: Distribution stats + handoff
out = pd.read_csv(OUT_CSV)
print(f"Aggregated {len(out)} HEST slides into {OUT_CSV}")

miss = int(out["prame_tpm_pseudobulk"].isna().sum())
if miss:
    print(f"  [warn] {miss} slides had no PRAME gene row / zero counts")

non_na = out.dropna(subset=["prame_tpm_pseudobulk"])
if len(non_na):
    print()
    print("prame_tpm_pseudobulk (CPM) distribution:")
    print(f"  min    = {non_na['prame_tpm_pseudobulk'].min():.3f}")
    print(f"  q25    = {non_na['prame_tpm_pseudobulk'].quantile(0.25):.3f}")
    print(f"  median = {non_na['prame_tpm_pseudobulk'].median():.3f}")
    print(f"  q75    = {non_na['prame_tpm_pseudobulk'].quantile(0.75):.3f}")
    print(f"  max    = {non_na['prame_tpm_pseudobulk'].max():.3f}")

if "hest_cohort" in out.columns:
    print()
    print("Cohort breakdown (top 10):")
    print(out["hest_cohort"].value_counts().head(10).to_string())

print()
print("Feed to 08_build_diagnostic_manifest.py via")
print(f"  --hest-aggregate {OUT_CSV}")
print("(this matches 08's default path, so the flag can also be omitted.)")""")


nb = {
    "cells": cells,
    "metadata": {
        # No "accelerator" key: pseudobulk is CPU-only; omitting it matches
        # generate_heatmaps_colab.ipynb (also CPU-pinned).
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("notebooks/hest_aggregate_colab.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size} bytes, {len(cells)} cells)")
