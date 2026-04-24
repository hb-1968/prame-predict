"""
Aggregate HEST-1k Visium spot counts into per-slide pseudobulk
PRAME expression.

HEST-1k (MahmoodLab, NeurIPS 2024) publishes per-slide AnnData
(.h5ad) objects with spot-level gene counts. For each non-melanoma
skin slide, this script:
    1. Downloads the .h5ad from HuggingFace (gated — requires
       approved access to MahmoodLab/hest).
    2. Restricts to in-tissue spots.
    3. Sums raw counts per gene across all in-tissue spots.
    4. Normalizes to per-million (pseudobulk CPM; labeled
       "prame_tpm_pseudobulk" for consistency with TCGA/GTEx
       terminology, though strict TPM additionally gene-length-
       corrects).
    5. Extracts the PRAME row.

Output: data/expression/hest_prame_aggregate.csv, one row per
slide:
    file_id, hest_cohort, disease, platform,
    n_in_tissue_spots, total_raw_counts,
    prame_raw_count, prame_tpm_pseudobulk

Feeds `08_build_diagnostic_manifest.py` which joins on file_id
and assigns prame_source="hest_pseudobulk".

Usage:
    python 07_aggregate_hest_prame.py                    # all non-melanoma skin
    python 07_aggregate_hest_prame.py --dry-run          # HF auth check + metadata only
    python 07_aggregate_hest_prame.py --limit 5          # aggregate 5 slides (testing)
    python 07_aggregate_hest_prame.py --include-melanoma # don't filter out SKCM samples

Optional dependencies (new to this repo):
    scanpy>=1.9, anndata>=0.9   — for AnnData I/O
    huggingface_hub (already in repo env)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PRAME_GENE_SYMBOL = "PRAME"
PRAME_ENSEMBL_ID = "ENSG00000185686"

HEST_REPO_ID = "MahmoodLab/hest"
HEST_METADATA_CANDIDATES = [
    "HEST_v1_1_0.csv",
    "HEST_v1_0_0.csv",
    "HEST_v1_3_0.csv",
    "HEST.csv",
]
HEST_ST_SUBDIR = "st"   # per-slide .h5ad location within the repo


# ---------- HF access + metadata ----------

def _download_metadata(cache_dir: Path) -> pd.DataFrame | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        return None

    for fname in HEST_METADATA_CANDIDATES:
        try:
            path = hf_hub_download(
                repo_id=HEST_REPO_ID,
                filename=fname,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            print(f"  loaded metadata: {fname}")
            return pd.read_csv(path)
        except Exception as e:
            print(f"  skip {fname}: {type(e).__name__}")
    return None


def _filter_skin_nonmelanoma(meta: pd.DataFrame, include_melanoma: bool) -> pd.DataFrame:
    organ_col = next((c for c in meta.columns if "organ" in c.lower()), None)
    if organ_col is None:
        print(f"  ERROR: no organ column in metadata. Columns: {meta.columns.tolist()}")
        return meta.iloc[0:0]

    skin = meta[meta[organ_col].astype(str).str.contains("skin", case=False, na=False)].copy()
    print(f"  {len(skin)} skin rows (organ column: {organ_col!r})")

    if include_melanoma:
        return skin

    disease_col = next(
        (c for c in meta.columns
         if any(k in c.lower() for k in ("disease", "oncotree", "cancer", "diagnosis"))),
        None,
    )
    if disease_col is None:
        print("  [warn] no disease/oncotree column; cannot exclude melanoma. "
              "Pass --include-melanoma to suppress this warning.")
        return skin

    melanoma_mask = skin[disease_col].astype(str).str.contains(
        "melanoma|skcm", case=False, na=False,
    )
    out = skin[~melanoma_mask].copy()
    print(f"  {len(out)} non-melanoma skin rows (filtered on column: {disease_col!r})")
    return out


# ---------- per-slide pseudobulk ----------

def _find_prame_row(var: pd.DataFrame) -> int | None:
    """Locate PRAME in an AnnData.var table by symbol or Ensembl ID."""
    if PRAME_GENE_SYMBOL in var.index:
        return var.index.get_loc(PRAME_GENE_SYMBOL)
    for col in var.columns:
        s = var[col].astype(str)
        if (s == PRAME_GENE_SYMBOL).any():
            return int(np.where(s.values == PRAME_GENE_SYMBOL)[0][0])
        if s.str.startswith(PRAME_ENSEMBL_ID).any():
            return int(np.where(s.str.startswith(PRAME_ENSEMBL_ID).values)[0][0])
    return None


def _pseudobulk_prame(adata) -> tuple[int, int, int, float]:
    """
    Compute per-slide PRAME pseudobulk.

    Returns:
        n_in_tissue, total_counts, prame_count, prame_cpm
    """
    # in_tissue flag: Visium standard, some HEST variants differ
    if "in_tissue" in adata.obs.columns:
        mask = adata.obs["in_tissue"].astype(bool).values
    else:
        mask = np.ones(adata.n_obs, dtype=bool)
    sub = adata[mask, :]
    n_in_tissue = int(mask.sum())

    X = sub.X
    # sum across spots -> per-gene total
    if hasattr(X, "toarray"):
        gene_sums = np.asarray(X.sum(axis=0)).ravel()
    else:
        gene_sums = np.asarray(X).sum(axis=0)
    total = float(gene_sums.sum())
    if total == 0:
        return n_in_tissue, 0, 0, float("nan")

    idx = _find_prame_row(sub.var)
    if idx is None:
        return n_in_tissue, int(total), 0, float("nan")

    prame_count = int(gene_sums[idx])
    prame_cpm = float(prame_count / total * 1e6)
    return n_in_tissue, int(total), prame_count, prame_cpm


def _download_and_pseudobulk(
    slide_id: str, cache_dir: Path,
) -> dict | None:
    """Download one slide's .h5ad from HEST and compute PRAME pseudobulk."""
    try:
        from huggingface_hub import hf_hub_download
        import anndata
    except ImportError as e:
        print(f"  ERROR: missing dependency ({e}). "
              "Install: pip install scanpy anndata huggingface_hub")
        return None

    remote = f"{HEST_ST_SUBDIR}/{slide_id}.h5ad"
    try:
        local = hf_hub_download(
            repo_id=HEST_REPO_ID, filename=remote,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as e:
        print(f"  [skip {slide_id}] download failed: {type(e).__name__}: {e}")
        return None

    try:
        adata = anndata.read_h5ad(local)
    except Exception as e:
        print(f"  [skip {slide_id}] read_h5ad failed: {type(e).__name__}: {e}")
        return None

    n_in_tissue, total_counts, prame_count, prame_cpm = _pseudobulk_prame(adata)
    return {
        "n_in_tissue_spots": n_in_tissue,
        "total_raw_counts": total_counts,
        "prame_raw_count": prame_count,
        "prame_tpm_pseudobulk": prame_cpm,
    }


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate HEST-1k Visium spots into per-slide pseudobulk PRAME.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/expression/hest_prame_aggregate.csv"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap processed slides (useful for testing)")
    parser.add_argument("--include-melanoma", action="store_true",
                        help="Do not filter out HEST melanoma samples (default: filter)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="HF hub cache dir (default: HF default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch metadata and report counts, don't download slides")
    args = parser.parse_args()

    print(f"Downloading HEST-1k metadata from {HEST_REPO_ID}...")
    meta = _download_metadata(args.cache_dir or args.output.parent / ".07_cache")
    if meta is None:
        print(f"\nERROR: could not fetch any HEST metadata CSV.")
        print(f"Likely cause: HuggingFace gating on {HEST_REPO_ID}.")
        print(f"Request access at https://huggingface.co/datasets/{HEST_REPO_ID}")
        return 1
    print(f"  {len(meta)} total HEST rows")

    skin = _filter_skin_nonmelanoma(meta, args.include_melanoma)
    if len(skin) == 0:
        print("No non-melanoma skin slides in HEST metadata.")
        return 1

    # Identify columns used downstream
    id_col = next((c for c in skin.columns if c.lower() in ("id", "sample_id", "slide_id")), None)
    if id_col is None:
        print(f"ERROR: no id column in metadata. Columns: {skin.columns.tolist()}")
        return 1
    cohort_col = next((c for c in skin.columns
                       if any(k in c.lower() for k in ("cohort", "dataset", "study"))),
                      None)
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

    if args.limit is not None:
        skin = skin.head(args.limit)
        print(f"  --limit: processing first {len(skin)} slides")

    if args.dry_run:
        print("\n[dry-run] Skipping .h5ad downloads.")
        print(f"Would aggregate {len(skin)} slides.")
        if cohort_col:
            print(f"Cohorts (top 10 by count):")
            print(skin[cohort_col].value_counts().head(10).to_string())
        return 0

    print(f"\nAggregating {len(skin)} slides (scanpy + anndata required)...")
    rows = []
    for _, r in tqdm(skin.iterrows(), total=len(skin), desc="HEST slides"):
        slide_id = str(r[id_col])
        result = _download_and_pseudobulk(
            slide_id, args.cache_dir or args.output.parent / ".07_cache",
        )
        if result is None:
            continue
        rows.append({
            "file_id": slide_id,
            "hest_cohort": str(r[cohort_col]) if cohort_col else "",
            "disease": str(r[disease_col]) if disease_col else "",
            "platform": str(r[tech_col]) if tech_col else "Visium",
            **result,
        })

    out = pd.DataFrame(rows)
    print()
    print("=== Summary ===")
    print(f"Aggregated {len(out)} of {len(skin)} HEST slides")
    if len(out):
        miss = int(out["prame_tpm_pseudobulk"].isna().sum())
        if miss:
            print(f"  [warn] {miss} slides had no PRAME gene row / zero counts")
        non_na = out.dropna(subset=["prame_tpm_pseudobulk"])
        if len(non_na):
            print(f"  PRAME CPM  median={non_na['prame_tpm_pseudobulk'].median():.2f}  "
                  f"min={non_na['prame_tpm_pseudobulk'].min():.2f}  "
                  f"max={non_na['prame_tpm_pseudobulk'].max():.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
