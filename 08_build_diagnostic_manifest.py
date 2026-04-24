"""
Build the Component-2 training manifest by composing four sources:

    1. TCGA-SKCM positives     (200 slides, measured PRAME, from Component 1)
    2. TCGA-SKCM tumor-free    (3 slides, biopsy, GDC mini-query)
    3. GTEx normal skin        (200 slides, measured PRAME, direct download)
    4. COBRA BCC               (~115 slides, NaN or predicted PRAME, S3 inventory + 06 output)
    5. HEST-1k Visium          (~85 slides, pseudobulk PRAME, 07 output)

Output: data/expression/diagnostic_manifest.csv — the single source
of truth for Component-2 training. Every row has a stable schema
and an explicit `prame_source` tag so the downstream classifier
can reason about per-source label provenance.

Plan E decision (2026-04-23): `prame_tpm` is kept in source-native
units. Harmonization across sources (quantile-normalize vs.
binarize vs. ignore) is pushed to 09's modeling stage rather than
baked into this manifest.

Prerequisites:
    - Component-1 completed (slide_manifest.csv with measured PRAME
      for 200 SKCM slides).
    - Optional: 06_predict_cobra_prame.py has produced
      cobra_prame_predictions.csv.
    - Optional: 07_aggregate_hest_prame.py has produced
      hest_prame_aggregate.csv.

If 06 or 07 output is missing, those slots are skipped with a
warning and the manifest ships partial.

Usage:
    python 08_build_diagnostic_manifest.py                    # default full build
    python 08_build_diagnostic_manifest.py --dry-run          # inventory + write nothing
    python 08_build_diagnostic_manifest.py --skip-gtex        # exclude GTEx
    python 08_build_diagnostic_manifest.py --skip-cobra       # exclude COBRA
    python 08_build_diagnostic_manifest.py --skip-hest        # exclude HEST

Optional dependency:
    boto3  — required for COBRA S3 inventory (pip install boto3)
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ---------- constants ----------

PRAME_GENE_ID = "ENSG00000185686"

# GDC — for the 3 SKCM tumor-free rows
GDC_API = "https://api.gdc.cancer.gov"

# GTEx V10 open-access
GTEX_TPM_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/"
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
)
GTEX_SAMPLE_ATTR_URL = (
    "https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/"
    "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
)
GTEX_HISTOLOGY_URL = "https://brd.nci.nih.gov/brd/imagedownload/{sampid}"

GTEX_SKIN_TISSUES = {
    "Skin - Sun Exposed (Lower leg)": "sun_exposed",
    "Skin - Not Sun Exposed (Suprapubic)": "not_sun_exposed",
}

# COBRA
COBRA_S3_BUCKET = "cobra-pathology"
COBRA_S3_REGION = "us-west-2"

# Final manifest schema (16 columns)
MANIFEST_SCHEMA = [
    "file_id", "file_name", "file_size_gb",
    "case_id", "submitter_id",
    "project_id", "sample_type", "experimental_strategy",
    "melanoma_label", "source_group",
    "prame_tpm", "prame_group", "prame_label",
    "has_prame", "prame_source", "download_url",
]


# ---------- small helpers ----------

def _download_cached(url: str, cache_dir: Path, cache_days: int = 30) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:16] + "_" + Path(url).name
    path = cache_dir / name
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days < cache_days:
            return path
    print(f"  downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def _stem(file_name: str) -> str:
    return Path(str(file_name)).stem


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_SCHEMA)


# ---------- GDC mini-query (SKCM tumor-free) ----------

def _gdc_cache_path(cache_dir: Path, payload: dict) -> Path:
    raw = json.dumps(payload, sort_keys=True).encode()
    h = hashlib.sha256(raw).hexdigest()
    return cache_dir / f"{h}.json"


def _gdc_post(endpoint: str, body: dict, cache_dir: Path, cache_days: int) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _gdc_cache_path(cache_dir, {"endpoint": endpoint, **body})
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days < cache_days:
            return json.loads(path.read_text(encoding="utf-8"))
    r = requests.post(
        f"{GDC_API}/{endpoint}",
        headers={"Content-Type": "application/json"},
        json=body,
    )
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def ingest_skcm_normal(cache_dir: Path, cache_days: int) -> pd.DataFrame:
    """Pull the 3 TCGA-SKCM tumor-free slides via GDC."""
    print("\n[SKCM-normal] Querying GDC (TCGA-SKCM, Solid Tissue Normal)...")

    def _query(diagnostic_only: bool) -> list:
        base = [
            {"op": "=", "content": {
                "field": "cases.project.project_id", "value": "TCGA-SKCM",
            }},
            {"op": "=", "content": {
                "field": "cases.samples.sample_type", "value": "Solid Tissue Normal",
            }},
            {"op": "=", "content": {
                "field": "data_type", "value": "Slide Image",
            }},
        ]
        if diagnostic_only:
            base.append({"op": "=", "content": {
                "field": "experimental_strategy", "value": "Diagnostic Slide",
            }})
        body = {
            "filters": {"op": "and", "content": base},
            "fields": (
                "file_id,file_name,file_size,experimental_strategy,"
                "cases.case_id,cases.submitter_id,"
                "cases.project.project_id,cases.samples.sample_type"
            ),
            "format": "JSON",
            "size": 1000,
            "from": 0,
        }
        data = _gdc_post("files", body, cache_dir, cache_days)
        return data["data"]["hits"]

    hits = _query(diagnostic_only=True)
    if not hits:
        print("  no Diagnostic Slides, retrying without the filter...")
        hits = _query(diagnostic_only=False)
    print(f"  found {len(hits)} SKCM-normal slides")

    rows = []
    for h in hits:
        cases = h.get("cases") or []
        case = cases[0] if cases else {}
        samples = case.get("samples") or []
        sample = next((s for s in samples if s.get("sample_type") == "Solid Tissue Normal"),
                      samples[0] if samples else {})
        file_id = h.get("file_id", "")
        rows.append({
            "file_id": file_id,
            "file_name": h.get("file_name", ""),
            "file_size_gb": (h.get("file_size") or 0) / (1024 ** 3),
            "case_id": case.get("case_id", ""),
            "submitter_id": case.get("submitter_id", ""),
            "project_id": (case.get("project") or {}).get("project_id", "TCGA-SKCM"),
            "sample_type": sample.get("sample_type", "Solid Tissue Normal"),
            "experimental_strategy": h.get("experimental_strategy", ""),
            "melanoma_label": 0,
            "source_group": "skcm_normal",
            "prame_tpm": np.nan,         # PRAME measurement exists in TCGA but not backfilled here
            "prame_group": np.nan,
            "prame_label": np.nan,
            "has_prame": False,
            "prame_source": "tcga_unmeasured",
            "download_url": f"{GDC_API}/data/{file_id}",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_SCHEMA)
    print(f"  -> {len(df)} SKCM-normal rows")
    return df


# ---------- SKCM positives from slide_manifest.csv ----------

def ingest_skcm_positives(slide_manifest_path: Path) -> pd.DataFrame:
    print(f"\n[SKCM-positive] Reading {slide_manifest_path}")
    if not slide_manifest_path.exists():
        print(f"  ERROR: slide manifest not found at {slide_manifest_path}")
        return _empty_manifest()
    mani = pd.read_csv(slide_manifest_path)
    print(f"  {len(mani)} rows in Component-1 manifest")
    rows = []
    for _, r in mani.iterrows():
        file_id = r["file_id"]
        rows.append({
            "file_id": file_id,
            "file_name": r["file_name"],
            "file_size_gb": float(r.get("file_size_gb", np.nan) or np.nan),
            "case_id": r.get("case_id", ""),
            "submitter_id": r.get("submitter_id", ""),
            "project_id": "TCGA-SKCM",
            "sample_type": "Primary Tumor",   # SKCM positives are 01Z/06Z; captured generically
            "experimental_strategy": "Diagnostic Slide",
            "melanoma_label": 1,
            "source_group": "skcm_melanoma",
            "prame_tpm": float(r["prame_tpm"]) if pd.notna(r.get("prame_tpm")) else np.nan,
            "prame_group": r.get("prame_group", np.nan),
            "prame_label": int(r["prame_label"]) if pd.notna(r.get("prame_label")) else np.nan,
            "has_prame": pd.notna(r.get("prame_tpm")),
            "prame_source": "tcga",
            "download_url": f"{GDC_API}/data/{file_id}",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_SCHEMA)
    print(f"  -> {len(df)} SKCM-positive rows")
    return df


# ---------- GTEx ingest ----------

def _read_gct_gene_row(path: Path, ensembl_id: str) -> pd.Series:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        _version = f.readline()
        _dims = f.readline()
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0].split(".")[0] == ensembl_id:
                values = pd.to_numeric(parts[2:], errors="coerce")
                return pd.Series(values, index=header[2:], name="prame_tpm")
    raise KeyError(f"gene {ensembl_id} not found in {path}")


def _stratified_by_tpm(df: pd.DataFrame, n: int, col: str = "prame_tpm") -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    ordered = df.sort_values(col).reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, n).astype(int)
    return ordered.iloc[idx].reset_index(drop=True)


def ingest_gtex(n_per_tissue: int, cache_dir: Path, cache_days: int) -> pd.DataFrame:
    print(f"\n[GTEx] Ingesting ~{n_per_tissue * len(GTEX_SKIN_TISSUES)} normal-skin samples")
    tpm_path = _download_cached(GTEX_TPM_URL, cache_dir, cache_days)
    attr_path = _download_cached(GTEX_SAMPLE_ATTR_URL, cache_dir, cache_days)

    print(f"  parsing TPM matrix for {PRAME_GENE_ID}...")
    prame = _read_gct_gene_row(tpm_path, PRAME_GENE_ID)
    prame_df = prame.reset_index().rename(columns={"index": "SAMPID"})

    print("  reading sample annotations...")
    attr = pd.read_csv(attr_path, sep="\t", low_memory=False)
    skin_attr = attr[attr["SMTSD"].isin(GTEX_SKIN_TISSUES)][["SAMPID", "SMTSD"]]
    print(f"  {len(skin_attr)} GTEx skin rows in sample-attrs file")

    joined = skin_attr.merge(prame_df, on="SAMPID", how="inner").dropna(subset=["prame_tpm"])
    print(f"  {len(joined)} GTEx skin rows with PRAME TPM")

    picks = []
    for tissue in GTEX_SKIN_TISSUES:
        sub = joined[joined["SMTSD"] == tissue]
        take = _stratified_by_tpm(sub, n_per_tissue)
        print(f"    {tissue}: {len(sub)} available, picking {len(take)}")
        picks.append(take)
    picked = pd.concat(picks, ignore_index=True)

    rows = []
    for _, r in picked.iterrows():
        sampid = r["SAMPID"]
        donor = "-".join(sampid.split("-")[:2])
        rows.append({
            "file_id": sampid,
            "file_name": f"{sampid}.svs",
            "file_size_gb": np.nan,
            "case_id": donor,
            "submitter_id": donor,
            "project_id": "GTEx",
            "sample_type": f"Normal - {GTEX_SKIN_TISSUES[r['SMTSD']]}",
            "experimental_strategy": "Diagnostic Slide",
            "melanoma_label": 0,
            "source_group": "gtex_normal",
            "prame_tpm": float(r["prame_tpm"]),
            "prame_group": np.nan,
            "prame_label": np.nan,
            "has_prame": True,
            "prame_source": "gtex",
            "download_url": GTEX_HISTOLOGY_URL.format(sampid=sampid),
        })
    df = pd.DataFrame(rows, columns=MANIFEST_SCHEMA)
    print(f"  -> {len(df)} GTEx rows")
    return df


# ---------- COBRA inventory + join with 06 predictions ----------

def ingest_cobra(
    n: int, predictions_path: Path | None, seed: int = 42,
) -> pd.DataFrame:
    print(f"\n[COBRA] Listing s3://{COBRA_S3_BUCKET}/ (unsigned)...")
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("  ERROR: boto3 not installed. pip install boto3")
        return _empty_manifest()

    s3 = boto3.client(
        "s3", region_name=COBRA_S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=COBRA_S3_BUCKET):
        for obj in page.get("Contents", []):
            keys.append({"key": obj["Key"], "size_gb": obj["Size"] / (1024 ** 3)})
    print(f"  {len(keys)} objects in bucket")

    wsi_exts = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif")
    wsis = [k for k in keys if k["key"].lower().endswith(wsi_exts)]
    print(f"  {len(wsis)} WSI files")

    bcc_subtype_keys = (
        "ia", "ib", "ii", "iii",
        "nodular", "superficial", "medium", "high", "aggressive",
    )
    risky_keywords = ("risky", "melanoma", "merkel")
    scc_keywords = ("scc", "squamous")

    def classify(key: str) -> str | None:
        k = key.lower()
        if any(r in k for r in risky_keywords + scc_keywords):
            return None
        for sub in bcc_subtype_keys:
            if f"/{sub}/" in k or f"_{sub}_" in k or f"_{sub}." in k or f"-{sub}-" in k:
                return sub
        if "bcc" in k:
            return "bcc_unspecified"
        return None

    classified = []
    for w in wsis:
        sub = classify(w["key"])
        if sub is None:
            continue
        classified.append({**w, "subtype": sub})
    print(f"  {len(classified)} BCC WSIs (post risky-bucket + SCC filter)")

    if not classified:
        print("  [warn] BCC heuristic matched nothing. Inspect layout:")
        print(f"    aws s3 ls --no-sign-request --recursive s3://{COBRA_S3_BUCKET}/ | head -50")
        return _empty_manifest()

    df = pd.DataFrame(classified)
    rng = np.random.default_rng(seed)
    per_sub = max(1, n // df["subtype"].nunique())
    picks = []
    for sub, group in df.groupby("subtype"):
        if len(group) <= per_sub:
            picks.append(group)
        else:
            idx = rng.choice(len(group), size=per_sub, replace=False)
            picks.append(group.iloc[idx])
    picked = pd.concat(picks).head(n).reset_index(drop=True)

    # Load predictions from 06 if available
    predictions = None
    if predictions_path and predictions_path.exists():
        try:
            predictions = pd.read_csv(predictions_path)
            predictions = predictions.set_index(
                predictions["file_id"].apply(_stem)
            )
            print(f"  loaded {len(predictions)} predictions from {predictions_path}")
        except Exception as e:
            print(f"  [warn] failed to read predictions ({e}); shipping COBRA with NaN PRAME")
            predictions = None
    else:
        print(f"  [info] no predictions file at {predictions_path}; COBRA PRAME will be NaN. "
              "Run 06_predict_cobra_prame.py once COBRA embeddings exist.")

    rows = []
    for _, r in picked.iterrows():
        key = r["key"]
        stem = _stem(key)
        prame_tpm = np.nan
        prame_source = "cobra_missing"
        has_prame = False
        if predictions is not None and stem in predictions.index:
            p = predictions.loc[stem]
            prame_tpm = float(p["prob_mean"]) if pd.notna(p.get("prob_mean")) else np.nan
            if pd.notna(prame_tpm):
                prame_source = "component1_predicted"
                has_prame = True
        rows.append({
            "file_id": key,
            "file_name": Path(key).name,
            "file_size_gb": float(r["size_gb"]),
            "case_id": stem,
            "submitter_id": stem,
            "project_id": "COBRA",
            "sample_type": f"BCC - {r['subtype']}",
            "experimental_strategy": "Diagnostic Slide",
            "melanoma_label": 0,
            "source_group": "cobra_bcc",
            "prame_tpm": prame_tpm,
            "prame_group": np.nan,
            "prame_label": np.nan,
            "has_prame": has_prame,
            "prame_source": prame_source,
            "download_url": f"s3://{COBRA_S3_BUCKET}/{key}",
        })
    df_out = pd.DataFrame(rows, columns=MANIFEST_SCHEMA)
    populated = int(df_out["has_prame"].sum())
    print(f"  -> {len(df_out)} COBRA rows  ({populated} with predicted PRAME, "
          f"{len(df_out) - populated} NaN)")
    return df_out


# ---------- HEST from 07 output ----------

def ingest_hest(aggregate_path: Path) -> pd.DataFrame:
    print(f"\n[HEST] Reading {aggregate_path}")
    if not aggregate_path.exists():
        print(f"  [info] no aggregate file at {aggregate_path}; HEST slot empty. "
              "Run 07_aggregate_hest_prame.py after HEST HF access is granted.")
        return _empty_manifest()
    agg = pd.read_csv(aggregate_path)
    print(f"  {len(agg)} HEST slides aggregated")

    rows = []
    for _, r in agg.iterrows():
        rid = str(r["file_id"])
        prame = r.get("prame_tpm_pseudobulk", np.nan)
        prame = float(prame) if pd.notna(prame) else np.nan
        has_prame = pd.notna(prame)
        rows.append({
            "file_id": rid,
            "file_name": f"{rid}.tif",
            "file_size_gb": np.nan,
            "case_id": rid,
            "submitter_id": rid,
            "project_id": str(r.get("hest_cohort", "HEST-1k")),
            "sample_type": str(r.get("disease", "non_melanoma_skin")),
            "experimental_strategy": str(r.get("platform", "Visium")),
            "melanoma_label": 0,
            "source_group": "hest_visium",
            "prame_tpm": prame,
            "prame_group": np.nan,
            "prame_label": np.nan,
            "has_prame": has_prame,
            "prame_source": "hest_pseudobulk" if has_prame else "hest_missing",
            "download_url": f"hf://datasets/MahmoodLab/hest/wsis/{rid}.tif",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_SCHEMA)
    print(f"  -> {len(df)} HEST rows")
    return df


# ---------- summary ----------

def print_summary(df: pd.DataFrame) -> None:
    print()
    print("=== Final manifest summary ===")
    print(f"Total rows: {len(df)}")
    print()
    print(f"{'source_group':<18s} {'count':>6s} {'has_prame':>10s} {'size_gb':>10s}")
    for grp, count in df["source_group"].value_counts().items():
        sub = df[df["source_group"] == grp]
        prame_frac = sub["has_prame"].mean()
        gb = sub["file_size_gb"].sum()
        print(f"{grp:<18s} {count:>6d} {prame_frac:>9.0%} {gb:>9.1f}")
    print()
    pos = int((df["melanoma_label"] == 1).sum())
    neg = int((df["melanoma_label"] == 0).sum())
    ratio = f" (ratio={pos/neg:.3f})" if neg else ""
    print(f"Positives:Negatives = {pos}:{neg}{ratio}")
    hp = int(df["has_prame"].sum())
    print(f"Rows with PRAME label: {hp} / {len(df)}  ({hp / max(len(df), 1):.1%})")
    print()
    print("PRAME source breakdown:")
    for src, count in df["prame_source"].value_counts().items():
        print(f"  {src:<25s} {count}")


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Component-2 diagnostic manifest from all sources.",
    )
    parser.add_argument(
        "--slide-manifest", type=Path,
        default=Path("data/expression/slide_manifest.csv"),
        help="Component-1 SKCM manifest (default: data/expression/slide_manifest.csv)",
    )
    parser.add_argument(
        "--cobra-predictions", type=Path,
        default=Path("data/expression/cobra_prame_predictions.csv"),
        help="Output of 06_predict_cobra_prame.py",
    )
    parser.add_argument(
        "--hest-aggregate", type=Path,
        default=Path("data/expression/hest_prame_aggregate.csv"),
        help="Output of 07_aggregate_hest_prame.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/expression/diagnostic_manifest.csv"),
    )
    parser.add_argument("--gtex-n", type=int, default=100,
                        help="GTEx samples per skin tissue (x2 tissues)")
    parser.add_argument("--cobra-n", type=int, default=115)
    parser.add_argument("--skip-skcm-normal", action="store_true")
    parser.add_argument("--skip-gtex", action="store_true")
    parser.add_argument("--skip-cobra", action="store_true")
    parser.add_argument("--skip-hest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-days", type=int, default=30)
    args = parser.parse_args()

    cache_dir = args.output.parent / ".08_cache"
    parts = []

    # Always: SKCM positives
    parts.append(ingest_skcm_positives(args.slide_manifest))

    if args.skip_skcm_normal:
        print("\n[SKCM-normal] Skipped (--skip-skcm-normal)")
    else:
        parts.append(ingest_skcm_normal(cache_dir, args.cache_days))

    if args.skip_gtex:
        print("\n[GTEx] Skipped (--skip-gtex)")
    else:
        parts.append(ingest_gtex(args.gtex_n, cache_dir, args.cache_days))

    if args.skip_cobra:
        print("\n[COBRA] Skipped (--skip-cobra)")
    else:
        parts.append(ingest_cobra(args.cobra_n, args.cobra_predictions, seed=args.seed))

    if args.skip_hest:
        print("\n[HEST] Skipped (--skip-hest)")
    else:
        parts.append(ingest_hest(args.hest_aggregate))

    combined = pd.concat(parts, ignore_index=True, sort=False)
    pre = len(combined)
    combined = combined.drop_duplicates(subset=["file_id"], keep="first").reset_index(drop=True)
    if len(combined) < pre:
        print(f"\nDeduped {pre - len(combined)} rows on file_id")

    # Ensure column order
    for col in MANIFEST_SCHEMA:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[MANIFEST_SCHEMA]

    print_summary(combined)

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"\nWrote {len(combined)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
