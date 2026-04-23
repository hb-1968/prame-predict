"""
Prepare the diagnostic (melanoma vs. not-melanoma) manifest for
Component 2.

This script queries the GDC API for non-melanoma cohorts, merges
them with the existing SKCM (melanoma-positive) manifest, and writes
a unified CSV with melanoma_label and source_group columns.

Sources:
- skcm_melanoma: all 200 existing TCGA-SKCM slides (positives)
- skcm_normal:   TCGA-SKCM Solid Tissue Normal (negatives)
- pan_normal:    Solid Tissue Normal across all TCGA projects except
                 SKCM and UVM (negatives)
- other_cancer:  Primary Tumor from a configurable list of
                 non-melanoma projects (negatives)

Downloading, tiling, and feature extraction are out of scope — the
existing 02/03 pipeline is per-slide and manifest-agnostic.

Usage:
    python 06_prepare_diagnostic_labels.py
    python 06_prepare_diagnostic_labels.py --dry-run
    python 06_prepare_diagnostic_labels.py --neg-cap 1000
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


GDC_API = "https://api.gdc.cancer.gov"

DEFAULT_NEGATIVE_PROJECTS = [
    "TCGA-BRCA",
    "TCGA-LUAD",
    "TCGA-COAD",
    "TCGA-PRAD",
    "TCGA-STAD",
]

# Melanoma projects excluded from pan_normal — SKCM is the positive
# cohort; UVM is uveal melanoma (not cutaneous), so including its
# normals would conflate negative signal with a melanoma subtype.
EXCLUDED_FROM_PAN_NORMAL = {"TCGA-SKCM", "TCGA-UVM"}

SLIDE_FIELDS = (
    "file_id,file_name,file_size,experimental_strategy,"
    "cases.case_id,cases.submitter_id,"
    "cases.project.project_id,"
    "cases.samples.sample_type"
)

SLIDE_COLUMNS = [
    "file_id", "file_name", "file_size_gb", "case_id",
    "submitter_id", "project_id", "sample_type", "experimental_strategy",
]

# Dedupe priority: earlier wins. Positives first so any spurious
# overlap keeps the PRAME-labeled row.
SOURCE_ORDER = ["skcm_melanoma", "skcm_normal", "pan_normal", "other_cancer"]


# ---------- GDC query helpers ----------

def _cache_path(cache_dir: Path, payload: dict) -> Path:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    return cache_dir / f"{h}.json"


def _cached_post(endpoint: str, body: dict, cache_dir: Path, cache_days: int) -> dict:
    """POST to GDC with a JSON body; cache the response on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, {"endpoint": endpoint, **body})
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days < cache_days:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    response = requests.post(
        f"{GDC_API}/{endpoint}",
        headers={"Content-Type": "application/json"},
        json=body,
    )
    response.raise_for_status()
    data = response.json()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _fetch_all_hits(
    endpoint: str,
    filters: dict,
    fields: str,
    cache_dir: Path,
    cache_days: int,
    page_size: int = 1000,
) -> list:
    """Page through a GDC query and return all hits."""
    all_hits = []
    offset = 0
    while True:
        body = {
            "filters": filters,
            "fields": fields,
            "format": "JSON",
            "size": page_size,
            "from": offset,
        }
        data = _cached_post(endpoint, body, cache_dir, cache_days)
        hits = data["data"]["hits"]
        all_hits.extend(hits)
        if len(hits) < page_size:
            break
        offset += page_size
    return all_hits


def _pick_sample(samples: list, expected: str | None) -> dict:
    """Pick the sample matching `expected` sample_type, else first."""
    if expected:
        for s in samples:
            if s.get("sample_type") == expected:
                return s
    return samples[0] if samples else {}


def _hits_to_df(hits: list, expected_sample_type: str | None = None) -> pd.DataFrame:
    """Flatten GDC file hits into the standard slide-row schema."""
    if not hits:
        return pd.DataFrame(columns=SLIDE_COLUMNS)
    rows = []
    for h in hits:
        cases = h.get("cases") or []
        case = cases[0] if cases else {}
        samples = case.get("samples") or []
        sample = _pick_sample(samples, expected_sample_type)
        rows.append({
            "file_id": h.get("file_id"),
            "file_name": h.get("file_name"),
            "file_size_gb": (h.get("file_size") or 0) / (1024 ** 3),
            "case_id": case.get("case_id"),
            "submitter_id": case.get("submitter_id"),
            "project_id": (case.get("project") or {}).get("project_id"),
            "sample_type": sample.get("sample_type"),
            "experimental_strategy": h.get("experimental_strategy"),
        })
    return pd.DataFrame(rows, columns=SLIDE_COLUMNS)


def query_slides_by_filter(
    base_filter_content: list,
    cache_dir: Path,
    cache_days: int,
    diagnostic_only: bool,
    expected_sample_type: str | None,
) -> pd.DataFrame:
    """
    Query GDC for slide images matching `base_filter_content`.

    When diagnostic_only is False, try `Diagnostic Slide` first and
    fall back to any slide image if that returns nothing — mirroring
    the fallback in 01_download_data.py. Tissue-level normals are
    often tagged `Tissue Slide` rather than `Diagnostic Slide`.
    """
    slide_filter = [
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
    ]
    diag_filter = [
        {"op": "=", "content": {
            "field": "experimental_strategy",
            "value": "Diagnostic Slide",
        }},
    ]

    diag_full = {
        "op": "and",
        "content": base_filter_content + slide_filter + diag_filter,
    }
    hits = _fetch_all_hits("files", diag_full, SLIDE_FIELDS, cache_dir, cache_days)

    if not hits and not diagnostic_only:
        fallback_full = {
            "op": "and",
            "content": base_filter_content + slide_filter,
        }
        hits = _fetch_all_hits("files", fallback_full, SLIDE_FIELDS, cache_dir, cache_days)

    return _hits_to_df(hits, expected_sample_type)


def list_tcga_projects(cache_dir: Path, cache_days: int) -> list:
    """Return all TCGA project_ids from the GDC /projects endpoint."""
    body = {
        "filters": {
            "op": "=",
            "content": {"field": "program.name", "value": "TCGA"},
        },
        "fields": "project_id",
        "format": "JSON",
        "size": 1000,
    }
    data = _cached_post("projects", body, cache_dir, cache_days)
    return [h["project_id"] for h in data["data"]["hits"]]


# ---------- sampling ----------

def stratified_cap(neg_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows from neg_df stratified by source_group, proportional
    to each group's size. Remainder rows go to the largest group.
    """
    if len(neg_df) <= n:
        return neg_df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sizes = neg_df["source_group"].value_counts().to_dict()
    total = sum(sizes.values())
    take = {g: (sizes[g] * n) // total for g in sizes}
    remainder = n - sum(take.values())
    for g in sorted(sizes, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        take[g] += 1
        remainder -= 1
    pieces = []
    for g, sub in neg_df.groupby("source_group", sort=False):
        k = min(take.get(g, 0), len(sub))
        if k == 0:
            continue
        idx = rng.choice(len(sub), size=k, replace=False)
        pieces.append(sub.iloc[idx])
    return pd.concat(pieces, ignore_index=True)


# ---------- reporting ----------

def print_summary(df: pd.DataFrame, title: str) -> None:
    print()
    print(f"=== {title} ===")
    print(f"Total rows: {len(df)}")
    print()
    print("Per source_group:")
    for grp, count in df["source_group"].value_counts().items():
        gb = df.loc[df["source_group"] == grp, "file_size_gb"].sum()
        print(f"  {grp:16s} n={count:5d}  size={gb:7.1f} GB")
    print()
    print("Top project_ids:")
    for pid, count in df["project_id"].value_counts().head(10).items():
        print(f"  {pid:16s} n={count}")
    print()
    pos = int((df["melanoma_label"] == 1).sum())
    neg = int((df["melanoma_label"] == 0).sum())
    if neg > 0:
        print(f"Positives:Negatives = {pos}:{neg}  ratio={pos/neg:.3f}")
    else:
        print(f"Positives:Negatives = {pos}:{neg}")
    total_gb = df["file_size_gb"].sum()
    print(f"Total download size: {total_gb:.1f} GB")


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Component-2 diagnostic manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/expression/slide_manifest.csv"),
        help="Existing SKCM melanoma manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/expression/diagnostic_manifest.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--negative-projects",
        type=str,
        default=",".join(DEFAULT_NEGATIVE_PROJECTS),
        help="Comma-separated TCGA projects for the other_cancer source.",
    )
    parser.add_argument("--skip-skcm-normal", action="store_true")
    parser.add_argument("--skip-pan-normal", action="store_true")
    parser.add_argument("--skip-other-cancer", action="store_true")
    parser.add_argument(
        "--neg-cap",
        type=int,
        default=None,
        help="Cap total negatives, stratified across source_group.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cache-days",
        type=int,
        default=30,
        help="Reuse cached GDC responses younger than N days.",
    )
    args = parser.parse_args()

    cache_dir = args.output.parent / ".06_gdc_cache"

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}")
        return 1

    print(f"Loading existing SKCM manifest: {args.manifest}")
    skcm_manifest = pd.read_csv(args.manifest)
    print(f"  loaded {len(skcm_manifest)} SKCM slide rows")

    # ---- [A] Re-query GDC for existing SKCM positives ----
    print()
    print("[A] Re-querying GDC for existing SKCM positives "
          "(fetching project_id, sample_type, experimental_strategy)...")
    pos_filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "file_id",
                "value": skcm_manifest["file_id"].tolist(),
            }},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        ],
    }
    pos_hits = _fetch_all_hits(
        "files", pos_filters, SLIDE_FIELDS, cache_dir, args.cache_days,
    )
    pos_gdc = _hits_to_df(pos_hits, expected_sample_type=None)
    print(f"  GDC returned metadata for {len(pos_gdc)} / {len(skcm_manifest)} file_ids")

    positives = skcm_manifest.merge(
        pos_gdc[["file_id", "project_id", "sample_type", "experimental_strategy"]],
        on="file_id",
        how="left",
    )
    missing_meta = positives["project_id"].isna().sum()
    if missing_meta > 0:
        print(f"  WARNING: {missing_meta} positive rows missing GDC metadata")

    # ---- [B] skcm_normal ----
    if args.skip_skcm_normal:
        print("\n[B] Skipping skcm_normal (--skip-skcm-normal)")
        skcm_normal = _hits_to_df([])
    else:
        print("\n[B] Querying skcm_normal (TCGA-SKCM, Solid Tissue Normal)...")
        skcm_normal = query_slides_by_filter(
            base_filter_content=[
                {"op": "=", "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-SKCM",
                }},
                {"op": "=", "content": {
                    "field": "cases.samples.sample_type",
                    "value": "Solid Tissue Normal",
                }},
            ],
            cache_dir=cache_dir,
            cache_days=args.cache_days,
            diagnostic_only=False,
            expected_sample_type="Solid Tissue Normal",
        )
        print(f"  found {len(skcm_normal)} skcm_normal slides")

    # ---- [C] pan_normal ----
    if args.skip_pan_normal:
        print("\n[C] Skipping pan_normal (--skip-pan-normal)")
        pan_normal = _hits_to_df([])
    else:
        print("\n[C] Querying pan_normal "
              "(all TCGA projects except SKCM/UVM, Solid Tissue Normal)...")
        tcga_projects = list_tcga_projects(cache_dir, args.cache_days)
        pan_projects = sorted(p for p in tcga_projects if p not in EXCLUDED_FROM_PAN_NORMAL)
        print(f"  {len(pan_projects)} TCGA projects after excluding "
              f"{sorted(EXCLUDED_FROM_PAN_NORMAL)}")
        pan_normal = query_slides_by_filter(
            base_filter_content=[
                {"op": "in", "content": {
                    "field": "cases.project.project_id",
                    "value": pan_projects,
                }},
                {"op": "=", "content": {
                    "field": "cases.samples.sample_type",
                    "value": "Solid Tissue Normal",
                }},
            ],
            cache_dir=cache_dir,
            cache_days=args.cache_days,
            diagnostic_only=False,
            expected_sample_type="Solid Tissue Normal",
        )
        print(f"  found {len(pan_normal)} pan_normal slides")

    # ---- [D] other_cancer ----
    if args.skip_other_cancer:
        print("\n[D] Skipping other_cancer (--skip-other-cancer)")
        other_cancer = _hits_to_df([])
    else:
        neg_projects = [p.strip() for p in args.negative_projects.split(",") if p.strip()]
        print(f"\n[D] Querying other_cancer "
              f"(Primary Tumor, projects={neg_projects})...")
        other_cancer = query_slides_by_filter(
            base_filter_content=[
                {"op": "in", "content": {
                    "field": "cases.project.project_id",
                    "value": neg_projects,
                }},
                {"op": "=", "content": {
                    "field": "cases.samples.sample_type",
                    "value": "Primary Tumor",
                }},
            ],
            cache_dir=cache_dir,
            cache_days=args.cache_days,
            diagnostic_only=True,
            expected_sample_type="Primary Tumor",
        )
        print(f"  found {len(other_cancer)} other_cancer slides")

    # ---- [E/F] Tag, concat, dedupe on file_id ----
    print("\n[E/F] Tagging, concatenating, and deduping on file_id...")
    named = [
        ("skcm_melanoma", positives, 1),
        ("skcm_normal", skcm_normal, 0),
        ("pan_normal", pan_normal, 0),
        ("other_cancer", other_cancer, 0),
    ]
    parts = []
    for name, df, label in named:
        if len(df) == 0:
            continue
        df = df.copy()
        df["source_group"] = name
        df["melanoma_label"] = label
        df["_order"] = SOURCE_ORDER.index(name)
        parts.append(df)

    pre_count = sum(len(p) for p in parts)
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = combined.sort_values("_order", kind="stable")
    combined = combined.drop_duplicates(subset=["file_id"], keep="first")
    combined = combined.drop(columns=["_order"]).reset_index(drop=True)
    print(f"  {pre_count} -> {len(combined)} rows after dedupe "
          f"({pre_count - len(combined)} duplicates removed)")

    # ---- [G] Optional stratified cap on negatives ----
    if args.neg_cap is not None:
        pos_mask = combined["melanoma_label"] == 1
        pos_df = combined[pos_mask]
        neg_df = combined[~pos_mask]
        print(f"\n[G] Capping negatives at {args.neg_cap} "
              f"(stratified, seed={args.seed})")
        print(f"  pre-cap: {len(neg_df)} negatives")
        neg_df = stratified_cap(neg_df, args.neg_cap, args.seed)
        print(f"  post-cap: {len(neg_df)} negatives")
        combined = pd.concat([pos_df, neg_df], ignore_index=True)

    # ---- [H] Finalize column order + write output ----
    column_order = [
        "file_id", "file_name", "file_size_gb",
        "case_id", "submitter_id",
        "project_id", "sample_type", "experimental_strategy",
        "melanoma_label", "source_group",
        "prame_tpm", "prame_group", "prame_label",
    ]
    for col in column_order:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[column_order]

    print_summary(combined, title="Final manifest")

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"\nWrote {len(combined)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
