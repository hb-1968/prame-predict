"""
Download PRAME expression data from TCGA-SKCM and identify
which slides to download based on expression levels.

This script:
1. Queries the GDC API for TCGA-SKCM RNA-Seq gene expression
2. Extracts PRAME (ENSG00000185686) expression values
3. Performs a median split into high/low expression groups
4. Identifies available diagnostic H&E slides for each case
5. Saves a manifest for WSI download

Usage:
    python 01_download_data.py
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from pathlib import Path


GDC_API = "https://api.gdc.cancer.gov"
PRAME_GENE_ID = "ENSG00000185686"


def query_expression_data():
    """
    Query GDC for TCGA-SKCM RNA-Seq STAR counts.

    The GDC API uses a JSON-based filter syntax. We request:
    - Project: TCGA-SKCM (skin cutaneous melanoma)
    - Data type: Gene Expression Quantification
    - Workflow: STAR - Counts (the current standard pipeline)

    The API returns file metadata, not the actual expression values.
    We then download each file to extract PRAME expression.
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {
                "field": "cases.project.project_id",
                "value": "TCGA-SKCM"
            }},
            {"op": "=", "content": {
                "field": "data_type",
                "value": "Gene Expression Quantification"
            }},
            {"op": "=", "content": {
                "field": "analysis.workflow_type",
                "value": "STAR - Counts"
            }},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "file_id,file_name,"
            "cases.case_id,"
            "cases.submitter_id,"
            "cases.samples.sample_type,"
            "cases.samples.portions.analytes.aliquots.submitter_id"
        ),
        "format": "JSON",
        "size": "1000",
    }

    print("Querying GDC for TCGA-SKCM expression files...")
    response = requests.get(f"{GDC_API}/files", params=params)
    response.raise_for_status()
    data = response.json()

    hits = data["data"]["hits"]
    print(f"Found {len(hits)} expression files")
    return hits


def _download_one(session, hit):
    """
    Download a single expression file and return the PRAME record, or None.

    Uses a shared requests.Session for TCP connection reuse.
    """
    from io import StringIO

    file_id = hit["file_id"]
    case_id = hit["cases"][0]["case_id"]
    submitter_id = hit["cases"][0]["submitter_id"]

    try:
        response = session.get(f"{GDC_API}/data/{file_id}")
        response.raise_for_status()

        expr_df = pd.read_csv(StringIO(response.text), sep="\t", comment="#")

        # Find PRAME row — column name varies by GDC version
        if "gene_id" in expr_df.columns:
            prame_rows = expr_df[
                expr_df["gene_id"].str.startswith(PRAME_GENE_ID)
            ]
        elif "gene_name" in expr_df.columns:
            prame_rows = expr_df[expr_df["gene_name"] == "PRAME"]
        else:
            return None

        if prame_rows.empty:
            return None

        prame_row = prame_rows.iloc[0]

        # Extract TPM (transcripts per million) — the normalized measure
        tpm_col = None
        for col in ["tpm_unstranded", "TPM", "FPKM"]:
            if col in prame_row.index:
                tpm_col = col
                break

        if tpm_col is None:
            return None

        return {
            "case_id": case_id,
            "submitter_id": submitter_id,
            "file_id": file_id,
            "prame_tpm": float(prame_row[tpm_col]),
        }

    except Exception as e:
        print(f"  Error on {file_id}: {e}")
        return None


def extract_prame_expression(hits, max_workers=16):
    """
    Download expression files in parallel and extract PRAME values.

    Uses a thread pool (I/O-bound) and a shared session for connection reuse.
    """
    records = []
    session = requests.Session()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, session, hit): i
            for i, hit in enumerate(hits)
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  Processed {done}/{len(hits)}...")
            result = future.result()
            if result is not None:
                records.append(result)

    return pd.DataFrame(records)


def query_available_slides(case_ids):
    """
    Query GDC for diagnostic H&E slide images available for our cases.

    We specifically request:
    - data_type: Slide Image
    - experimental_strategy: Diagnostic Slide
    - data_format: SVS (the standard whole-slide image format)
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {
                "field": "cases.project.project_id",
                "value": "TCGA-SKCM"
            }},
            {"op": "=", "content": {
                "field": "data_type",
                "value": "Slide Image"
            }},
            {"op": "=", "content": {
                "field": "experimental_strategy",
                "value": "Diagnostic Slide"
            }},
            {"op": "in", "content": {
                "field": "cases.case_id",
                "value": case_ids
            }},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "file_id,file_name,file_size,"
            "cases.case_id,cases.submitter_id"
        ),
        "format": "JSON",
        "size": "1000",
    }

    print("Querying GDC for available diagnostic slides...")
    response = requests.post(
        f"{GDC_API}/files",
        headers={"Content-Type": "application/json"},
        json={
            "filters": filters,
            "fields": (
                "file_id,file_name,file_size,"
                "cases.case_id,cases.submitter_id"
            ),
            "format": "JSON",
            "size": "1000",
        },
    )
    response.raise_for_status()
    data = response.json()

    slides = []
    for hit in data["data"]["hits"]:
        slides.append({
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "file_size_gb": hit["file_size"] / (1024 ** 3),
            "case_id": hit["cases"][0]["case_id"],
            "submitter_id": hit["cases"][0]["submitter_id"],
        })

    slides_df = pd.DataFrame(slides)
    print(f"Found {len(slides_df)} diagnostic slides")
    return slides_df


def main():
    out_dir = Path("data/expression")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check for cached expression data
    expr_path = out_dir / "prame_expression.csv"
    if expr_path.exists():
        print(f"Found cached expression data at {expr_path}, skipping download")
        expr_df = pd.read_csv(expr_path)
        print(f"Loaded {len(expr_df)} cases")
    else:
        # Step 2: Download and extract PRAME expression
        hits = query_expression_data()

        print("\nDownloading expression data and extracting PRAME values...")
        expr_df = extract_prame_expression(hits)
        print(f"\nPRAME expression obtained for {len(expr_df)} samples")

        # Remove duplicates (some cases have multiple aliquots)
        # Keep the one with highest TPM as the representative
        expr_df = (expr_df
                   .sort_values("prame_tpm", ascending=False)
                   .drop_duplicates(subset="submitter_id", keep="first")
                   .reset_index(drop=True))
        print(f"Unique cases: {len(expr_df)}")

    # Step 3: Tag expression quartiles
    q25 = expr_df["prame_tpm"].quantile(0.25)
    q75 = expr_df["prame_tpm"].quantile(0.75)

    expr_df["prame_group"] = "middle"
    expr_df.loc[expr_df["prame_tpm"] >= q75, "prame_group"] = "high"
    expr_df.loc[expr_df["prame_tpm"] <= q25, "prame_group"] = "low"

    # Binary label for quartile extremes (used for ground-truth evaluation)
    # Middle cases get label -1 to distinguish them
    expr_df["prame_label"] = -1
    expr_df.loc[expr_df["prame_group"] == "high", "prame_label"] = 1
    expr_df.loc[expr_df["prame_group"] == "low", "prame_label"] = 0

    high = expr_df[expr_df["prame_group"] == "high"]
    low = expr_df[expr_df["prame_group"] == "low"]
    middle = expr_df[expr_df["prame_group"] == "middle"]

    print(f"\nPRAME TPM statistics (full cohort):")
    print(f"  Q25:    {q25:.2f}")
    print(f"  Median: {expr_df['prame_tpm'].median():.2f}")
    print(f"  Q75:    {q75:.2f}")
    print(f"  Mean:   {expr_df['prame_tpm'].mean():.2f}")
    print(f"  Min:    {expr_df['prame_tpm'].min():.2f}")
    print(f"  Max:    {expr_df['prame_tpm'].max():.2f}")
    print(f"\nQuartile groups:")
    print(f"  High (>= Q75): {len(high)} cases")
    print(f"  Low  (<= Q25): {len(low)} cases")
    print(f"  Middle:         {len(middle)} cases")

    # Save expression data
    expr_path = out_dir / "prame_expression.csv"
    expr_df.to_csv(expr_path, index=False)
    print(f"\nSaved expression data to {expr_path}")

    # Step 4: Find available slides
    case_ids = expr_df["case_id"].tolist()
    slides_df = query_available_slides(case_ids)

    if slides_df.empty:
        print("No diagnostic slides found — trying tissue slides...")
        # Some TCGA-SKCM cases only have tissue slide images
        # Retry without the diagnostic filter
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-SKCM"
                }},
                {"op": "=", "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }},
                {"op": "in", "content": {
                    "field": "cases.case_id",
                    "value": case_ids
                }},
            ]
        }
        params = {
            "filters": json.dumps(filters),
            "fields": (
                "file_id,file_name,file_size,"
                "cases.case_id,cases.submitter_id"
            ),
            "format": "JSON",
            "size": "1000",
        }
        response = requests.get(f"{GDC_API}/files", params=params)
        response.raise_for_status()
        data = response.json()

        slides = []
        for hit in data["data"]["hits"]:
            slides.append({
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size_gb": hit["file_size"] / (1024 ** 3),
                "case_id": hit["cases"][0]["case_id"],
                "submitter_id": hit["cases"][0]["submitter_id"],
            })
        slides_df = pd.DataFrame(slides)
        print(f"Found {len(slides_df)} total slide images")

    # Merge with expression data
    merged = slides_df.merge(
        expr_df[["case_id", "prame_tpm", "prame_label", "prame_group"]],
        on="case_id",
        how="inner",
    )
    print(f"Slides with matched expression data: {len(merged)}")

    # Step 5: Select slides
    total_size_gb = merged["file_size_gb"].sum()
    print(f"Total download size for all matched slides: {total_size_gb:.1f} GB")

    disk_budget_gb = 200  # leave headroom for tiles and embeddings

    if total_size_gb <= disk_budget_gb:
        selected = merged.reset_index(drop=True)
    else:
        # Prioritize quartile extremes (ground-truth evaluation set),
        # then fill remaining budget with middle cases
        extremes = merged[merged["prame_label"] != -1].copy()
        middle = merged[merged["prame_label"] == -1].copy()

        extreme_size = extremes["file_size_gb"].sum()

        if extreme_size <= disk_budget_gb:
            # All extremes fit — fill remaining space with smallest middle slides
            remaining_budget = disk_budget_gb - extreme_size
            middle_sorted = middle.nsmallest(len(middle), "file_size_gb")
            middle_cumsize = middle_sorted["file_size_gb"].cumsum()
            middle_selected = middle_sorted[middle_cumsize <= remaining_budget]
            selected = pd.concat([extremes, middle_selected]).reset_index(drop=True)
        else:
            # Extremes alone exceed budget — take balanced sample from
            # high and low groups, smallest slides first, up to budget
            high = extremes[extremes["prame_label"] == 1].nsmallest(len(extremes), "file_size_gb")
            low = extremes[extremes["prame_label"] == 0].nsmallest(len(extremes), "file_size_gb")

            selected_parts = []
            budget_left = disk_budget_gb
            # Interleave high/low to keep classes balanced
            for h_row, l_row in zip(high.itertuples(), low.itertuples()):
                if h_row.file_size_gb + l_row.file_size_gb > budget_left:
                    break
                selected_parts.append(high.loc[[h_row.Index]])
                selected_parts.append(low.loc[[l_row.Index]])
                budget_left -= h_row.file_size_gb + l_row.file_size_gb

            selected = pd.concat(selected_parts).reset_index(drop=True) if selected_parts else pd.DataFrame()

    high_count = len(selected[selected["prame_label"] == 1])
    low_count = len(selected[selected["prame_label"] == 0])
    middle_count = len(selected[selected["prame_label"] == -1])
    subset_size_gb = selected["file_size_gb"].sum()

    print(f"\nSelected slides:")
    print(f"  Total: {len(selected)}")
    print(f"  High PRAME (quartile): {high_count}")
    print(f"  Low PRAME (quartile): {low_count}")
    print(f"  Middle (full-range training): {middle_count}")
    print(f"  Download size: {subset_size_gb:.1f} GB")

    # Save manifest
    manifest_path = out_dir / "slide_manifest.csv"
    selected.to_csv(manifest_path, index=False)
    print(f"Saved slide manifest to {manifest_path}")

    # Save GDC download manifest (for gdc-client)
    gdc_manifest_path = out_dir / "gdc_manifest.txt"
    with open(gdc_manifest_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for _, row in selected.iterrows():
            f.write(f"{row['file_id']}\t{row['file_name']}\t\t{int(row['file_size_gb'] * 1024**3)}\t\n")
    print(f"Saved GDC manifest to {gdc_manifest_path}")

    print(f"\nNext step: download slides using the GDC Data Transfer Tool:")
    print(f"  gdc-client download -m {gdc_manifest_path} -d data/wsi/")


if __name__ == "__main__":
    main()