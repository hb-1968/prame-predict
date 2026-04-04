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


def download_expression_file(file_id):
    """
    Download a single gene expression quantification file from GDC.

    GDC serves these as tab-separated files inside a gzipped tarball,
    but the /data endpoint with a single file_id returns just the TSV.
    """
    url = f"{GDC_API}/data/{file_id}"
    response = requests.get(url)
    response.raise_for_status()

    # The response is a TSV file
    from io import StringIO
    lines = response.text
    df = pd.read_csv(StringIO(lines), sep="\t", comment="#")
    return df


def extract_prame_expression(hits):
    """
    For each expression file, download it and extract the PRAME row.

    This takes a few minutes since we're downloading one file at a time.
    Each file is small (~2 MB) but there are hundreds of them.
    """
    records = []

    for i, hit in enumerate(hits):
        file_id = hit["file_id"]
        case_id = hit["cases"][0]["case_id"]
        submitter_id = hit["cases"][0]["submitter_id"]

        if (i + 1) % 20 == 0:
            print(f"  Processing {i + 1}/{len(hits)}...")

        try:
            expr_df = download_expression_file(file_id)

            # Find PRAME row — column name varies by GDC version
            # Look for Ensembl gene ID in gene_id or gene_name column
            if "gene_id" in expr_df.columns:
                prame_rows = expr_df[
                    expr_df["gene_id"].str.startswith(PRAME_GENE_ID)
                ]
            elif "gene_name" in expr_df.columns:
                prame_rows = expr_df[expr_df["gene_name"] == "PRAME"]
            else:
                print(f"  Skipping {file_id}: unrecognized column format")
                continue

            if prame_rows.empty:
                print(f"  Skipping {file_id}: PRAME not found")
                continue

            prame_row = prame_rows.iloc[0]

            # Extract TPM (transcripts per million) — the normalized measure
            # Some files use "tpm_unstranded", others "TPM"
            tpm_col = None
            for col in ["tpm_unstranded", "TPM", "FPKM"]:
                if col in prame_row.index:
                    tpm_col = col
                    break

            if tpm_col is None:
                print(f"  Skipping {file_id}: no TPM/FPKM column found")
                continue

            records.append({
                "case_id": case_id,
                "submitter_id": submitter_id,
                "file_id": file_id,
                "prame_tpm": float(prame_row[tpm_col]),
            })

        except Exception as e:
            print(f"  Error on {file_id}: {e}")
            continue

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

    # Step 1: Get expression files
    hits = query_expression_data()

    # Step 2: Extract PRAME expression
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

    # Step 3: Quartile split (exclude ambiguous middle)
    #
    # PRAME IHC is only diagnostically reliable at the extremes:
    # diffuse 4+ staining (>75% cells) favors melanoma, while
    # complete negativity favors nevus. Intermediate staining
    # (1+ through 3+) is noncontributory (O'Connor et al. 2022,
    # Sullivan et al. 2026). We mirror this clinical reality by
    # selecting only the top and bottom quartiles of PRAME mRNA
    # expression, discarding ambiguous intermediate cases.
    q25 = expr_df["prame_tpm"].quantile(0.25)
    q75 = expr_df["prame_tpm"].quantile(0.75)

    high = expr_df[expr_df["prame_tpm"] >= q75].copy()
    low = expr_df[expr_df["prame_tpm"] <= q25].copy()
    excluded = expr_df[
        (expr_df["prame_tpm"] > q25) & (expr_df["prame_tpm"] < q75)
    ]

    high["prame_label"] = 1
    high["prame_group"] = "high"
    low["prame_label"] = 0
    low["prame_group"] = "low"

    expr_df_full = expr_df.copy()  # keep full data for reference
    expr_df = pd.concat([high, low]).reset_index(drop=True)

    print(f"\nPRAME TPM statistics (full cohort):")
    print(f"  Q25:    {q25:.2f}")
    print(f"  Median: {expr_df_full['prame_tpm'].median():.2f}")
    print(f"  Q75:    {q75:.2f}")
    print(f"  Mean:   {expr_df_full['prame_tpm'].mean():.2f}")
    print(f"  Min:    {expr_df_full['prame_tpm'].min():.2f}")
    print(f"  Max:    {expr_df_full['prame_tpm'].max():.2f}")
    print(f"\nQuartile split:")
    print(f"  High (>= Q75): {len(high)} cases")
    print(f"  Low  (<= Q25): {len(low)} cases")
    print(f"  Excluded middle: {len(excluded)} cases")

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

    # Step 5: Select a manageable subset
    # Aim for ~80 slides, balanced between high and low PRAME
    total_size_gb = merged["file_size_gb"].sum()
    print(f"Total download size for all matched slides: {total_size_gb:.1f} GB")

    # Balance the groups and limit by disk space
    target_per_group = 40
    high_slides = merged[merged["prame_label"] == 1].nsmallest(
        target_per_group, "file_size_gb"
    )
    low_slides = merged[merged["prame_label"] == 0].nsmallest(
        target_per_group, "file_size_gb"
    )
    selected = pd.concat([high_slides, low_slides]).reset_index(drop=True)

    subset_size_gb = selected["file_size_gb"].sum()
    print(f"\nSelected subset:")
    print(f"  Slides: {len(selected)} ({len(high_slides)} high, {len(low_slides)} low)")
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