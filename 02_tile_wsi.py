"""
Tile whole-slide images into 224x224 patches at 20x magnification.

This script:
1. Reads the slide manifest from 01_download_data.py
2. Opens each .svs file with OpenSlide
3. Determines the correct level for 20x magnification
4. Extracts non-overlapping 224x224 patches
5. Filters out background tiles using tissue detection
6. Saves tiles as PNGs to data/tiles/{submitter_id}/

Usage:
    python 02_tile_wsi.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except ImportError:
    raise ImportError(
        "openslide-python is required. Install it with:\n"
        "  pip install openslide-python\n"
        "On Windows, also download OpenSlide binaries from openslide.org "
        "and add the bin/ directory to your PATH."
    )


TILE_SIZE = 224
TARGET_MAG = 20.0
TISSUE_THRESHOLD = 0.3  # minimum fraction of tissue pixels in a tile
SATURATION_THRESHOLD = 15  # minimum mean saturation to count as tissue


def get_best_level(slide, target_mag):
    """
    Find the OpenSlide level closest to the target magnification.

    WSIs store images at multiple resolutions. The native magnification
    (usually 20x or 40x) is at level 0. Higher levels are downsampled
    by powers of 2. We pick the level whose effective magnification is
    closest to our target without going below it.

    Returns:
        (level, downsample_factor) — the level index and the factor to
        further resize patches if the level isn't exactly target_mag.
    """
    native_mag = float(slide.properties.get(
        openslide.PROPERTY_NAME_OBJECTIVE_POWER, 0
    ))

    if native_mag == 0:
        # Fallback: assume 40x if not specified, which is common for TCGA
        print("    Warning: native magnification not found, assuming 40x")
        native_mag = 40.0

    target_downsample = native_mag / target_mag

    # Find the level with downsample closest to (but not exceeding) target
    best_level = 0
    best_downsample = 1.0
    for level in range(slide.level_count):
        level_downsample = slide.level_downsamples[level]
        if level_downsample <= target_downsample + 0.1:
            best_level = level
            best_downsample = level_downsample

    # Additional scaling needed if the level doesn't match exactly
    scale_factor = best_downsample / target_downsample

    return best_level, scale_factor


def is_tissue_tile(tile_img):
    """
    Determine whether a tile contains enough tissue to be useful.

    Background in H&E slides is typically white/near-white with low
    saturation. Tissue regions have color (pink/purple staining) which
    shows up as higher saturation in HSV space. We also reject tiles
    that are mostly uniform (e.g., pen marks, artifacts) by checking
    that brightness varies.
    """
    arr = np.array(tile_img)

    # Skip nearly-all-white tiles (background)
    gray = np.mean(arr, axis=2)
    white_fraction = np.mean(gray > 220)
    if white_fraction > (1 - TISSUE_THRESHOLD):
        return False

    # Check saturation in HSV — tissue has color from staining
    hsv = np.array(tile_img.convert("HSV"))
    mean_saturation = hsv[:, :, 1].mean()
    if mean_saturation < SATURATION_THRESHOLD:
        return False

    return True


def tile_slide(slide_path, output_dir):
    """
    Extract all tissue-containing tiles from a single WSI.

    Returns the number of tiles saved.
    """
    slide = openslide.OpenSlide(str(slide_path))
    level, scale_factor = get_best_level(slide, TARGET_MAG)

    level_dims = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]

    # Size to read at this level to get a 224x224 tile at target mag
    # If scale_factor < 1, we need to read a larger region and resize down
    read_size = int(round(TILE_SIZE / scale_factor))

    # Step size in level-0 coordinates
    step_level0 = int(round(read_size * level_downsample))

    width_level0, height_level0 = slide.level_dimensions[0]
    n_cols = width_level0 // step_level0
    n_rows = height_level0 // step_level0

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_count = 0

    for row in range(n_rows):
        for col in range(n_cols):
            # Top-left corner in level-0 coordinates
            x = col * step_level0
            y = row * step_level0

            # Read region (coordinates are always level-0, size is at target level)
            tile = slide.read_region((x, y), level, (read_size, read_size))
            tile = tile.convert("RGB")

            # Resize to exact 224x224 if the level wasn't a perfect match
            if read_size != TILE_SIZE:
                tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)

            if is_tissue_tile(tile):
                tile_path = output_dir / f"tile_{row:04d}_{col:04d}.png"
                tile.save(tile_path)
                tile_count += 1

    slide.close()
    return tile_count


def main():
    manifest_path = Path("data/expression/slide_manifest.csv")
    wsi_dir = Path("data/wsi")
    tiles_dir = Path("data/tiles")

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found. Run 01_download_data.py first.")
        return

    manifest = pd.read_csv(manifest_path)
    print(f"Slide manifest: {len(manifest)} slides")

    # Find downloaded WSIs — gdc-client puts each file in a subdirectory
    # named by file_id, so we search recursively for .svs files
    svs_files = {}
    for svs_path in wsi_dir.rglob("*.svs"):
        svs_files[svs_path.name] = svs_path

    print(f"Found {len(svs_files)} .svs files in {wsi_dir}/")

    if not svs_files:
        print(
            "\nNo WSI files found. Download them first using:\n"
            "  gdc-client download -m data/expression/gdc_manifest.txt -d data/wsi/"
        )
        return

    # Tile each slide
    results = []
    matched = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Tiling slides"):
        file_name = row["file_name"]
        submitter_id = row["submitter_id"]

        if file_name not in svs_files:
            continue

        matched += 1
        slide_path = svs_files[file_name]
        slide_tiles_dir = tiles_dir / submitter_id

        # Skip if already tiled
        if slide_tiles_dir.exists() and any(slide_tiles_dir.iterdir()):
            n_existing = len(list(slide_tiles_dir.glob("*.png")))
            print(f"  {submitter_id}: already tiled ({n_existing} tiles), skipping")
            results.append({
                "submitter_id": submitter_id,
                "n_tiles": n_existing,
                "slide_path": str(slide_path),
            })
            continue

        print(f"  {submitter_id}: tiling {file_name}...")
        try:
            n_tiles = tile_slide(slide_path, slide_tiles_dir)
            print(f"    -> {n_tiles} tissue tiles extracted")
            results.append({
                "submitter_id": submitter_id,
                "n_tiles": n_tiles,
                "slide_path": str(slide_path),
            })
        except Exception as e:
            print(f"    -> Error: {e}")
            continue

    print(f"\nMatched {matched}/{len(manifest)} slides from manifest")

    # Save tiling summary
    if results:
        results_df = pd.DataFrame(results)
        summary_path = Path("data/expression/tiling_summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"Tiling summary saved to {summary_path}")

        total_tiles = results_df["n_tiles"].sum()
        print(f"Total tiles across all slides: {total_tiles}")
        print(f"Average tiles per slide: {total_tiles / len(results_df):.0f}")


if __name__ == "__main__":
    main()
