"""
Tile whole-slide images into 224x224 patches at 20x magnification.

Detects tissue regions using saturation thresholding on a low-resolution
thumbnail, then extracts patches only from tissue-containing regions.

Usage:
    python 02_tile_wsi.py --slide data/wsi/TCGA-XX-XXXX.svs
    python 02_tile_wsi.py --all  # process all slides in data/wsi/
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import openslide
import cv2


def get_20x_level(slide):
    """
    Determine which pyramid level corresponds to 20x magnification
    and what downsample factor to apply.

    OpenSlide stores the scan magnification in properties.
    A 40x-scanned slide has level 0 at 40x, so we need
    downsample=2 to reach 20x. A 20x-scanned slide needs
    downsample=1 (read directly from level 0).

    Returns:
        level: which pyramid level to read from
        downsample: additional scaling needed after reading
    """
    # Get the objective magnification from slide metadata
    obj_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

    if obj_power is None:
        # If metadata is missing, assume 20x and read from level 0
        print("  Warning: no magnification metadata, assuming 20x")
        return 0, 1.0

    obj_power = float(obj_power)
    target = 20.0

    # Find the pyramid level closest to 20x
    # Each level has a downsample factor relative to level 0
    best_level = 0
    best_diff = float("inf")
    for level in range(slide.level_count):
        level_downsample = slide.level_downsamples[level]
        effective_mag = obj_power / level_downsample
        diff = abs(effective_mag - target)
        if diff < best_diff:
            best_diff = diff
            best_level = level
            best_downsample = level_downsample

    effective_mag = obj_power / best_downsample
    scale = effective_mag / target  # additional scaling if not exactly 20x

    return best_level, scale


def detect_tissue(slide, thumb_size=2048, sat_threshold=20):
    """
    Create a binary mask of tissue regions using saturation thresholding.

    Steps:
    1. Read a low-resolution thumbnail from the slide
    2. Convert from RGB to HSV color space
    3. Threshold the saturation channel — tissue has color (high S),
       background is white (low S)
    4. Clean up with morphological operations (close small holes,
       remove small noise)

    Returns:
        mask: binary numpy array (1=tissue, 0=background) at thumbnail scale
        scale_x, scale_y: ratio of full-resolution to thumbnail coordinates
    """
    # Get thumbnail dimensions maintaining aspect ratio
    slide_w, slide_h = slide.dimensions
    if slide_w > slide_h:
        thumb_w = thumb_size
        thumb_h = int(thumb_size * slide_h / slide_w)
    else:
        thumb_h = thumb_size
        thumb_w = int(thumb_size * slide_w / slide_h)

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_array = np.array(thumbnail)

    # Convert RGB to HSV and threshold on saturation
    hsv = cv2.cvtColor(thumb_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    mask = (saturation > sat_threshold).astype(np.uint8)

    # Morphological cleanup: close small holes, remove small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Calculate coordinate mapping between thumbnail and full-res
    actual_h, actual_w = mask.shape
    scale_x = slide_w / actual_w
    scale_y = slide_h / actual_h

    tissue_pct = mask.sum() / mask.size * 100
    print(f"  Tissue detected: {tissue_pct:.1f}% of slide")

    return mask, scale_x, scale_y


def extract_patches(slide, mask, scale_x, scale_y, level, level_scale,
                    patch_size=224, tissue_threshold=0.5):
    """
    Extract patches from tissue regions.

    Iterates over the tissue mask in a grid. For each grid position,
    checks if at least tissue_threshold fraction of the patch area
    overlaps with tissue. If so, reads the full-resolution patch
    from OpenSlide.

    Args:
        slide: OpenSlide object
        mask: binary tissue mask at thumbnail resolution
        scale_x, scale_y: coordinate mapping from thumbnail to full-res
        level: pyramid level to read from
        level_scale: additional scaling for exact 20x
        patch_size: output patch dimensions (224 for foundation models)
        tissue_threshold: minimum fraction of patch that must be tissue

    Returns:
        patches: list of PIL Images
        coords: list of (x, y) tuples in full-resolution coordinates
    """
    # Size of patch in full-resolution coordinates
    # If level_scale > 1, we need to read a larger region and resize
    read_size = int(patch_size * level_scale)

    # Size of patch in thumbnail coordinates
    thumb_patch_w = int(read_size / scale_x)
    thumb_patch_h = int(read_size / scale_y)

    # Ensure minimum 1 pixel in thumbnail space
    thumb_patch_w = max(thumb_patch_w, 1)
    thumb_patch_h = max(thumb_patch_h, 1)

    mask_h, mask_w = mask.shape
    patches = []
    coords = []

    # Stride across the mask in patch-sized steps
    for y in range(0, mask_h - thumb_patch_h + 1, thumb_patch_h):
        for x in range(0, mask_w - thumb_patch_w + 1, thumb_patch_w):
            # Check tissue coverage in this patch region
            patch_mask = mask[y:y + thumb_patch_h, x:x + thumb_patch_w]
            tissue_ratio = patch_mask.mean()

            if tissue_ratio < tissue_threshold:
                continue

            # Convert thumbnail coordinates to full-resolution
            full_x = int(x * scale_x)
            full_y = int(y * scale_y)

            # Read patch from slide at the chosen pyramid level
            # OpenSlide.read_region always takes level-0 coordinates
            patch = slide.read_region(
                (full_x, full_y),  # top-left corner in level-0 coords
                level,             # which pyramid level to read from
                (read_size, read_size)  # size in level pixels
            )

            # Convert RGBA to RGB (OpenSlide returns RGBA)
            patch = patch.convert("RGB")

            # Resize to exact patch_size if we read a larger region
            if read_size != patch_size:
                patch = patch.resize(
                    (patch_size, patch_size),
                    Image.LANCZOS
                )

            patches.append(patch)
            coords.append((full_x, full_y))

    return patches, coords


def tile_slide(slide_path, patch_size=224):
    """
    Full tiling pipeline for a single slide.

    Returns:
        patches: list of PIL Images (224x224 RGB)
        coords: list of (x, y) full-resolution coordinates
    """
    slide = openslide.OpenSlide(str(slide_path))

    print(f"  Dimensions: {slide.dimensions}")
    print(f"  Levels: {slide.level_count}")

    level, level_scale = get_20x_level(slide)
    print(f"  Using level {level} (scale {level_scale:.2f}) for 20x")

    mask, scale_x, scale_y = detect_tissue(slide)
    patches, coords = extract_patches(
        slide, mask, scale_x, scale_y, level, level_scale, patch_size
    )

    slide.close()
    print(f"  Extracted {len(patches)} patches")
    return patches, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str, help="path to a single .svs file")
    parser.add_argument("--all", action="store_true", help="process all slides")
    parser.add_argument("--wsi-dir", default="data/wsi")
    parser.add_argument("--out-dir", default="data/tiles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.slide:
        slides = [Path(args.slide)]
    elif args.all:
        slides = sorted(Path(args.wsi_dir).glob("*.svs"))
    else:
        print("Specify --slide <path> or --all")
        return

    print(f"Processing {len(slides)} slides")

    for i, slide_path in enumerate(slides):
        print(f"\n[{i+1}/{len(slides)}] {slide_path.name}")

        slide_out = out_dir / slide_path.stem
        coord_path = slide_out / "coords.npy"

        if coord_path.exists():
            print("  Already tiled, skipping")
            continue

        slide_out.mkdir(parents=True, exist_ok=True)

        patches, coords = tile_slide(slide_path)

        # Save patches as JPEG (smaller than PNG, fine for features)
        for j, patch in enumerate(patches):
            patch.save(slide_out / f"{j:05d}.jpg", quality=95)

        # Save coordinates for heatmap reconstruction later
        np.save(coord_path, np.array(coords))

        print(f"  Saved to {slide_out}")


if __name__ == "__main__":
    main()