"""
Tile whole-slide images into 224x224 patches at 20x magnification.

Detects tissue regions using saturation thresholding on a low-resolution
thumbnail, then extracts patches only from tissue-containing regions.

Usage:
    python 02_tile_wsi.py --slide data/wsi/TCGA-XX-XXXX.svs
    python 02_tile_wsi.py --all  # process all slides in data/wsi/
"""

import argparse
import os
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import openslide
import cv2
import torch
from tqdm import tqdm

torch.set_num_threads(torch.get_num_threads())

# Thread-local storage for reusing OpenSlide handles
_thread_local = threading.local()


def get_20x_level(slide):
    """
    Determine which pyramid level corresponds to 20x magnification
    and what downsample factor to apply.

    Always prefers oversampling (reading at >= 20x and downscaling)
    over undersampling (reading below 20x and upscaling), since
    downscaling preserves spatial detail while upscaling fabricates it.

    Returns:
        level: which pyramid level to read from
        scale: ratio of effective magnification to 20x (>= 1.0 means downscale)
    """
    obj_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

    if obj_power is None:
        print("  Warning: no magnification metadata, assuming 20x")
        return 0, 1.0

    obj_power = float(obj_power)
    target = 20.0

    # Find the best level at or above 20x (prefer downscaling over upscaling)
    best_level = 0
    best_mag = obj_power  # level 0 is always >= any lower level

    for level in range(slide.level_count):
        level_downsample = slide.level_downsamples[level]
        effective_mag = obj_power / level_downsample

        if effective_mag >= target - 0.1:  # at or above 20x (with float tolerance)
            # Prefer the level closest to 20x from above (least oversampling)
            if effective_mag < best_mag or best_mag < target:
                best_mag = effective_mag
                best_level = level

    scale = best_mag / target
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


def _find_tissue_coords(mask, scale_x, scale_y, read_size, tissue_threshold=0.5):
    """
    Scan the tissue mask and return full-resolution coordinates for
    every grid cell that passes the tissue coverage threshold.
    """
    thumb_patch_w = max(int(read_size / scale_x), 1)
    thumb_patch_h = max(int(read_size / scale_y), 1)
    mask_h, mask_w = mask.shape

    coords = []
    for y in range(0, mask_h - thumb_patch_h + 1, thumb_patch_h):
        for x in range(0, mask_w - thumb_patch_w + 1, thumb_patch_w):
            tissue_ratio = mask[y:y + thumb_patch_h, x:x + thumb_patch_w].mean()
            if tissue_ratio >= tissue_threshold:
                coords.append((int(x * scale_x), int(y * scale_y)))
    return coords


def _get_slide_handle(slide_path):
    """Get or create a thread-local OpenSlide handle (avoids re-opening per patch)."""
    key = str(slide_path)
    if not hasattr(_thread_local, 'handles'):
        _thread_local.handles = {}
    if key not in _thread_local.handles:
        _thread_local.handles[key] = openslide.OpenSlide(key)
    return _thread_local.handles[key]


def _close_thread_handles():
    """Close all thread-local OpenSlide handles."""
    if hasattr(_thread_local, 'handles'):
        for s in _thread_local.handles.values():
            s.close()
        _thread_local.handles.clear()


def _read_patch_into(slide_path, coord, level, read_size, patch_size, out_array, idx):
    """
    Read a patch and write it directly into a pre-allocated array slot.
    Bypasses PIL for conversion and resize — goes straight to numpy/cv2.
    """
    slide = _get_slide_handle(slide_path)
    region = slide.read_region(coord, level, (read_size, read_size))

    # RGBA buffer straight to numpy, drop alpha
    rgba = np.frombuffer(region.tobytes(), dtype=np.uint8).reshape(read_size, read_size, 4)
    rgb = rgba[:, :, :3]

    if read_size != patch_size:
        # cv2.resize is 3-5x faster than PIL.resize
        out_array[idx] = cv2.resize(rgb, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    else:
        out_array[idx] = rgb


def extract_patches(slide_path, mask, scale_x, scale_y, level, level_scale,
                    out_path, patch_size=224, tissue_threshold=0.5, workers=1,
                    chunk_size=2048, max_patches=None, in_memory=False):
    """
    Extract patches from tissue regions.

    When in_memory=False (default), writes to a memory-mapped .npy file on disk.
    When in_memory=True, builds the array in RAM and skips disk I/O entirely.
    Use in_memory=True when patches will be consumed immediately (e.g., Colab
    pipeline where extraction follows tiling with no intermediate save).

    Args:
        slide_path: path to the .svs file
        mask: binary tissue mask at thumbnail resolution
        scale_x, scale_y: coordinate mapping from thumbnail to full-res
        level: pyramid level to read from
        level_scale: additional scaling for exact 20x
        out_path: path to write patches.npy (ignored if in_memory=True)
        patch_size: output patch dimensions (224 for foundation models)
        tissue_threshold: minimum fraction of patch that must be tissue
        workers: number of threads for parallel reads
        chunk_size: patches per chunk (~300MB RAM at 224x224x3)
        max_patches: if set, randomly sample down to this many patches
        in_memory: if True, return patches array directly instead of writing to disk

    Returns:
        num_patches: number of patches extracted
        coords: list of (x, y) tuples in full-resolution coordinates
        patches (only if in_memory=True): numpy array of shape (N, 224, 224, 3)
    """
    read_size = int(patch_size * level_scale)
    coords = _find_tissue_coords(mask, scale_x, scale_y, read_size, tissue_threshold)

    if not coords:
        empty = np.empty((0, patch_size, patch_size, 3), dtype=np.uint8)
        if in_memory:
            return 0, [], empty
        np.save(out_path, empty)
        return 0, []

    if max_patches and len(coords) > max_patches:
        print(f"  Sampling {max_patches} of {len(coords)} tissue patches")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(coords), size=max_patches, replace=False)
        indices.sort()  # preserve spatial locality for sequential disk reads
        coords = [coords[i] for i in indices]

    n = len(coords)

    if in_memory:
        # Allocate full array in RAM — no disk I/O
        patches = np.empty((n, patch_size, patch_size, 3), dtype=np.uint8)
    else:
        # Create memory-mapped file on disk
        dummy = np.lib.format.open_memmap(
            str(out_path), mode='w+', dtype=np.uint8,
            shape=(n, patch_size, patch_size, 3)
        )
        del dummy
        patches = np.lib.format.open_memmap(str(out_path), mode='r+')

    pbar = tqdm(total=n, desc="  Reading patches", unit="patch")

    def _process_chunks():
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk_coords = coords[chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            # Write directly into the output array (RAM or memmap)
            out_slice = patches[chunk_start:chunk_end]

            if workers <= 1:
                for i, c in enumerate(chunk_coords):
                    _read_patch_into(slide_path, c, level, read_size, patch_size, out_slice, i)
                    pbar.update(1)
            else:
                futures = {
                    pool.submit(_read_patch_into, slide_path, c, level,
                                read_size, patch_size, out_slice, i): i
                    for i, c in enumerate(chunk_coords)
                }
                for f in as_completed(futures):
                    f.result()
                    pbar.update(1)

            if not in_memory:
                patches.flush()

    if workers <= 1:
        pool = None
        _process_chunks()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            _process_chunks()

    pbar.close()

    if in_memory:
        return n, coords, patches
    else:
        del patches
        return n, coords


def tile_slide(slide_path, out_dir, patch_size=224, workers=1, max_patches=None,
               in_memory=False):
    """
    Full tiling pipeline for a single slide.

    When in_memory=False (default), writes patches.npy to out_dir via memmap.
    When in_memory=True, returns patches array directly in RAM — no disk I/O.

    Returns:
        num_patches: number of patches extracted
        coords: list of (x, y) full-resolution coordinates
        patches (only if in_memory=True): numpy array of shape (N, 224, 224, 3)
    """
    slide = openslide.OpenSlide(str(slide_path))

    print(f"  Dimensions: {slide.dimensions}")
    print(f"  Levels: {slide.level_count}")

    level, level_scale = get_20x_level(slide)
    print(f"  Using level {level} (scale {level_scale:.2f}) for 20x")

    mask, scale_x, scale_y = detect_tissue(slide)
    slide.close()

    patches_path = Path(out_dir) / "patches.npy"
    result = extract_patches(
        slide_path, mask, scale_x, scale_y, level, level_scale,
        patches_path, patch_size, workers=workers, max_patches=max_patches,
        in_memory=in_memory
    )

    if in_memory:
        num_patches, coords, patches = result
        print(f"  Extracted {num_patches} patches (in-memory)")
        return num_patches, coords, patches
    else:
        num_patches, coords = result
        print(f"  Extracted {num_patches} patches")
        return num_patches, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str, help="path to a single .svs file")
    parser.add_argument("--all", action="store_true", help="process all slides")
    parser.add_argument("--wsi-dir", default="data/wsi")
    parser.add_argument("--out-dir", default="data/tiles")
    parser.add_argument("--max-patches", type=int, default=80000,
                        help="max patches per slide; randomly sample if exceeded (default: 80000)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="number of parallel workers (default: all CPUs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    workers = max(1, args.workers)
    print(f"Using {workers} workers")

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

        num_patches, coords = tile_slide(slide_path, slide_out, workers=workers,
                                         max_patches=args.max_patches)

        # Save coordinates for heatmap reconstruction
        np.save(coord_path, np.array(coords))

        # Clean up thread-local OpenSlide handles between slides
        _close_thread_handles()

        print(f"  Saved to {slide_out}")


if __name__ == "__main__":
    main()