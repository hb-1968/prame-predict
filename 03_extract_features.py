"""
Extract foundation model features from tiled WSI patches.

Loads patches for each slide, passes them through a frozen encoder
in batches, and saves the feature matrix as an HDF5 file.

Usage:
    python 03_extract_features.py --model uni --slide data/tiles/TCGA-XX-XXXX
    python 03_extract_features.py --model uni --all
    python 03_extract_features.py --model uni --pipeline  # tile + extract + cleanup per slide
"""

import argparse
import shutil
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
torch.set_float32_matmul_precision('medium')
import timm
import numpy as np
import h5py
import threading
import queue
from pathlib import Path
from tqdm import tqdm


def load_uni():
    """
    Load UNI with the LayerScale fix.
    Same approach validated in pathbench.
    """
    model = timm.create_model(
        "vit_large_patch16_224",
        init_values=1e-5,
        num_classes=0,
        pretrained=False,
    )

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/uni",
        filename="pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    return model


def load_conch():
    """
    Load CONCH visual encoder.
    CONCH is a CoCa-based model; we extract just the vision encoder
    for feature extraction, discarding the language components.
    """
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/conch",
        filename="pytorch_model.bin",
    )

    # CONCH uses a custom architecture — load via its own create function
    # timm model name for the vision backbone
    model = timm.create_model(
        "vit_base_patch16_224",
        num_classes=0,
        pretrained=False,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # CONCH checkpoint may have prefixed keys — filter for vision encoder
    vision_keys = {
        k.replace("visual.", ""): v
        for k, v in state_dict.items()
        if k.startswith("visual.")
    }

    if vision_keys:
        model.load_state_dict(vision_keys, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    return model


def load_model(model_name):
    """Load the requested model and return it with device info."""
    print(f"Loading model: {model_name}")

    if model_name == "uni":
        model = load_uni()
    elif model_name == "conch":
        model = load_conch()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running on: {device}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    return model, device


def _preprocess_batch(batch_np, device, use_amp):
    """
    Vectorized preprocessing: uint8 numpy (N,224,224,3) → normalized tensor.
    Replaces per-patch PIL.fromarray + timm transform pipeline.
    Both UNI and CONCH use mean=0.5, std=0.5, input_size=224×224.
    """
    # (N, H, W, C) uint8 → (N, C, H, W) float, fused scale+normalize
    # x/255*2-1 = x/127.5-1 — single multiply+add instead of three ops
    batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    dtype = torch.float16 if use_amp else torch.float32
    batch = batch.to(dtype=dtype).div_(127.5).sub_(1.0)
    return batch.to(device, non_blocking=True)


class _BatchPrefetcher:
    """
    Prefetch and preprocess the next batch in a background thread
    while the model processes the current batch.
    """
    def __init__(self, patches, batch_size, device, use_amp):
        self._patches = patches
        self._batch_size = batch_size
        self._device = device
        self._use_amp = use_amp
        self._queue = queue.Queue(maxsize=2)
        self._n = len(patches)
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        for i in range(0, self._n, self._batch_size):
            batch_np = np.array(self._patches[i:i + self._batch_size])
            tensor = _preprocess_batch(batch_np, self._device, self._use_amp)
            self._queue.put(tensor)
        self._queue.put(None)  # sentinel

    def __iter__(self):
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            yield batch


def extract_slide_features(slide_dir, model, device, batch_size=64):
    """
    Extract features for all patches in a single slide directory.

    Uses vectorized numpy→torch preprocessing instead of per-patch PIL
    transforms. On GPU, uses float16 inference via torch.amp for ~2x
    throughput.

    Args:
        slide_dir: Path containing patches.npy and coords.npy
        model: frozen encoder
        device: cuda or cpu
        batch_size: images per forward pass (64 for GPU, 16 for CPU)

    Returns:
        features: numpy array of shape (num_patches, feature_dim)
        coords: numpy array of shape (num_patches, 2)
    """
    patches_path = slide_dir / "patches.npy"

    if not patches_path.exists():
        print(f"  No patches.npy found in {slide_dir}")
        return None, None

    # Memory-map patches — reads from disk on demand, no full load into RAM
    patches = np.load(patches_path, mmap_mode='r')
    coords = np.load(slide_dir / "coords.npy")

    if len(patches) == 0:
        print(f"  Empty patches array in {slide_dir}")
        return None, None

    use_amp = device.type == "cuda"
    n = len(patches)
    num_batches = (n + batch_size - 1) // batch_size

    # Pre-allocate output — avoids list accumulation + vstack copy
    # Feature dim: UNI=1024, CONCH=768. Detect from first batch.
    features = None
    write_idx = 0

    prefetcher = _BatchPrefetcher(patches, batch_size, device, use_amp)

    with torch.inference_mode():
        for batch in tqdm(prefetcher, total=num_batches,
                          desc=f"  Extracting", leave=False):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    feats = model(batch.float())
            else:
                feats = model(batch)

            feats_np = feats.cpu().numpy().astype(np.float16)

            if features is None:
                features = np.empty((n, feats_np.shape[1]), dtype=np.float16)

            features[write_idx:write_idx + len(feats_np)] = feats_np
            write_idx += len(feats_np)

    del patches

    return features, coords


def save_features(out_path, features, coords, model_name, slide_name):
    """Save extracted features as compressed HDF5."""
    with h5py.File(out_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        f.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        f.attrs["model"] = model_name
        f.attrs["slide_name"] = slide_name
        f.attrs["num_patches"] = features.shape[0]
        f.attrs["feature_dim"] = features.shape[1]
    print(f"  Saved {features.shape} to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["uni", "conch"])
    parser.add_argument("--slide", type=str, help="path to a single tile dir")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pipeline", action="store_true",
                        help="tile + extract + cleanup per slide (minimal disk usage)")
    parser.add_argument("--wsi-dir", default="data/wsi")
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--out-dir", default="embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=80000,
                        help="max patches per slide for --pipeline mode (default: 80000)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="tiling workers for --pipeline mode (default: all CPUs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once, reuse for all slides
    model, device = load_model(args.model)

    # Adjust batch size for CPU
    if device.type == "cpu" and args.batch_size > 16:
        print(f"CPU detected, reducing batch size to 16")
        args.batch_size = 16

    if args.pipeline:
        # Pipeline mode: tile → extract → delete, one slide at a time
        import importlib
        tile_mod = importlib.import_module("02_tile_wsi")
        tile_slide = tile_mod.tile_slide
        _close_thread_handles = tile_mod._close_thread_handles
        slides = sorted(Path(args.wsi_dir).glob("*.svs"))
        print(f"\nPipeline mode: {len(slides)} slides")
        tiles_dir = Path(args.tiles_dir)

        for i, slide_path in enumerate(slides):
            slide_name = slide_path.stem
            out_path = out_dir / f"{slide_name}.h5"

            if out_path.exists():
                print(f"[{i+1}/{len(slides)}] Skipping {slide_name} (exists)")
                continue

            print(f"\n[{i+1}/{len(slides)}] {slide_name}")
            slide_out = tiles_dir / slide_name
            slide_out.mkdir(parents=True, exist_ok=True)

            # Step 1: Tile (writes patches.npy via memmap, bounded RAM)
            num_patches, coords = tile_slide(slide_path, slide_out, workers=max(1, args.workers),
                                             max_patches=args.max_patches)
            np.save(slide_out / "coords.npy", np.array(coords))
            _close_thread_handles()

            # Step 2: Extract features
            features, coords = extract_slide_features(
                slide_out, model, device, args.batch_size
            )

            # Step 3: Cleanup tiles
            shutil.rmtree(slide_out)
            print(f"  Cleaned up {slide_out}")

            if features is None:
                continue

            save_features(out_path, features, coords, args.model, slide_name)

    else:
        # Standard mode: tiles already exist on disk
        if args.slide:
            slide_dirs = [Path(args.slide)]
        elif args.all:
            slide_dirs = sorted(
                [d for d in Path(args.tiles_dir).iterdir() if d.is_dir()]
            )
        else:
            print("Specify --slide <path>, --all, or --pipeline")
            return

        print(f"\nProcessing {len(slide_dirs)} slides")

        for i, slide_dir in enumerate(slide_dirs):
            slide_name = slide_dir.name
            out_path = out_dir / f"{slide_name}.h5"

            if out_path.exists():
                print(f"[{i+1}/{len(slide_dirs)}] Skipping {slide_name} (exists)")
                continue

            print(f"[{i+1}/{len(slide_dirs)}] {slide_name}")

            features, coords = extract_slide_features(
                slide_dir, model, device, args.batch_size
            )

            if features is None:
                continue

            save_features(out_path, features, coords, args.model, slide_name)


if __name__ == "__main__":
    main()