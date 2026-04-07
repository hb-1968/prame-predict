"""
Extract foundation model features from tiled WSI patches.

Loads patches for each slide, passes them through a frozen encoder
in batches, and saves the feature matrix as an HDF5 file.

Usage:
    python 03_extract_features.py --model uni --slide data/tiles/TCGA-XX-XXXX
    python 03_extract_features.py --model uni --all
"""

import argparse
import torch
import timm
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config, create_transform
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
    """Load the requested model and its preprocessing transform."""
    print(f"Loading model: {model_name}")

    if model_name == "uni":
        model = load_uni()
    elif model_name == "conch":
        model = load_conch()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    # Get preprocessing config from the base architecture
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running on: {device}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    return model, transform, device


def extract_slide_features(slide_dir, model, transform, device,
                           batch_size=64):
    """
    Extract features for all patches in a single slide directory.

    Args:
        slide_dir: Path containing numbered JPEG patches and coords.npy
        model: frozen encoder
        transform: preprocessing pipeline
        device: cuda or cpu
        batch_size: images per forward pass (64 for GPU, 16 for CPU)

    Returns:
        features: numpy array of shape (num_patches, feature_dim)
        coords: numpy array of shape (num_patches, 2)
    """
    # Find all patch images, sorted by number
    patch_paths = sorted(slide_dir.glob("*.jpg"))

    if len(patch_paths) == 0:
        print(f"  No patches found in {slide_dir}")
        return None, None

    coords = np.load(slide_dir / "coords.npy")

    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(patch_paths), batch_size),
                      desc=f"  Extracting", leave=False):
            batch_paths = patch_paths[i:i + batch_size]
            tensors = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))

            batch = torch.stack(tensors).to(device)
            feats = model(batch)
            features.append(feats.cpu().numpy().astype(np.float16))


    features = np.vstack(features)
    return features, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["uni", "conch"])
    parser.add_argument("--slide", type=str, help="path to a single tile dir")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--out-dir", default="embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.slide:
        slide_dirs = [Path(args.slide)]
    elif args.all:
        slide_dirs = sorted(
            [d for d in Path(args.tiles_dir).iterdir() if d.is_dir()]
        )
    else:
        print("Specify --slide <path> or --all")
        return

    # Load model once, reuse for all slides
    model, transform, device = load_model(args.model)

    # Adjust batch size for CPU
    if device.type == "cpu" and args.batch_size > 16:
        print(f"CPU detected, reducing batch size to 16")
        args.batch_size = 16

    print(f"\nProcessing {len(slide_dirs)} slides")

    for i, slide_dir in enumerate(slide_dirs):
        slide_name = slide_dir.name
        out_path = out_dir / f"{slide_name}.h5"

        if out_path.exists():
            print(f"[{i+1}/{len(slide_dirs)}] Skipping {slide_name} (exists)")
            continue

        print(f"[{i+1}/{len(slide_dirs)}] {slide_name}")

        features, coords = extract_slide_features(
            slide_dir, model, transform, device, args.batch_size
        )

        if features is None:
            continue

        # Save as HDF5 — efficient for large arrays, supports metadata
        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=features, compression="gzip", compression_opts=9)
            f.create_dataset("coords", data=coords, compression_opts = 9)
            f.attrs["model"] = args.model
            f.attrs["slide_name"] = slide_name
            f.attrs["num_patches"] = features.shape[0]
            f.attrs["feature_dim"] = features.shape[1]

        print(f"  Saved {features.shape} to {out_path}")


if __name__ == "__main__":
    main()