"""
Cohort-aware H&E feature extractor for Component 2.

Reads `data/expression/diagnostic_manifest.csv`, filters by
`--source-group`, downloads each WSI via that cohort's appropriate
channel (HuggingFace for HEST, BRD HTTP for GTEx), tiles in-memory
via `02_tile_wsi.py:tile_slide`, extracts UNI features via
`03_extract_features.py:load_uni`, and saves the `.h5` to
`<emb_dir>/uni_<cohort_suffix>/{file_id}.h5`. Resumable: any slide
that already has a `.h5` at the target path is skipped.

This script is the numbered-pipeline equivalent of the per-cohort
Colab notebooks under `notebooks/`. The notebooks (HEST + GTEx)
are thin wrappers that mount Drive and invoke this script. SKCM
features come from Component 1 (`03_extract_features.py` proper);
COBRA features come from `notebooks/cobra_predict_colab.ipynb`
which also runs Component-1 ensemble prediction.

CONCH is intentionally unsupported. Component-1 showed CONCH near
chance for PRAME, so the diagnostic feature surface is UNI only.

Usage:
    # HEST (Visium-tissue) — pulls .tif from MahmoodLab/hest
    python 08a_extract_features.py --source-group hest_visium

    # GTEx normal-skin — streams .svs from BRD URLs in the manifest
    python 08a_extract_features.py --source-group gtex_normal

    # Smoke test (one slide, CPU)
    python 08a_extract_features.py --source-group hest_visium \\
        --limit 1 --device cpu --emb-dir /tmp/emb_smoketest
"""

import argparse
import os
import queue
import shutil
import sys
import threading
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sibling-script loading (numeric prefix is not a valid module name; project
# convention is spec_from_file_location).
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name, filename):
    spec = spec_from_file_location(name, str(SCRIPT_DIR / filename))
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tile_mod = _load_module("_tile_02", "02_tile_wsi.py")
tile_slide = _tile_mod.tile_slide
_close_thread_handles = _tile_mod._close_thread_handles

_ext_mod = _load_module("_ext_03", "03_extract_features.py")
load_uni = _ext_mod.load_uni  # CONCH intentionally not used


# ---------------------------------------------------------------------------
# Cohort dispatch
# ---------------------------------------------------------------------------

# Keep these in sync with SOURCE_EMB_SUBDIR in 10_train_component2.py.
COHORT_EMB_SUBDIR = {
    "hest_visium":   "{model}_hest",
    "gtex_normal":   "{model}_gtex",
}

# HF candidate paths inside MahmoodLab/hest. Tried in order; first match wins
# for the run.
HEST_REPO_ID = "MahmoodLab/hest"
HEST_WSI_CANDIDATES = (
    "wsis/{rid}.tif",
    "wsis_pyramidal/{rid}.tif",
    "wsis/{rid}.ome.tif",
)


def hest_downloader(state, row, local_path):
    """Download a HEST WSI from HuggingFace. Caches the working path template
    in `state` after the first success so subsequent slides skip the probe."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    file_id = str(row["file_id"])
    candidates = ([state["template"]] if state.get("template")
                  else list(HEST_WSI_CANDIDATES))
    last_err = None
    for tmpl in candidates:
        remote = tmpl.format(rid=file_id)
        try:
            cached = hf_hub_download(
                repo_id=HEST_REPO_ID,
                filename=remote,
                repo_type="dataset",
                cache_dir=str(state["hf_cache_dir"]),
            )
            shutil.copy2(cached, local_path)
            try:
                os.remove(cached)
            except OSError:
                pass
            state["template"] = tmpl
            return local_path
        except (EntryNotFoundError, FileNotFoundError) as e:
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise FileNotFoundError(
        f"HEST download failed for {file_id!r}. Tried {candidates}. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def brd_downloader(state, row, local_path):
    """Stream a GTEx WSI from the BRD URL in the manifest.

    Validates the response is actually a TIFF/SVS (Content-Type, min size,
    magic bytes) before renaming the .tmp to the final path. BRD returns
    HTTP 200 with an HTML page when the sample ID is wrong or the asset
    is unavailable, so without these checks failures look identical to
    "OpenSlideUnsupportedFormatError" downstream.
    """
    import requests

    url = row.get("download_url")
    if not isinstance(url, str) or not url:
        raise FileNotFoundError(
            f"row {row.get('file_id')!r} has no download_url"
        )
    session = state["session"]
    timeout = state["timeout"]
    chunk = state["chunk_bytes"]
    if local_path.exists():
        return local_path
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "") or ""
        status_code = r.status_code
        with open(tmp, "wb") as fh:
            for piece in r.iter_content(chunk_size=chunk):
                if piece:
                    fh.write(piece)
    size = tmp.stat().st_size
    with open(tmp, "rb") as fh:
        head = fh.read(16)
    magic = head[:4]
    bad_type = content_type.lower().startswith("text/")
    too_small = size < 1_000_000
    bad_magic = magic not in (b"II*\x00", b"MM\x00*")
    if bad_type or too_small or bad_magic:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"BRD returned a non-TIFF payload for {row.get('file_id')!r}. "
            f"url={url}  http={status_code}  content-type={content_type!r}  "
            f"size={size}B  first16={head.hex()}  "
            f"(bad_type={bad_type} too_small={too_small} bad_magic={bad_magic})"
        )
    tmp.rename(local_path)
    return local_path


COHORT_DOWNLOADER = {
    "hest_visium": hest_downloader,
    "gtex_normal": brd_downloader,
}


def _file_name_for(row, default_ext):
    """Pick a sensible local filename for a manifest row."""
    base = row.get("file_name") or row["file_id"]
    base = str(base)
    if not (base.endswith(".svs") or base.endswith(".tif")
            or base.endswith(".tiff") or base.endswith(".ome.tif")):
        base = f"{row['file_id']}{default_ext}"
    return base


# ---------------------------------------------------------------------------
# UNI feature extractor (GPU-resident or prefetcher branch, lifted from
# notebooks/cobra_predict_colab.ipynb).
# ---------------------------------------------------------------------------

def _preprocess_batch(batch_np, device):
    batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    batch = batch.to(dtype=torch.float16).div_(127.5).sub_(1.0)
    return batch.to(device, non_blocking=True)


class _BatchPrefetcher:
    def __init__(self, patches, batch_size, device):
        self._patches = patches
        self._bs = batch_size
        self._n = len(patches)
        self._device = device
        self._q = queue.Queue(maxsize=2)
        threading.Thread(target=self._produce, daemon=True).start()

    def _produce(self):
        for i in range(0, self._n, self._bs):
            self._q.put(_preprocess_batch(
                np.array(self._patches[i:i + self._bs]),
                self._device,
            ))
        self._q.put(None)

    def __iter__(self):
        while (b := self._q.get()) is not None:
            yield b


def extract_features(patches, model, device, batch_size, amp=False):
    n = len(patches)
    num_batches = (n + batch_size - 1) // batch_size
    features = None
    write_idx = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gpu_free = torch.cuda.mem_get_info()[0]
    else:
        gpu_free = 0
    patch_bytes = n * 3 * 224 * 224 * 2
    use_gpu_resident = (
        device.type == "cuda"
        and (patch_bytes * 1.5) < (gpu_free - 2 * 1024 ** 3)
    )
    mode = "GPU-resident" if use_gpu_resident else "prefetcher" if device.type == "cuda" else "CPU"
    print(f"    {mode}: {patch_bytes / 1024 ** 3:.2f} GB patches"
          f"{f' / {gpu_free / 1024 ** 3:.2f} GB free' if device.type == 'cuda' else ''}",
          flush=True)

    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else _NullCtx())

    with torch.inference_mode():
        if use_gpu_resident:
            all_patches = torch.from_numpy(np.array(patches)).permute(0, 3, 1, 2)
            all_patches = all_patches.to(device=device, dtype=torch.float16).div_(127.5).sub_(1.0)
            for i in range(0, n, batch_size):
                with autocast:
                    feats = model(all_patches[i:i + batch_size].float())
                feats_np = feats.half().cpu().numpy()
                if features is None:
                    features = np.empty((n, feats_np.shape[1]), dtype=np.float16)
                features[write_idx:write_idx + len(feats_np)] = feats_np
                write_idx += len(feats_np)
            del all_patches
            torch.cuda.empty_cache()
        else:
            iterator = (
                _BatchPrefetcher(patches, batch_size, device)
                if device.type == "cuda"
                else _cpu_batch_iter(patches, batch_size, device)
            )
            for batch in iterator:
                with autocast:
                    feats = model(batch.float())
                feats_np = feats.half().cpu().numpy()
                if features is None:
                    features = np.empty((n, feats_np.shape[1]), dtype=np.float16)
                features[write_idx:write_idx + len(feats_np)] = feats_np
                write_idx += len(feats_np)
    return features


def _cpu_batch_iter(patches, batch_size, device):
    n = len(patches)
    for i in range(0, n, batch_size):
        yield _preprocess_batch(np.array(patches[i:i + batch_size]), device)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def save_h5(path, features, coords, slide_name, model_name):
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features,
                         compression="gzip", compression_opts=4)
        f.create_dataset("coords", data=coords,
                         compression="gzip", compression_opts=4)
        f.attrs["model"] = model_name
        f.attrs["slide_name"] = slide_name
        f.attrs["num_patches"] = features.shape[0]
        f.attrs["feature_dim"] = features.shape[1]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("  [warn] --device cuda requested but CUDA not available; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def _emb_subdir(cohort, model):
    template = COHORT_EMB_SUBDIR.get(cohort)
    if template is None:
        raise ValueError(f"no SOURCE_EMB_SUBDIR mapping for cohort {cohort!r}")
    return template.format(model=model)


def parse_args():
    p = argparse.ArgumentParser(
        description="Component-2 H&E feature extractor (HEST / GTEx).",
    )
    p.add_argument("--source-group", required=True,
                   choices=tuple(COHORT_EMB_SUBDIR.keys()),
                   help="Manifest source_group to extract.")
    p.add_argument("--manifest",
                   default="data/expression/diagnostic_manifest.csv")
    p.add_argument("--emb-dir", default="embeddings",
                   help="Embedding root; cohort subdir is appended.")
    p.add_argument("--model", choices=("uni",), default="uni",
                   help="UNI only; CONCH intentionally not exposed.")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"),
                   default="auto")
    p.add_argument("--max-patches", type=int, default=80000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tile-workers", type=int, default=16)
    p.add_argument("--download-workers", type=int, default=8,
                   help="Parallel downloads (currently only used by GTEx).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N rows (0 = no limit).")
    p.add_argument("--no-resume", action="store_true",
                   help="Process all rows even if .h5 already exists.")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast for forward pass (CUDA only).")
    p.add_argument("--local-wsi-dir", default="",
                   help="Where to stage downloaded WSIs locally. "
                        "Defaults to /tmp/prame_extract_wsi/.")
    p.add_argument("--local-tile-dir", default="",
                   help="Where to stage scratch tile dirs locally. "
                        "Defaults to /tmp/prame_extract_tiles/.")
    return p.parse_args()


def _build_state(cohort, args):
    """Cohort-specific shared state (cache dirs, HTTP session, etc.)."""
    state = {}
    if cohort == "hest_visium":
        state["template"] = None  # Set on first success.
        state["hf_cache_dir"] = Path(
            args.local_wsi_dir or "/tmp/prame_extract_wsi"
        ) / "hf_cache"
        state["hf_cache_dir"].mkdir(parents=True, exist_ok=True)
    elif cohort == "gtex_normal":
        import requests
        session = requests.Session()
        session.headers.update({"User-Agent": "prame-predict/08a-extract"})
        state["session"] = session
        state["timeout"] = 600
        state["chunk_bytes"] = 1 << 20  # 1 MiB
    return state


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    cohort = args.source_group
    model_name = args.model

    print("=" * 64)
    print(f"08a_extract_features: cohort={cohort}, model={model_name}")
    print("=" * 64)
    print(f"Device:      {device}")
    print(f"Manifest:    {args.manifest}")
    print(f"Embeddings:  {args.emb_dir}/{_emb_subdir(cohort, model_name)}/")
    print(f"Resume:      {not args.no_resume}")
    print(f"AMP:         {'bf16' if args.amp else 'fp32'}")
    print()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run 08_build_diagnostic_manifest.py first."
        )
    df = pd.read_csv(manifest_path)
    rows = df[df["source_group"] == cohort].copy().reset_index(drop=True)
    print(f"Manifest rows for {cohort}: {len(rows)}")

    emb_subdir = Path(args.emb_dir) / _emb_subdir(cohort, model_name)
    emb_subdir.mkdir(parents=True, exist_ok=True)
    local_wsi_dir = Path(args.local_wsi_dir or "/tmp/prame_extract_wsi")
    local_tile_dir = Path(args.local_tile_dir or "/tmp/prame_extract_tiles")
    local_wsi_dir.mkdir(parents=True, exist_ok=True)
    local_tile_dir.mkdir(parents=True, exist_ok=True)

    def _emb_path(file_id):
        return emb_subdir / f"{file_id}.h5"

    if not args.no_resume:
        before = len(rows)
        rows = rows[~rows["file_id"].apply(
            lambda r: _emb_path(str(r)).exists()
        )].reset_index(drop=True)
        print(f"  Already extracted: {before - len(rows)}")
        print(f"  Remaining:         {len(rows)}")

    if args.limit and args.limit > 0:
        rows = rows.head(args.limit).reset_index(drop=True)
        print(f"  --limit applied: processing {len(rows)} rows")

    if rows.empty:
        print("\nNothing to do.")
        return

    print("\nLoading UNI...")
    model = load_uni()
    model.eval().to(device)
    print("UNI ready.\n")

    downloader = COHORT_DOWNLOADER[cohort]
    state = _build_state(cohort, args)
    default_ext = ".tif" if cohort == "hest_visium" else ".svs"

    extracted, failed = [], []
    for idx, row in rows.iterrows():
        file_id = str(row["file_id"])
        slide_name = _file_name_for(row, default_ext)
        local_path = local_wsi_dir / slide_name
        emb_path = _emb_path(file_id)
        slide_out = local_tile_dir / file_id

        if emb_path.exists() and not args.no_resume:
            continue

        print(f"[{idx + 1}/{len(rows)}] {file_id}")
        try:
            downloader(state, row, local_path)
        except Exception as e:  # noqa: BLE001
            print(f"  download failed: {type(e).__name__}: {e}")
            failed.append((file_id, "download"))
            continue

        try:
            num_patches, coords, patches = tile_slide(
                local_path, slide_out,
                workers=args.tile_workers,
                max_patches=args.max_patches,
                in_memory=True,
            )
            _close_thread_handles()
        except Exception as e:  # noqa: BLE001
            print(f"  tile failed: {type(e).__name__}: {e}")
            failed.append((file_id, "tile"))
            local_path.unlink(missing_ok=True)
            continue

        if num_patches == 0:
            print("  0 patches; skipping")
            failed.append((file_id, "zero_patches"))
            local_path.unlink(missing_ok=True)
            if slide_out.exists():
                shutil.rmtree(slide_out, ignore_errors=True)
            continue

        try:
            features = extract_features(
                patches, model, device, args.batch_size, amp=args.amp,
            )
            del patches
        except Exception as e:  # noqa: BLE001
            print(f"  extract failed: {type(e).__name__}: {e}")
            failed.append((file_id, "extract"))
            local_path.unlink(missing_ok=True)
            if slide_out.exists():
                shutil.rmtree(slide_out, ignore_errors=True)
            continue

        save_h5(emb_path, features, np.array(coords), slide_name, model_name)
        extracted.append(file_id)
        print(f"  saved {features.shape} -> {emb_path}")
        del features

        local_path.unlink(missing_ok=True)
        if slide_out.exists():
            shutil.rmtree(slide_out, ignore_errors=True)

    # Final cleanup of staging dirs.
    for f in local_wsi_dir.glob("*"):
        try:
            if f.is_file():
                f.unlink()
        except OSError:
            pass
    for d in local_tile_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    print()
    print("=" * 64)
    print(f"Extraction complete for {cohort}.")
    print(f"  extracted: {len(extracted)}")
    print(f"  failed:    {len(failed)}")
    if failed:
        print()
        print("Failed slides:")
        for fid, why in failed:
            print(f"  {fid:18s}  ({why})")
        print()
        print("Re-run with the same command to retry (resume skips successes).")


if __name__ == "__main__":
    main()
