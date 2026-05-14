"""One-shot builder for notebooks/gtex_extract_colab.ipynb.

This notebook is a thin Colab wrapper around the pipeline script
`08a_extract_features.py`. The notebook only handles Colab-specific
concerns (Drive mount, HF login for UNI, env setup) and then shells
out to the script with `--source-group gtex_normal`. All extraction
logic (BRD HTTP stream + tile + UNI + save .h5) lives in 08a so the
pipeline is invocable from a plain CLI as well.

Run order on Colab:
    06 -> 07 -> 08 -> hest_extract_colab.ipynb + THIS NOTEBOOK ->
    09 tuning notebook -> 10 train.
"""

import json
from pathlib import Path

cells = []


def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    })


def md(src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    })


md("""# GTEx Feature Extraction (Colab)

Thin Colab wrapper around the pipeline script
`08a_extract_features.py`. The script does the actual work
(stream each WSI from the BRD URL in the manifest, tile in-memory,
run UNI, save `.h5` to Drive at
`embeddings/uni_gtex/{file_id}.h5`). This notebook only handles
Drive mount + HF login (for UNI weights) + invoking the script
with streamed output.

**Prerequisites**
- Hugging Face access to `MahmoodLab/uni` (for UNI feature
  extractor weights).
- Diagnostic manifest at
  `<Drive>/prame-predict/data/expression/diagnostic_manifest.csv`
  with `source_group=='gtex_normal'` rows whose `download_url`
  points at BRD.

**Runtime** - L4 GPU, ~200 slides: roughly 45-90 min wall-clock
(BRD bandwidth is the bottleneck; GPU-resident extract is ~5 sec
per slide).
""")

code("""# Cell 1: Install dependencies, mount Drive, clone repo.
!pip install -q timm huggingface_hub openslide-python h5py opencv-python-headless
!apt-get install -qq -y openslide-tools

from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('prame-predict'):
    !git clone https://github.com/hb-1968/prame-predict.git
else:
    !cd prame-predict && git pull --ff-only
""")

code("""# Cell 2: HuggingFace login (MahmoodLab/uni is gated).
from huggingface_hub import login
login()
""")

code("""# Cell 3: Paths.
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/prame-predict')
MANIFEST   = DRIVE_ROOT / 'data' / 'expression' / 'diagnostic_manifest.csv'
EMB_DIR    = DRIVE_ROOT / 'embeddings'           # 08a appends uni_gtex/

LOCAL_REPO = Path('/content/prame-predict')

assert MANIFEST.exists(), (
    f'Manifest not found at {MANIFEST}. Run 08_build_diagnostic_manifest.py '
    'and sync data/expression/ to Drive first.'
)
print(f'Manifest: {MANIFEST}')
print(f'Embeddings dir (cohort subdir auto-appended): {EMB_DIR}')
""")

code("""# Cell 4: Run 08a_extract_features.py with streaming output.
import os, select, subprocess, time

cmd = [
    'python', '-u', str(LOCAL_REPO / '08a_extract_features.py'),
    '--source-group', 'gtex_normal',
    '--manifest', str(MANIFEST),
    '--emb-dir',  str(EMB_DIR),
    '--device',   'cuda',
    '--amp',
    '--download-workers', '8',
]
print('Command:')
print('  ' + ' '.join(cmd))
print()

env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
t0 = time.time()
proc = subprocess.Popen(
    cmd, cwd=str(LOCAL_REPO),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=env,
)

try:
    while True:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end='', flush=True)
        if proc.poll() is not None:
            tail = proc.stdout.read()
            if tail:
                print(tail, end='', flush=True)
            break
finally:
    rc = proc.wait()

print(f'\\nFinished in {(time.time() - t0) / 60:.1f} min  (exit code {rc})')
""")

code("""# Cell 5: QC counts.
GTEX_EMB_DIR = EMB_DIR / 'uni_gtex'
on_drive = sorted(GTEX_EMB_DIR.glob('*.h5')) if GTEX_EMB_DIR.exists() else []
print(f'embeddings/uni_gtex on Drive: {len(on_drive)} .h5 files')

import pandas as pd
df = pd.read_csv(MANIFEST)
total = int((df['source_group'] == 'gtex_normal').sum())
print(f'manifest gtex_normal rows:    {total}')
missing = total - len(on_drive)
if missing > 0:
    print(f'  [warn] {missing} gtex_normal rows still missing on Drive.')
    print('         Inspect the failed list in the cell above and re-run if transient.')
else:
    print('  All gtex_normal rows have embeddings.')
""")


nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "gtex_extract_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out_path}  ({out_path.stat().st_size} bytes, {len(cells)} cells)")
