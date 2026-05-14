"""One-shot builder for notebooks/hest_extract_colab.ipynb.

This notebook is a thin Colab wrapper around the pipeline script
`08a_extract_features.py`. The notebook only handles Colab-specific
concerns (Drive mount, HF login, env setup) and then shells out to
the script with `--source-group hest_visium`. All extraction logic
(download from HF + tile + UNI + save .h5) lives in 08a so the
pipeline is invocable from a plain CLI as well.

Run order on Colab:
    06 -> 07 -> 08 -> THIS NOTEBOOK + gtex_extract_colab.ipynb ->
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


md("""# HEST Feature Extraction (Colab)

Thin Colab wrapper around the pipeline script
`08a_extract_features.py`. The script does the actual work
(download HEST `.tif` from `MahmoodLab/hest`, tile in-memory, run
UNI, save `.h5` to Drive at `embeddings/uni_hest/{file_id}.h5`).
This notebook only handles Drive mount + HF login + invoking the
script with streamed output.

**Prerequisites**
- Hugging Face access to `MahmoodLab/hest` (gated; approved
  2026-05-14 for this project) and `MahmoodLab/uni`.
- Diagnostic manifest tracked in the repo at
  `data/expression/diagnostic_manifest.csv`. The notebook reads
  the cloned copy directly (no Drive sync needed). Push manifest
  updates from your laptop with `git push`; the notebook's Cell 1
  `git pull --ff-only` picks them up on the next run.

**Runtime** - L4 GPU, ~88 slides: roughly 25-40 min wall-clock
(HF download bandwidth is the bottleneck; GPU-resident extract is
~5 sec per slide).
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

code("""# Cell 2: HuggingFace login (MahmoodLab/hest + MahmoodLab/uni are gated).
from huggingface_hub import login
login()
""")

code("""# Cell 3: Paths.
from pathlib import Path

LOCAL_REPO = Path('/content/prame-predict')
# Manifest comes from the cloned repo (tracked in git), not Drive.
# Push manifest updates with `git push` from your laptop; this notebook's
# Cell 1 `git pull --ff-only` picks them up before this cell runs.
MANIFEST   = LOCAL_REPO / 'data' / 'expression' / 'diagnostic_manifest.csv'

DRIVE_ROOT = Path('/content/drive/MyDrive/prame-predict')
EMB_DIR    = DRIVE_ROOT / 'embeddings'           # 08a appends uni_hest/

assert MANIFEST.exists(), (
    f'Manifest not found at {MANIFEST}. Run 08_build_diagnostic_manifest.py '
    'on your laptop, commit the CSV, and `git push`; then re-run Cell 1 to '
    '`git pull --ff-only` here.'
)
print(f'Manifest: {MANIFEST}  (from cloned repo)')
print(f'Embeddings dir (cohort subdir auto-appended): {EMB_DIR}')
""")

code("""# Cell 4: Run 08a_extract_features.py with streaming output.
import os, select, subprocess, time

cmd = [
    'python', '-u', str(LOCAL_REPO / '08a_extract_features.py'),
    '--source-group', 'hest_visium',
    '--manifest', str(MANIFEST),
    '--emb-dir',  str(EMB_DIR),
    '--device',   'cuda',
    '--amp',
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
HEST_EMB_DIR = EMB_DIR / 'uni_hest'
on_drive = sorted(HEST_EMB_DIR.glob('*.h5')) if HEST_EMB_DIR.exists() else []
print(f'embeddings/uni_hest on Drive: {len(on_drive)} .h5 files')

import pandas as pd
df = pd.read_csv(MANIFEST)
total = int((df['source_group'] == 'hest_visium').sum())
print(f'manifest hest_visium rows:    {total}')
missing = total - len(on_drive)
if missing > 0:
    print(f'  [warn] {missing} hest_visium rows still missing on Drive.')
    print('         Inspect the failed list in the cell above and re-run if transient.')
else:
    print('  All hest_visium rows have embeddings.')
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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "hest_extract_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out_path}  ({out_path.stat().st_size} bytes, {len(cells)} cells)")
