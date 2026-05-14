"""One-shot builder for notebooks/tune_component2_colab.ipynb.

Mirrors scripts/build_cobra_colab_notebook.py. Kept as a tracked script
so the notebook can be regenerated / diffed without hand-editing JSON.

The generated notebook:
  1. Mounts Drive, installs optuna, clones the repo.
  2. Detects GPU tier (T4 / L4 / A100 / A100-80GB) and sets
     `n_jobs` and `vram_budget_gb` automatically. Override either by
     editing the config cell.
  3. Syncs the diagnostic manifest and embeddings subdirs from Drive
     into /content.
  4. Runs `python 09_tune_component2.py --vram-cache --amp
     --n-jobs {N} --vram-budget-gb {B} --storage sqlite:///...`.
  5. Copies `results/{model}/component2_tune/` + the sqlite study file
     back to Drive.
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


md("""# Component 2 Hyperparameter Tuning (Colab)

Bayesian (Optuna TPE) hyperparameter search for the Component-2
PRAME-conditioned MIL classifier. 10-fold StratifiedGroupKFold CV
on a stratified random ~200-slide subsample of the diagnostic
manifest.

**Why a GPU notebook**: Component 2's model is small (~200K params),
so the bottleneck is per-slide h5 I/O and kernel-launch latency rather
than raw compute. This notebook does two things sequential CPU tuning
cannot:

1. **VRAM cache** preloads the full subsample to GPU memory once at
   study start; subsequent epochs / folds / trials read zero bytes
   from disk.
2. **Parallel trials** run N Optuna trials concurrently in threads
   sharing one CUDA context. Multiple threads' kernel launches
   interleave on the GPU command queue, hiding launch latency that
   would otherwise dominate per-step wall-clock.

GPU tier auto-detection sets `n_jobs` and `vram_budget_gb`:
| GPU         | VRAM   | n_jobs | vram_budget_gb |
|-------------|--------|--------|----------------|
| T4          | 16 GB  | 4      | 12             |
| L4          | 24 GB  | 8      | 18             |
| A100        | 40 GB  | 16     | 32             |
| A100 80GB   | 80 GB  | 24     | 64             |

Edit the config cell to override either value.

**Prerequisites**
- The diagnostic manifest at `prame-predict/data/expression/diagnostic_manifest.csv` on Drive.
- Embeddings on Drive under `prame-predict/embeddings/{model}*/...h5`.
- HuggingFace login is NOT needed (no model download; the foundation model features are already extracted).

**Runtime** — L4 with 8 parallel trials: ~12 to 25 min for 30 trials.
A100 with 16 parallel trials: ~6 to 15 min.
""")

code("""# Cell 1: Install dependencies, mount Drive, clone repo
!pip install -q optuna
# scikit-learn / h5py / pandas / matplotlib are pre-installed on Colab.

from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('prame-predict'):
    !git clone https://github.com/hb-1968/prame-predict.git
else:
    !cd prame-predict && git pull --ff-only
""")

code("""# Cell 2: Detect GPU tier; set n_jobs and vram_budget_gb.
import subprocess
import torch

def _gpu_info():
    if not torch.cuda.is_available():
        return None, 0.0
    name = torch.cuda.get_device_name(0)
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            text=True,
        ).strip().splitlines()[0]
        total_gb = float(out) / 1024.0
    except Exception:
        # fall back to torch's reported total
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return name, total_gb


def _tier_settings(name, total_gb):
    # Brutal-utilization defaults: maximize concurrent trials and push the
    # VRAM cache budget to ~90 percent of device memory. If a trial OOMs
    # because hyperparameter sampling lands on the high end of hidden_dim
    # and attn_dim simultaneously, Optuna catches the RuntimeError and
    # the study continues; halve N_JOBS in the override block below and
    # set RESUME=True in Cell 3 to recover.
    nm = (name or '').upper()
    if 'A100' in nm and total_gb >= 70:
        return 32, 74.0
    if 'A100' in nm:
        return 24, 36.0
    if 'L4' in nm:
        return 12, 22.0
    if 'T4' in nm:
        return 6, 14.0
    # Unknown GPU: scale by VRAM, still aggressive.
    return max(4, int(total_gb // 3)), max(10.0, total_gb * 0.85)


gpu_name, gpu_vram_gb = _gpu_info()
if gpu_name is None:
    raise RuntimeError(
        'No CUDA GPU detected. Switch the runtime to GPU '
        '(Runtime -> Change runtime type).'
    )
N_JOBS, VRAM_BUDGET_GB = _tier_settings(gpu_name, gpu_vram_gb)

# Manual overrides (lower N_JOBS if you see OOMs in Cell 5; raise it if you
# want to push utilization even harder).
# N_JOBS = 8
# VRAM_BUDGET_GB = 18.0

print(f'GPU:             {gpu_name}')
print(f'Total VRAM:      {gpu_vram_gb:.1f} GB')
print(f'n_jobs:          {N_JOBS}  (concurrent Optuna trials)')
print(f'vram_budget_gb:  {VRAM_BUDGET_GB}  (~{VRAM_BUDGET_GB / max(gpu_vram_gb, 1e-6) * 100:.0f}% of device)')
""")

code("""# Cell 3: Tuning configuration. Edit before running for shorter / longer sweeps.
MODEL = 'uni'              # 'uni' or 'conch'
TRIALS = 30                # Optuna trial budget
FOLDS = 10                 # inner CV folds
EPOCHS = 50                # max epochs per fold (early stopping per --patience)
MAX_SLIDES = 200           # stratified subsample target
MIN_PER_COHORT = 20        # floor per (source_group, label) stratum
N_STARTUP_TRIALS = 10      # TPE random warmup before Bayesian phase
PRUNER_WARMUP = 2          # folds completed before pruning kicks in
SEED = 42
RESUME = False             # set True to continue an existing study at STUDY_DB_DRIVE
                           # (the sqlite file in Cell 6's copy-back is reused)

# Drive paths (adjust if your project lives elsewhere on Drive).
DRIVE_ROOT = '/content/drive/MyDrive/prame-predict'
MANIFEST_DRIVE = f'{DRIVE_ROOT}/data/expression/diagnostic_manifest.csv'
EMB_DRIVE = f'{DRIVE_ROOT}/embeddings'
RESULTS_DRIVE = f'{DRIVE_ROOT}/results'

# Local working paths.
LOCAL_REPO = '/content/prame-predict'
LOCAL_MANIFEST = f'{LOCAL_REPO}/data/expression/diagnostic_manifest.csv'
LOCAL_EMB = f'{LOCAL_REPO}/embeddings'
LOCAL_RESULTS = f'{LOCAL_REPO}/results'

# Optuna sqlite path (in /content, copied back to Drive after).
STUDY_DB_LOCAL = '/content/component2_tune_study.db'
STUDY_DB_DRIVE = f'{RESULTS_DRIVE}/{MODEL}/component2_tune/study.db'
""")

code("""# Cell 4: Sync inputs from Drive, then sanity-check that every
# source_group in the manifest has an embedding subdir on Drive.
import os
import shutil
from pathlib import Path
import pandas as pd

Path(os.path.dirname(LOCAL_MANIFEST)).mkdir(parents=True, exist_ok=True)
shutil.copy2(MANIFEST_DRIVE, LOCAL_MANIFEST)
print(f'Copied manifest: {MANIFEST_DRIVE} -> {LOCAL_MANIFEST}')

# Mirror per-cohort subdirs. Same mapping as 10_train_component2.py's
# SOURCE_EMB_SUBDIR; keep these two lists in sync.
SOURCE_TO_SUBDIR = {
    'skcm_melanoma': f'{MODEL}',
    'skcm_normal':   f'{MODEL}',
    'gtex_normal':   f'{MODEL}_gtex',
    'cobra_bcc':     f'{MODEL}_cobra',
    'hest_visium':   f'{MODEL}_hest',
}

Path(LOCAL_EMB).mkdir(parents=True, exist_ok=True)
emb_subdirs = sorted(set(SOURCE_TO_SUBDIR.values()))
for sd in emb_subdirs:
    drive_sd = f'{EMB_DRIVE}/{sd}'
    local_sd = f'{LOCAL_EMB}/{sd}'
    if not os.path.isdir(drive_sd):
        print(f'  [skip] {drive_sd} (does not exist on Drive)')
        continue
    if os.path.isdir(local_sd):
        shutil.rmtree(local_sd)
    shutil.copytree(drive_sd, local_sd)
    n_h5 = len([f for f in os.listdir(local_sd) if f.endswith('.h5')])
    print(f'  Copied {sd}: {n_h5} .h5 files')

# Sanity check: every source_group present in the manifest must map to a
# Drive subdir that exists and is non-empty. Fail fast with a clear list of
# misses; otherwise we burn ~5 minutes booting Optuna before discovering
# that uni_gtex/ was never uploaded.
print()
print('Manifest source_group sanity check:')
manifest_df = pd.read_csv(LOCAL_MANIFEST)
present_sources = sorted(manifest_df['source_group'].dropna().unique().tolist())
missing = []
for src in present_sources:
    sd = SOURCE_TO_SUBDIR.get(src)
    if sd is None:
        print(f'  [warn] {src}: no known SOURCE_TO_SUBDIR mapping; '
              f'the script will fall back to {LOCAL_EMB}/{MODEL}/')
        continue
    local_sd = f'{LOCAL_EMB}/{sd}'
    if not os.path.isdir(local_sd):
        missing.append((src, sd, 'subdir absent'))
        continue
    n_h5 = len([f for f in os.listdir(local_sd) if f.endswith('.h5')])
    if n_h5 == 0:
        missing.append((src, sd, 'subdir empty'))
    else:
        n_rows = int((manifest_df['source_group'] == src).sum())
        print(f'  ok   {src:14s} -> {sd:14s} '
              f'({n_h5} .h5 files for {n_rows} manifest rows)')

if missing:
    print()
    print('=' * 60)
    print('FATAL: required embedding subdirs are missing on Drive:')
    for src, sd, why in missing:
        print(f'  {src:14s} expected at {EMB_DRIVE}/{sd}  ({why})')
    print()
    print('Upload the missing .h5 files to Drive and re-run Cell 4.')
    print('=' * 60)
    raise RuntimeError(f'{len(missing)} source_group(s) missing embeddings on Drive')

print('All source_groups have embeddings on Drive.')
""")

code("""# Cell 5: Run the Optuna TPE sweep with streaming output + periodic
# nvidia-smi snapshots so SM occupancy is visible during the run.
import os
import select
import subprocess
import sys
import time

cmd = [
    'python', '-u', f'{LOCAL_REPO}/09_tune_component2.py',
    '--model', MODEL,
    '--manifest', LOCAL_MANIFEST,
    '--emb-dir', LOCAL_EMB,
    '--results-dir', LOCAL_RESULTS,
    '--trials', str(TRIALS),
    '--folds', str(FOLDS),
    '--epochs', str(EPOCHS),
    '--max-slides', str(MAX_SLIDES),
    '--min-per-cohort', str(MIN_PER_COHORT),
    '--n-startup-trials', str(N_STARTUP_TRIALS),
    '--pruner-warmup', str(PRUNER_WARMUP),
    '--seed', str(SEED),
    '--device', 'cuda',
    '--n-jobs', str(N_JOBS),
    '--vram-cache',
    '--vram-budget-gb', str(VRAM_BUDGET_GB),
    '--amp',
    '--storage', f'sqlite:///{STUDY_DB_LOCAL}',
]
if RESUME:
    cmd.append('--resume')

print('Command:')
print('  ' + ' '.join(cmd))
print()


def _gpu_snapshot():
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,utilization.memory,memory.used',
             '--format=csv,noheader,nounits'],
            text=True, timeout=5,
        ).strip()
        # Single-GPU: one CSV line like '94, 78, 17312'
        first = out.splitlines()[0]
        sm, mem, used = (x.strip() for x in first.split(','))
        return f'[gpu] {sm}% sm / {mem}% mem-bw / {used} MiB used'
    except Exception as exc:  # noqa: BLE001
        return f'[gpu] snapshot failed: {exc}'


env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
t0 = time.time()
last_snap = 0.0
SNAPSHOT_EVERY = 30.0  # seconds

proc = subprocess.Popen(
    cmd, cwd=LOCAL_REPO,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=env,
)

try:
    while True:
        # Non-blocking drain of any available lines.
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:  # EOF
                break
            print(line, end='', flush=True)

        # Periodic GPU snapshot so the user can confirm SM occupancy.
        now = time.time()
        if now - last_snap > SNAPSHOT_EVERY:
            print(_gpu_snapshot(), flush=True)
            last_snap = now

        if proc.poll() is not None:
            # Drain any final tail of output before exiting.
            tail = proc.stdout.read()
            if tail:
                print(tail, end='', flush=True)
            break
finally:
    rc = proc.wait()

elapsed = time.time() - t0
print(f'\\nFinished in {elapsed / 60:.1f} min  (exit code {rc})')
""")

code("""# Cell 6: Inline dashboard + best-config preview.
import json
import os
from pathlib import Path

local_tune_dir = f'{LOCAL_RESULTS}/{MODEL}/component2_tune'
dashboard_path = f'{local_tune_dir}/tune_dashboard.png'
best_config_path = f'{local_tune_dir}/best_config.json'

if os.path.exists(dashboard_path):
    from IPython.display import Image, display
    display(Image(filename=dashboard_path))
else:
    print(f'[warn] {dashboard_path} not found.')
    print('       The sweep likely finished with zero completed trials; '
          'check Cell 5 output for errors.')

if os.path.exists(best_config_path):
    with open(best_config_path) as f:
        cfg = json.load(f)
    print()
    print('=' * 60)
    print(f'BEST TRIAL: val AUC = {cfg.get(\"best_value\", float(\"nan\")):.4f}')
    print('=' * 60)
    for k, v in cfg.get('hyperparameters', {}).items():
        print(f'  {k:18s} = {v}')
    print()
    print(f'n_complete / n_trials: {cfg.get(\"n_complete\")} / {cfg.get(\"n_trials\")}')
    print(f'n_pruned:              {cfg.get(\"n_pruned\")}')
    sub = cfg.get('subsample', {}) or {}
    print(f'subsample:             {sub.get(\"actual_slides\")} slides '
          f'from {sub.get(\"actual_patients\")} patients '
          f'(seed {sub.get(\"seed\")})')
    print()
    drive_cfg = f'{RESULTS_DRIVE}/{MODEL}/component2_tune/best_config.json'
    print('Next: after Cell 7 syncs results to Drive, run')
    print(f'  python 10_train_component2.py --compare --config {drive_cfg}')
else:
    print(f'[warn] {best_config_path} not found.')
""")

code("""# Cell 7: Copy results back to Drive.
import os
import shutil
from pathlib import Path

local_tune_dir = f'{LOCAL_RESULTS}/{MODEL}/component2_tune'
drive_tune_dir = f'{RESULTS_DRIVE}/{MODEL}/component2_tune'

Path(drive_tune_dir).mkdir(parents=True, exist_ok=True)
for fname in os.listdir(local_tune_dir):
    src = f'{local_tune_dir}/{fname}'
    dst = f'{drive_tune_dir}/{fname}'
    if os.path.isdir(src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f'  -> {dst}')

if os.path.exists(STUDY_DB_LOCAL):
    Path(os.path.dirname(STUDY_DB_DRIVE)).mkdir(parents=True, exist_ok=True)
    shutil.copy2(STUDY_DB_LOCAL, STUDY_DB_DRIVE)
    print(f'  -> {STUDY_DB_DRIVE}')

print('\\nDone. best_config.json is on Drive; 10_train_component2.py --config can pick it up.')
""")

md("""## After the sweep

Inspect `best_config.json` and `tune_dashboard.png` on Drive, then run
the production training:

```
python 10_train_component2.py --compare \\
    --config results/uni/component2_tune/best_config.json
```

If a trial gets pruned early or one fold returns NaN (degenerate val
set), that's expected behavior — the median pruner and per-fold AUC
masking handle both cases.

To resume an interrupted sweep, re-run Cell 5 with `--resume` added to
the command. The sqlite study file on Drive retains every completed
and pruned trial.
""")


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "tune_component2_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
