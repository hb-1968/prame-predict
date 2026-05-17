"""One-shot builder for notebooks/train_component2_colab.ipynb.

Mirrors scripts/build_tune_component2_colab_notebook.py. Kept as a tracked
script so the notebook can be regenerated / diffed without hand-editing JSON.

The generated notebook:
  1. Mounts Drive, clones the repo.
  2. Detects GPU (single training run, so just confirms CUDA + enables AMP).
  3. Syncs the diagnostic manifest (from the cloned repo), the per-cohort
     embedding subdirs, and `best_config.json` (from a prior
     tune_component2 run) from Drive into /content.
  4. Runs `python 10_train_component2.py` with --compare and the tuned
     config by default, streaming stdout + periodic nvidia-smi snapshots.
  5. Inlines the bundled compare plot + per-mode summary tables.
  6. Copies `results/{model}/component2/` back to Drive (per-mode CSV/JSON,
     PNGs, fold checkpoints, compare/comparison.json, compare/compare_variants.png).
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


md("""# Component 2 Training (Colab)

PRAME-conditioned diagnostic MIL classifier. Runs `10_train_component2.py`
on Colab GPU with the tuned hyperparameters from a prior
`tune_component2_colab.ipynb` sweep.

**What this notebook does**
- Defaults to `--compare`, which runs the full N-fold CV for all three
  PRAME ablation modes (`full`, `no_predicted`, `no_prame`) using the
  same deterministic patient-level split.
- Loads `best_config.json` from Drive (produced by the tuning notebook)
  so hyperparameters are pinned to the tuned values.
- Streams training output and copies every artifact under
  `results/{model}/component2/` back to Drive when done.

**Prerequisites (everything else has been run already)**
- Diagnostic manifest at `data/expression/diagnostic_manifest.csv` in the
  cloned repo (tracked in git, populated by `08_build_diagnostic_manifest.py`).
- Per-cohort UNI embeddings on Drive under
  `prame-predict/embeddings/uni*/...h5` (built by the SKCM, GTEx, HEST,
  and COBRA extraction notebooks).
- `best_config.json` on Drive under
  `prame-predict/results/{model}/component2_tune/best_config.json`
  (built by `tune_component2_colab.ipynb`). Optional: set
  `USE_TUNED_CONFIG = False` in Cell 3 to skip and use script defaults.

**Runtime** - T4: ~10 to 25 min for `--compare --folds 5`. L4: ~5 to 12 min.
A100: ~3 to 7 min.
""")

code("""# Cell 1: Mount Drive, clone repo.
# scikit-learn / h5py / pandas / matplotlib / torch are pre-installed on Colab.

from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('prame-predict'):
    !git clone https://github.com/hb-1968/prame-predict.git
else:
    !cd prame-predict && git pull --ff-only
""")

code("""# Cell 2: Confirm CUDA. Single training run, so no multi-trial / VRAM cache
# knobs to tune; the only GPU optimization here is bf16 autocast via --amp.
import subprocess
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        'No CUDA GPU detected. Switch the runtime to GPU '
        '(Runtime -> Change runtime type). CPU runs of the production '
        '--compare CV are slow enough to be impractical.'
    )

gpu_name = torch.cuda.get_device_name(0)
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        text=True,
    ).strip().splitlines()[0]
    gpu_vram_gb = float(out) / 1024.0
except Exception:
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f'GPU:        {gpu_name}')
print(f'Total VRAM: {gpu_vram_gb:.1f} GB')
print(f'AMP:        bf16 autocast enabled in Cell 5 via --amp')
""")

code("""# Cell 3: Training configuration. Edit before running for ablations / shorter sweeps.
MODEL = 'uni'              # 'uni' or 'conch' (uni recommended; component 1 showed conch near chance)
MODE = 'compare'           # 'compare' | 'full' | 'no_predicted' | 'no_prame'
FOLDS = 5                  # CV folds (5 is the production default)
EPOCHS = 50                # max epochs per fold (early stopping kicks in via tuned --patience)
SEED = 42
USE_TUNED_CONFIG = True    # load best_config.json from Drive; set False to use script defaults
USE_AMP = True             # bf16 autocast on CUDA forward pass

# Drive paths (adjust if your project lives elsewhere on Drive).
DRIVE_ROOT = '/content/drive/MyDrive/prame-predict'
EMB_DRIVE = f'{DRIVE_ROOT}/embeddings'
RESULTS_DRIVE = f'{DRIVE_ROOT}/results'
TUNED_CONFIG_DRIVE = f'{RESULTS_DRIVE}/{MODEL}/component2_tune/best_config.json'

# Local working paths.
LOCAL_REPO = '/content/prame-predict'
LOCAL_MANIFEST = f'{LOCAL_REPO}/data/expression/diagnostic_manifest.csv'
LOCAL_EMB = f'{LOCAL_REPO}/embeddings'
LOCAL_RESULTS = f'{LOCAL_REPO}/results'
LOCAL_TUNED_CONFIG = f'{LOCAL_RESULTS}/{MODEL}/component2_tune/best_config.json'
""")

code("""# Cell 4: Sync embeddings + tuned config from Drive, then sanity-check that every
# source_group in the manifest has an embedding subdir on Drive.
# Manifest is read directly from the cloned repo (tracked in git).
import os
import shutil
from pathlib import Path
import pandas as pd

assert os.path.exists(LOCAL_MANIFEST), (
    f'Manifest not found at {LOCAL_MANIFEST}. Run '
    '08_build_diagnostic_manifest.py on your laptop, commit the CSV, '
    'and `git push`; then re-run Cell 1 to `git pull --ff-only` here.'
)
print(f'Manifest: {LOCAL_MANIFEST}  (from cloned repo)')

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

# Sync the tuned config from Drive (produced by tune_component2_colab.ipynb).
if USE_TUNED_CONFIG:
    if not os.path.exists(TUNED_CONFIG_DRIVE):
        raise FileNotFoundError(
            f'USE_TUNED_CONFIG=True but {TUNED_CONFIG_DRIVE} not found on Drive. '
            f'Run notebooks/tune_component2_colab.ipynb first, or set '
            f'USE_TUNED_CONFIG=False in Cell 3 to use script defaults.'
        )
    Path(os.path.dirname(LOCAL_TUNED_CONFIG)).mkdir(parents=True, exist_ok=True)
    shutil.copy2(TUNED_CONFIG_DRIVE, LOCAL_TUNED_CONFIG)
    print(f'\\nTuned config: {LOCAL_TUNED_CONFIG}  (copied from Drive)')
    import json as _json
    with open(LOCAL_TUNED_CONFIG) as _f:
        _cfg = _json.load(_f)
    print(f'  best val AUC during tuning: {_cfg.get(\"best_value\", float(\"nan\")):.4f}')
    for k, v in (_cfg.get('hyperparameters', {}) or {}).items():
        print(f'    {k:18s} = {v}')

# Sanity check: every source_group present in the manifest must map to a
# Drive subdir that exists and is non-empty.
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
    REMEDIATION = {
        'skcm_melanoma':     'notebooks/prame_predict.ipynb (Component-1 SKCM extraction)',
        'skcm_normal':       'notebooks/prame_predict.ipynb (Component-1 SKCM extraction)',
        'gtex_normal':       'notebooks/gtex_extract_colab.ipynb',
        'hest_visium':       'notebooks/hest_extract_colab.ipynb',
        'cobra_bcc':         'notebooks/cobra_predict_colab.ipynb',
    }
    print()
    print('=' * 64)
    print('FATAL: required embedding subdirs are missing on Drive:')
    for src, sd, why in missing:
        print(f'  {src:14s} expected at {EMB_DRIVE}/{sd}  ({why})')
        remedy = REMEDIATION.get(src, 'no extraction notebook registered for this cohort')
        print(f'                  -> run {remedy}')
    print('=' * 64)
    raise RuntimeError(f'{len(missing)} source_group(s) missing embeddings on Drive')

print('\\nAll source_groups have embeddings on Drive.')
""")

code("""# Cell 5: Run 10_train_component2.py with streaming output + periodic
# nvidia-smi snapshots.
import os
import select
import subprocess
import time

cmd = [
    'python', '-u', f'{LOCAL_REPO}/10_train_component2.py',
    '--model', MODEL,
    '--manifest', LOCAL_MANIFEST,
    '--emb-dir', LOCAL_EMB,
    '--results-dir', LOCAL_RESULTS,
    '--folds', str(FOLDS),
    '--epochs', str(EPOCHS),
    '--seed', str(SEED),
    '--device', 'cuda',
]
if MODE == 'compare':
    cmd.append('--compare')
else:
    cmd += ['--mode', MODE]
if USE_TUNED_CONFIG:
    cmd += ['--config', LOCAL_TUNED_CONFIG]
if USE_AMP:
    cmd.append('--amp')

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
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end='', flush=True)

        now = time.time()
        if now - last_snap > SNAPSHOT_EVERY:
            print(_gpu_snapshot(), flush=True)
            last_snap = now

        if proc.poll() is not None:
            tail = proc.stdout.read()
            if tail:
                print(tail, end='', flush=True)
            break
finally:
    rc = proc.wait()

elapsed = time.time() - t0
print(f'\\nFinished in {elapsed / 60:.1f} min  (exit code {rc})')
if rc != 0:
    raise RuntimeError(f'10_train_component2.py exited with code {rc}')
""")

code("""# Cell 6: Inline preview of result artifacts.
import json
import os
from pathlib import Path
from IPython.display import Image, display

local_c2_dir = Path(f'{LOCAL_RESULTS}/{MODEL}/component2')

if MODE == 'compare':
    # Bundled comparison plot
    compare_png = local_c2_dir / 'compare' / 'compare_variants.png'
    if compare_png.exists():
        display(Image(filename=str(compare_png)))
    else:
        print(f'[warn] {compare_png} not found')

    # Per-mode CV plots
    for sub_mode in ('full', 'no_predicted', 'no_prame'):
        png = local_c2_dir / f'cv_results_{sub_mode}.png'
        if png.exists():
            print(f'\\n=== {sub_mode} ===')
            display(Image(filename=str(png)))

    # Comparison JSON headline
    comp_json = local_c2_dir / 'compare' / 'comparison.json'
    if comp_json.exists():
        with open(comp_json) as f:
            comp = json.load(f)
        print()
        print('=' * 64)
        print(f'COMPARISON SUMMARY ({FOLDS}-fold CV per mode)')
        print('=' * 64)
        print(f'  {\"mode\":14s} {\"val_auc\":>17s} {\"sens\":>17s} '
              f'{\"spec\":>17s} {\"pooled\":>8s}')
        for sub_mode in ('full', 'no_predicted', 'no_prame'):
            s = comp.get(sub_mode, {})
            if not s:
                continue
            def _fmt(m, sd):
                return f'{m:.3f} +/- {sd:.3f}'
            print(f'  {sub_mode:14s} '
                  f'{_fmt(s[\"mean_val_auc\"], s[\"std_val_auc\"]):>17s} '
                  f'{_fmt(s[\"mean_sensitivity\"], s[\"std_sensitivity\"]):>17s} '
                  f'{_fmt(s[\"mean_specificity\"], s[\"std_specificity\"]):>17s} '
                  f'{s[\"pooled_auc\"]:>8.3f}')
else:
    # Single-mode run: show that mode's plots + summary
    png = local_c2_dir / f'cv_results_{MODE}.png'
    if png.exists():
        display(Image(filename=str(png)))
    curves = local_c2_dir / f'training_curves_{MODE}.png'
    if curves.exists():
        display(Image(filename=str(curves)))
    summary_json = local_c2_dir / f'summary_{MODE}.json'
    if summary_json.exists():
        with open(summary_json) as f:
            s = json.load(f)
        print()
        print('=' * 64)
        print(f'SUMMARY - mode={MODE}')
        print('=' * 64)
        for k in ('mean_auc', 'std_auc', 'mean_acc', 'std_acc',
                  'mean_sensitivity', 'std_sensitivity',
                  'mean_specificity', 'std_specificity', 'pooled_auc'):
            if k in s:
                print(f'  {k:18s} = {s[k]:.4f}')
""")

code("""# Cell 7: Copy results back to Drive.
import os
import shutil
from pathlib import Path

local_c2_dir = f'{LOCAL_RESULTS}/{MODEL}/component2'
drive_c2_dir = f'{RESULTS_DRIVE}/{MODEL}/component2'

Path(drive_c2_dir).mkdir(parents=True, exist_ok=True)
for entry in os.listdir(local_c2_dir):
    src = f'{local_c2_dir}/{entry}'
    dst = f'{drive_c2_dir}/{entry}'
    if os.path.isdir(src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f'  -> {dst}')

print('\\nDone. Per-mode CSV/JSON, plots, fold checkpoints, and the')
print('comparison bundle are now on Drive under')
print(f'  {drive_c2_dir}/')
""")

md("""## After training

Inspect `compare/compare_variants.png` and `compare/comparison.json` on
Drive. The headline number is the pooled AUC per mode:

- `full`: model has access to PRAME (measured for SKCM/GTEx, pseudobulked
  for HEST, predicted by Component 1 for COBRA).
- `no_predicted`: same as `full` but COBRA's predicted PRAME is silenced
  (treated as `has_prame=False`); isolates the contribution of measured-only
  PRAME.
- `no_prame`: pure visual MIL baseline (no PRAME branch at all);
  architecturally identical to Component 1's `AttentionMIL`.

The `full vs. no_prame` gap is the headline Component 2 contribution.
The `full vs. no_predicted` gap quantifies how much Component 1's
predicted PRAME on the COBRA cohort is actually buying you on top of
measured PRAME from SKCM/GTEx/HEST.

To re-run a single mode without re-doing the whole compare sweep, set
`MODE = 'full'` (or `'no_predicted'` / `'no_prame'`) in Cell 3 and
re-run Cells 5 to 7. The deterministic patient-level split (seed=42)
guarantees the per-fold metrics will match the corresponding mode from
the compare run exactly.
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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "train_component2_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
