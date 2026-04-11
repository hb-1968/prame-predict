# PRAME Expression Prediction from H&E Morphology

Predicting PRAME gene expression levels in cutaneous melanoma from
H&E-stained whole-slide images using pathology foundation model
features and multiple instance learning. Part of a larger research
effort toward a PRAME-conditioned melanoma diagnostic system.

## Background

PRAME (Preferentially Expressed Antigen in Melanoma) is a diagnostic
immunohistochemical marker increasingly used to distinguish melanoma
from benign melanocytic lesions. Manual scoring of PRAME IHC suffers
from poor inter-rater reliability (Kappa = 0.16). This project
investigates whether PRAME expression can be predicted directly from
routine H&E-stained slides, bypassing the need for IHC staining.

## System Architecture

The project builds toward a three-component diagnostic system:

**Component 1 — PRAME Expression Predictor (current pipeline)**
Train a model to predict PRAME expression from H&E morphology alone,
using the full TCGA-SKCM cohort. RNA-Seq TPM values provide objective
ground truth across the full expression range. The model trains on all
available slides. Quartile-extreme cases (top/bottom 25%) are preserved
as a labeled subset for ground-truth evaluation — these represent cases
where PRAME status would be clinically actionable. The model outputs a
predicted PRAME level and a confidence score.

**Component 2 — PRAME-Conditioned Melanoma Classifier (future)**
Train a diagnostic model that takes H&E features AND PRAME expression
(real or predicted) as inputs to classify melanoma vs. not-melanoma.
PRAME becomes an additional learned parameter that the model weighs
alongside morphological features.

**Component 3 — Routing Logic (future)**
- If predicted PRAME is reliably extreme → route to PRAME-conditioned
  classifier (Component 2)
- If predicted PRAME is indeterminate → fall back to standard CONCH/UNI
  classification without PRAME conditioning

## Data

- **Source**: TCGA-SKCM (GDC Data Portal)
- **Full cohort**: 469 unique cases with PRAME expression data
- **Expression stats**: Min 0.00, Q25 224.61, Median 437.93, Q75 632.46,
  Max 2673.94 TPM
- **Training set**: 200 slides with embeddings — quartile extremes only (100 high, 100 low)
- **Patients**: 194 unique (4 patients have multiple slides — handled by patient-level CV splitting)
- **Ground-truth evaluation subset**: Quartile extremes (high ≥ Q75, low ≤ Q25)
  where PRAME is clinically reliable
- **Middle cases**: Included in training but separately tracked for
  indeterminate-tier analysis

### Why Train on the Full Range?

PRAME IHC scoring is unreliable at intermediate levels, but the RNA-Seq
TPM values used as ground truth are objective molecular measurements.
A TPM of 400 is not ambiguous — it is a real number from a sequencing
machine. The clinical reliability concern applies to IHC interpretation,
not to the expression data itself. Training on the full range maximizes
learning signal, while the routing logic in Component 3 handles the
clinical thresholds downstream.

## Models

- **UNI** (Chen et al., 2024) — pathology-specific vision foundation
  model (ViT-Large, DINOv2, 100M+ patches). Vision-only encoder.
- **CONCH** (Lu et al., 2024) — pathology-specific vision-language
  foundation model (ViT-Base, CoCa, 1.17M image-caption pairs).

Both models are compared as frozen feature extractors under the same
MIL aggregation pipeline.

### MIL Classifier

Gated Attention-based MIL (Ilse et al. 2018) for slide-level
classification. Each slide's patch embeddings are transformed through
a feature network, weighted by a learned gated attention mechanism
(element-wise product of tanh and sigmoid branches), aggregated into
a fixed-size slide representation, and classified as high or low PRAME.

- **Architecture**: Linear(feat_dim, 256) + ReLU + Dropout(0.25) →
  Gated Attention(256, 128) → Linear(256, 1)
- **Training**: Patient-level stratified 5-fold cross-validation
  (prevents data leakage from patients with multiple slides)
- **Optimization**: Adam (lr=1e-4, weight_decay=1e-4), cosine
  annealing LR, early stopping on validation AUC (patience=10)
- **Loss**: BCEWithLogitsLoss (binary: high vs. low PRAME)
- **Metrics**: AUC, accuracy, sensitivity, specificity (per-fold
  and pooled across folds)

## Pipeline
```bash
python 01_download_data.py              # Fetch expression data, generate slide manifest
# Download WSIs via gdc-client:
# gdc-client download -m data/expression/gdc_manifest.txt -d data/wsi -n 4

# Option A: Pipeline mode (recommended — tiles, extracts, and cleans up per slide)
# Peak ~11GB temp disk per slide (at default 80K patch cap), cleaned up automatically
python 03_extract_features.py --model uni --pipeline
python 03_extract_features.py --model conch --pipeline

# Option B: Separate steps (requires disk space for all tiles simultaneously)
python 02_tile_wsi.py --all             # Tile all WSIs into patches (.npy, max 80K per slide)
python 03_extract_features.py --model uni --all
python 03_extract_features.py --model conch --all

# Option A: Train locally (CPU)
python 04_train_mil.py --model uni           # Train attention MIL classifier (5-fold CV)
python 04_train_mil.py --model conch         # Train attention MIL classifier (5-fold CV)

# Option B: Train on Colab (GPU) — use notebooks/train_mil_colab.ipynb
# Reads embeddings from Drive, saves models + results back to Drive
python 05_generate_heatmaps.py               # Visualize attention on WSIs
python 06_compare_models.py                  # Side-by-side evaluation
```

## Compute Setup

Development on a Windows laptop (CPU only). Heavy compute (tiling,
feature extraction, MIL training) on Google Colab with a T4 GPU.

Two Colab notebooks handle the GPU-dependent stages:

- **`notebooks/prame_predict.ipynb`** — Feature extraction pipeline:
  downloads WSIs from the GDC API, tiles them in-memory, extracts
  embeddings with both UNI and CONCH, and saves to Google Drive.
  WSIs are deleted after processing to manage disk space.

- **`notebooks/train_mil_colab.ipynb`** — MIL training: trains the
  gated attention MIL classifier on pre-extracted embeddings from
  Drive. Runs patient-level stratified 5-fold CV with early stopping.
  Saves model weights, per-fold metrics (CSV/JSON), ROC curves, and
  training curves to Drive. Set `MODEL = "uni"` or `"conch"` to
  switch between feature sets.

### Colab Pipeline Optimizations

- **In-memory tiling**: Patches are built directly in RAM instead of
  writing to disk via memmap, eliminating the disk round-trip when
  patches are consumed immediately for feature extraction.
- **Thread-local OpenSlide handles**: Avoids re-opening the SVS file
  per patch. 8 parallel worker threads overlap JPEG decode I/O.
- **GPU-resident extraction**: When patches fit in VRAM (with 2GB
  headroom), all patches are pre-loaded onto the GPU as a single
  tensor. Batches are sliced directly from VRAM, eliminating
  CPU-to-GPU transfer during inference. Falls back to a CPU-side
  prefetcher for slides that exceed GPU memory.
- **torch.compile**: Model is compiled on first batch for fused
  operations and ~10-20% inference speedup on subsequent batches.
- **Float16 inference**: `torch.amp.autocast` for forward pass,
  embeddings saved as float16 in compressed HDF5.
- **Resumable**: Slides with existing `.h5` embeddings are skipped
  automatically, allowing safe interruption and restart.
- **Batch downloads**: WSIs are downloaded in batches of 75 with 16
  parallel threads and connection pooling. Cleaned up after each batch.

### Colab MIL Training

- Clones the repo to access the slide manifest (for labels and
  patient IDs), then loads embeddings directly from Drive
- Patient-level stratified splits via `StratifiedGroupKFold` —
  no patient appears in both train and val within a fold
- Cosine annealing LR schedule, early stopping on validation AUC
- Outputs per model saved to `Drive/prame-predict/results/{model}/`:
  `fold*_model.pt`, `cv_results.csv`, `summary.json`,
  `cv_results.png`, `training_curves.png`
- Run twice (once with `MODEL = "uni"`, once with `MODEL = "conch"`)
  to produce results for both feature extractors

### Slide Statistics

- **200 slides** downloaded from TCGA-SKCM
- **Median ~26K patches per slide** at 20x magnification, 224x224 pixels
- **Embeddings**: ~10MB per slide per model (compressed HDF5)
- **Tiling + extraction target**: ~5 minutes per median slide

## Environment
```bash
conda create -n prame python=3.11 -y
conda activate prame
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pandas pillow tqdm huggingface_hub timm
pip install openslide-python h5py requests
```

UNI and CONCH require Hugging Face access tokens and license agreements:
- UNI: https://huggingface.co/MahmoodLab/uni
- CONCH: https://huggingface.co/MahmoodLab/conch

## Prior Work

This project builds on a completed benchmarking study
([pathbench](https://github.com/hb-1968/pathbench)) comparing
foundation model representations on the MHIST dataset:

| Model | AUC (5-Fold CV) |
|-------|-----------------|
| ResNet-50 (ImageNet supervised) | 0.890 ± 0.008 |
| DINO ViT-B/16 (ImageNet self-supervised) | 0.878 ± 0.014 |
| UNI ViT-L/16 (Pathology self-supervised) | **0.917 ± 0.008** |

Key finding: pathology-specific training data, not architecture alone,
drives performance gains.

## Results
*(To be added)*
