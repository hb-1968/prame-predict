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
- **Training set**: All matched slides (~227 slides) — full expression range
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

## Pipeline
```bash
python 01_download_data.py              # Fetch expression data, generate slide manifest
# Download WSIs via gdc-client:
# gdc-client download -m data/expression/gdc_manifest.txt -d data/wsi -n 4
python 02_tile_wsi.py                   # Tile WSIs into patches
python 03_extract_features.py --model uni    # Extract UNI embeddings
python 03_extract_features.py --model conch  # Extract CONCH embeddings
python 04_train_mil.py --model uni           # Train attention MIL classifier
python 04_train_mil.py --model conch         # Train attention MIL classifier
python 05_generate_heatmaps.py               # Visualize attention on WSIs
python 06_compare_models.py                  # Side-by-side evaluation
```

## Compute Setup

Development on a Windows laptop (CPU only). Heavy compute (tiling,
feature extraction) on Google Colab with a T4 GPU. Per-slide processing
loop:

1. Upload one .svs from laptop to Google Drive
2. Colab tiles, extracts features, saves embeddings to Drive
3. Delete WSI and tiles from Drive to free space
4. Repeat

WSI download recommended on a fast connection (campus network).
At home speeds (~1 MB/s), the full dataset takes impractically long.
University networks typically achieve 100+ MB/s to GDC servers.

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
