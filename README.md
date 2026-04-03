# PRAME Expression Prediction from H&E Morphology

Predicting PRAME gene expression levels in cutaneous melanoma from
H&E-stained whole-slide images using pathology foundation model
features and multiple instance learning.

## Background

PRAME (Preferentially Expressed Antigen in Melanoma) is a diagnostic
immunohistochemical marker increasingly used to distinguish melanoma
from benign melanocytic lesions. Manual scoring of PRAME IHC suffers
from poor inter-rater reliability (Kappa = 0.16). This project
investigates whether PRAME expression can be predicted directly from
routine H&E-stained slides, bypassing the need for IHC staining.

## Data

- **Images**: H&E whole-slide images from TCGA-SKCM (GDC Data Portal)
- **Labels**: PRAME mRNA expression from TCGA RNA-Seq, binarized at
  median into high/low expression groups

## Models

- **UNI** (Chen et al., 2024) — pathology-specific vision foundation
  model (ViT-Large, DINOv2, 100M+ patches). Vision-only encoder.
- **CONCH** (Lu et al., 2024) — pathology-specific vision-language
  foundation model (ViT-Base, CoCa, 1.17M image-caption pairs).
  Language-aligned representations may capture clinically meaningful
  morphological patterns differently than vision-only pretraining.

Both models are compared as frozen feature extractors under the same
MIL aggregation pipeline, isolating the effect of vision-only vs.
vision-language pretraining on PRAME expression prediction.

## Method

1. Tile WSIs into 224×224 patches at 20× magnification
2. Extract patch-level features using UNI and CONCH
3. Aggregate patch features via attention-based MIL for slide-level prediction
4. Generate attention heatmaps to visualize morphological correlates of PRAME expression
5. Compare UNI vs. CONCH representations for PRAME prediction

## Pipeline
```bash
python 01_download_data.py              # Fetch WSIs and expression data from GDC
python 02_tile_wsi.py                   # Tile WSIs into patches
python 03_extract_features.py --model uni    # Extract UNI embeddings per patch
python 03_extract_features.py --model conch  # Extract CONCH embeddings per patch
python 04_train_mil.py --model uni           # Train attention MIL classifier
python 04_train_mil.py --model conch         # Train attention MIL classifier
python 05_generate_heatmaps.py               # Visualize attention on WSIs
python 06_compare_models.py                  # Side-by-side evaluation
```

## Setup
```bash
conda create -n prame python=3.11 -y
conda activate prame
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pandas pillow tqdm huggingface_hub timm
pip install openslide-python h5py
```

UNI and CONCH require Hugging Face access tokens and license agreements:
- UNI: https://huggingface.co/MahmoodLab/uni
- CONCH: https://huggingface.co/MahmoodLab/conch

## Results
*(To be added)*