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

## Method

1. Tile WSIs into 224×224 patches at 20× magnification
2. Extract patch-level features using UNI (pathology foundation model)
3. Aggregate patch features via attention-based MIL for slide-level prediction
4. Generate attention heatmaps to visualize morphological correlates of PRAME expression

## Pipeline
```bash
python 01_download_data.py    # Fetch WSIs and expression data from GDC
python 02_tile_wsi.py         # Tile WSIs into patches
python 03_extract_features.py # Extract UNI embeddings per patch
python 04_train_mil.py        # Train attention MIL classifier
python 05_generate_heatmaps.py # Visualize attention on WSIs
```

## Setup
```bash
conda create -n prame python=3.11 -y
conda activate prame
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pandas pillow tqdm huggingface_hub timm
pip install openslide-python h5py
```

## Results
*(To be added)*