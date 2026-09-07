# Uncertainty-Aware Explainable Hybrid CNN-Transformer for Skin Lesion Classification

Explainable Vision Transformer Framework for Multi-Class Skin Lesion Classification Using Dermoscopic Images, with quantitative XAI faithfulness validation and predictive-uncertainty-based Trust Scoring.

## Overview

This project classifies dermoscopic skin lesion images (HAM10000, 7 classes) using a
lightweight hybrid CNN-Transformer, then goes beyond typical qualitative heatmap-only
explainability by:

1. Generating explanations from **three independent XAI methods** (Grad-CAM++, Attention
   Rollout, SHAP) and measuring their **cross-method agreement**.
2. **Quantitatively validating explanation faithfulness** against ground-truth ISIC lesion
   segmentation masks (IoU, Pointing Game, Deletion/Insertion AUC) instead of relying on
   visual inspection alone.
3. Estimating **predictive uncertainty** via MC-Dropout and fusing it with faithfulness and
   agreement into a single per-prediction **Trust Score**.

See [docs/methodology.md](docs/methodology.md) for the full research design (research gap,
objectives, novelty, mathematical formulation, evaluation protocol, baselines, ablations).

## Status

Project scaffold — architecture, preprocessing, XAI, uncertainty, and evaluation code are
implemented. Training/evaluation has **not yet been run**; no results are reported here until
genuine experiments are executed (per project policy: no fabricated datasets, references, or
results).

## Dataset

**HAM10000** (Tschandl et al., 2018) — 10,015 dermoscopic images, 7 diagnostic classes.

- Official source: [Harvard Dataverse, DOI 10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Also mirrored on the [ISIC Archive](https://www.isic-archive.com/) and Kaggle.
- Distributed for **non-commercial research use only** — download it yourself and accept the
  terms of use; this repository does not redistribute the images. Ground-truth lesion
  segmentation masks (needed for the faithfulness metrics) are available from the ISIC
  Archive under the same lesion IDs.

Place the downloaded data as:

```
data/
  raw/
    images/            # HAM10000_images_part_1/, HAM10000_images_part_2/
    HAM10000_metadata.csv
    segmentations/      # ISIC ground-truth lesion masks, matched by image_id
```

## Repository Structure

```
src/
  preprocessing/   # hair removal, color normalization, lesion-wise leakage-safe split
  models/          # hybrid CNN-Transformer + baseline models (ResNet50, EfficientNet-B0, ViT-Base)
  xai/             # Grad-CAM++, Attention Rollout, SHAP, cross-method agreement
  uncertainty/     # MC-Dropout predictive uncertainty
  evaluation/      # classification metrics, XAI faithfulness metrics, Trust Score
  train.py         # training entry point
  evaluate.py      # evaluation / XAI / uncertainty / trust-score entry point
configs/config.yaml
docs/methodology.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# 1. Prepare a lesion-wise (patient-wise) leakage-safe split
python -m src.preprocessing.dataset_split --metadata data/raw/HAM10000_metadata.csv --out data/splits

# 2. Train the proposed model (or a baseline via --model)
python -m src.train --config configs/config.yaml --model hybrid_cnn_transformer

# 3. Evaluate: classification metrics + XAI faithfulness + uncertainty + Trust Score
python -m src.evaluate --config configs/config.yaml --checkpoint runs/hybrid_cnn_transformer/best.pt
```

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset use is governed separately by the HAM10000 /
ISIC Archive terms of use (non-commercial).
