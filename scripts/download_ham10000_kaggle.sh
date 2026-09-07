#!/usr/bin/env bash
# Download HAM10000 images/metadata and ground-truth segmentation masks via the Kaggle API.
#
# Requires the `kaggle` package and a configured API token (~/.kaggle/kaggle.json), which you
# must set up yourself (Kaggle account -> Settings -> Create New Token). This script never
# handles the token directly.
set -euo pipefail

OUT_DIR="${1:-data/raw}"
mkdir -p "$OUT_DIR" "$OUT_DIR/segmentations"

pip install -q kaggle

echo "Downloading kmader/skin-cancer-mnist-ham10000 (images + metadata)..."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$OUT_DIR" --unzip

echo "Downloading tschandl/ham10000-lesion-segmentations (ground-truth masks)..."
kaggle datasets download -d tschandl/ham10000-lesion-segmentations -p "$OUT_DIR/segmentations" --unzip

echo "Reorganizing into the canonical data/raw/ layout..."
python scripts/prepare_kaggle_data.py --raw-dir "$OUT_DIR"

echo "Done. See $OUT_DIR for images/, HAM10000_metadata.csv, and segmentations/."
