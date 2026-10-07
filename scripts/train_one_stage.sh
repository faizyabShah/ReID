#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="configs/experiments/faiz_augmentation_gridsearch/train.yml"

# ✅ Now using WORKING directory (writeable)
CKPT_PATH="/kaggle/working/ReID_dataset"

echo "Using checkpoint: $CKPT_PATH"


echo "[Stage 2] Training on Urban2026 using pretrained checkpoint"

python train.py --config_file "${CONFIG_PATH}" \
  MODEL.PRETRAIN_CHOICE 'self' \
  MODEL.PRETRAIN_PATH "$CKPT_PATH" \
  SOLVER.MAX_EPOCHS 20 \
  DATASETS.ROOT_DIR "${REPO_ROOT}/Urban2026/" \
  LOG_NAME "urban_two_stage/urban2026"

echo "Training completed."