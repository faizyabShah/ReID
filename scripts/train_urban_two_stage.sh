#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="configs/experiments/faiz_augmentation_gridsearch/train.yml"

echo "[1/2] Training on UAM_Unified for 20 epochs"
python train.py --config_file "${CONFIG_PATH}" \
  SOLVER.MAX_EPOCHS 20 \
  DATASETS.ROOT_DIR "${REPO_ROOT}/UAM_Unified/" \
  LOG_NAME "urban_two_stage/uam_unified"

# Path to checkpoint from stage 1
CKPT_PATH="${REPO_ROOT}/models/model/part_attention_vit_20.pth"

echo "Checkpoint expected at: $CKPT_PATH"

# Optional: fail early if missing
if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: Checkpoint not found!"
  exit 1
fi

echo "[2/2] Training on Urban2026 using pretrained checkpoint"
python train.py --config_file "${CONFIG_PATH}" \
  MODEL.PRETRAIN_CHOICE 'self' \
  MODEL.PRETRAIN_PATH "$CKPT_PATH" \
  SOLVER.MAX_EPOCHS 20 \
  DATASETS.ROOT_DIR "${REPO_ROOT}/Urban2026/" \
  LOG_NAME "urban_two_stage/urban2026"

echo "Two-stage training completed."