#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="config/UrbanElementsReID_train.yml"

echo "[1/2] Training on UAM_Unified for 20 epochs"
python train.py --config_file "${CONFIG_PATH}" \
  SOLVER.MAX_EPOCHS 20 \
  DATASETS.ROOT_DIR "${REPO_ROOT}/UAM_Unified/" \
  LOG_NAME "urban_two_stage/uam_unified"

echo "[2/2] Training on Urban2026 for 20 epochs"
python train.py --config_file "${CONFIG_PATH}" \
  SOLVER.MAX_EPOCHS 20 \
  DATASETS.ROOT_DIR "${REPO_ROOT}/Urban2026/" \
  LOG_NAME "urban_two_stage/urban2026"

echo "Two-stage training completed."
