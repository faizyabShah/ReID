#!/bin/bash
# Paper method: train, then write the ranking + Kaggle submission.
# Run from the repository root. Paths are set in configs/paper/{train,test}.yml.
set -e
python train.py --config_file configs/paper/train.yml
python update.py --config_file configs/paper/test.yml --track outputs/paper.txt
