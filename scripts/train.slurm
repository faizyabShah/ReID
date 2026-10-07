#!/bin/bash
#SBATCH --job-name=adam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH --nodelist=renyi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.e
# If renyi is busy, change to: --gres=gpu:v100:2 (erdos/lanczos)
# or --gres=gpu:v100:1 (neumann, single GPU — script auto-detects)

source ~/.bashrc

conda activate pat
cd ~/ReID_Project/ReID   # repository root
python train.py --config_file configs/paper/train.yml