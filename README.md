# Multi-Layer Feature and Part-Token Fusion for Urban Object Re-Identification

Faizyab Ali Shah, Narmeen Sabah Siddiqui — National University of Sciences and Technology, Islamabad

Code for re-identifying urban objects (traffic signs, crosswalks, containers, rubbish bins) on
URVAM-ReID2026. The method builds on the Part-Aware Transformer (PAT) with a ViT-L/16 backbone:

- **Training:** stochastic augmentation, AdamW and 320×192 inputs on Urban2026 + UrbAM-ReID.
- **Inference:** multi-layer CLS-token fusion with part-token concatenation.
- **Post-processing:** class-aware re-ranking.

Every experiment we ran is in this one codebase and switched on by config flags. That includes the
negative results reported in the paper.

## Results

Cumulative ablation, mAP@100 on the Kaggle public leaderboard (paper, Table 1):

| Configuration | mAP | Config |
|---|---|---|
| PAT ViT-L baseline (SGD, 256×128) | .101 | `configs/paper/ablation/01_*` |
| + External data (UrbAM-ReID) | .120 | `configs/paper/ablation/02_*` |
| + Hard class filtering | .123 | `configs/paper/ablation/03_*` |
| + CLS fusion (K = 8) | .139 | `configs/paper/ablation/04_*` |
| + RPT + CJ (LGT off) | .148 | `configs/paper/ablation/05_*` |
| + Stochastic aug + AdamW + 320×192 | .160 | `configs/paper/train.yml` + `ablation/06_*` |
| **+ Part-token concatenation** | **.164** | `configs/paper/{train,test}.yml` |

Negative results, compared with the best system:

| Configuration | mAP | Config |
|---|---|---|
| Class-conditioned training | .086 | `configs/paper/negative/class_conditioned_train.yml` |
| Query expansion (Query Majority Voting) | .087 | `configs/paper/negative/query_expansion_qmv_test.yml` |
| Camera feature normalisation | .118 | `configs/paper/negative/camera_feature_norm_test.yml` |
| Model ensemble (SGD + AdamW) | .144 | — (no code in the repository) |
| Prototype filtering (GTSRB ViT) | .140 | `configs/paper/negative/prototype_filtering_test.yml` |
| Traffic-sign classifier | .154 | `configs/paper/negative/traffic_classifier_test.yml` |

The Table 1 training configs for rows 01–05 are reconstructed from the paper and the repository
history. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) gives the details of every experiment, including
the ones that are not in the paper.

## Method in one paragraph

PAT prepends three learnable part tokens to the ViT patch sequence; each attends to a masked
horizontal band. We train with identity cross-entropy (label smoothing 0.1), triplet loss and PAT's
patch-clustering loss. Each training image receives 0, 1 or 2 augmentations (p = 0.15 / 0.45 / 0.40)
from perspective, colour jitter, rotation and zoom, plus colour jitter and Random Patch Transfer.
Local grayscale and random erasing are off. At inference the embedding of an image and its
horizontal flip is

    f = L2norm([CLS_{L-7}; …; CLS_L; Part_1^L; Part_2^L; Part_3^L])   (8 + 3) × 1024 = 11264-d

Per-class k-reciprocal re-ranking, camera-aware (CAJ) scaling and a hard class filter are then
applied to the distances.

## Repository layout

```
train.py               training entry point
update.py              inference + post-processing -> ranking file and Kaggle submission CSV
test.py                plain evaluation on a labelled split
config/defaults.py     every config key, with a comment saying which experiment it belongs to
configs/
  paper/               the paper's method, Table 1 ablation rows and negative results
  experiments/<name>/  one folder per experiment: runnable train/test YAMLs + original/ branch configs
model/, processor/, loss/, solver/, data/, utils/   PAT model, training loop, losses, data pipeline
tools/                 helper scripts (python -m tools.<name>), tools/signs/ traffic-sign labelling
scripts/               shell / SLURM launchers and environment setup
results/submissions/   Kaggle submission CSVs produced during the project
Urban2026/, UAM_Unified/, Urban2026_UAM_MERGED/, MergedDataset/   datasets (images + CSVs)
```

## Setup

```bash
git clone https://github.com/faizyabShah/ReID.git
cd ReID
conda create -n pat python==3.10
conda activate pat
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
bash scripts/install_env.sh
```

Download the ImageNet-21k ViT-L/16 weights into `pretrained/`:

```bash
mkdir -p pretrained
wget -P pretrained https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth
```

All paths in `configs/` are relative to the repository root, so run every command from there.
`Urban2026_UAM_MERGED/` is built from Urban2026 and UrbAM-ReID with `python -m tools.merge`.

## Reproducing the paper

```bash
python train.py  --config_file configs/paper/train.yml                          # -> models/paper/
python update.py --config_file configs/paper/test.yml --track outputs/paper.txt # -> outputs/paper_submission.csv
```

`configs/paper/test.yml` loads `models/paper/part_attention_vit_60.pth`. To use another checkpoint,
change `TEST.WEIGHT` in the YAML or pass it on the command line, e.g.
`python update.py --config_file configs/paper/test.yml TEST.WEIGHT path/to.pth`.
`scripts/run.sh` runs both steps, and `scripts/train.slurm` is the cluster job we used.

## Running an experiment

Every experiment has a folder in `configs/experiments/` containing runnable `train.yml` and/or
`test.yml` files. These are the paper configuration plus that experiment's flags; the configs as they
were on the original branch are in `original/`. For example:

```bash
python train.py  --config_file configs/experiments/faiz_gradient_reversal/train.yml
python update.py --config_file configs/experiments/narmyn_camera_norm/test.yml --track outputs/camnorm.txt
```

Any config key can be overridden on the command line as `KEY VALUE` pairs after the other arguments.
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) lists every experiment with its flags, authors, original
branch and known issues.

## Notes

- **Training-time validation:** `DATASETS.TEST: ('UrbanElementsReID',)` uses the training images as
  query and gallery, so the mAP / Rank-1 printed during training measures fit, not generalisation.
  Methods were compared on the Kaggle leaderboard.
- **Determinism:** inference is not bit-identical from run to run, because part of the test pipeline
  uses unseeded randomness.
- **Development history:** each experiment was developed on its own branch and merged into `main`
  with its history intact (`git log --first-parent main` lists the merges).

## Acknowledgements

Built on [Part-Aware Transformer (PAT)](https://github.com/liyuke65535/Part-Aware-Transformer)
(Ni et al., ICCV 2023), which in turn builds on TransReID. Stochastic augmentation follows
Díaz Benito and Sequeiro González (ICIPW 2025).
