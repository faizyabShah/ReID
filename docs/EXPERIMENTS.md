# Experiments

Every experiment from the project lives on `main` and is switched on by config keys; with all keys at
their defaults (`config/defaults.py`) the code runs the plain PAT pipeline, and
`configs/paper/{train,test}.yml` run the paper's method.

For each experiment this page lists who built it, the branch it was developed on (each branch is
merged into `main` with its history), what it does, how to run it, the paper result if there is one,
and known issues. Known issues are recorded as found; the code was ported as it was on the branch
except where a change was needed to combine the experiments (marked **merge change**).

Paper numbers are mAP@100 on the Kaggle public leaderboard. Metrics printed during training are
computed on the training images (`DATASETS.TEST: ('UrbanElementsReID',)` uses `image_train/` as query
and gallery) and should not be used to compare methods.

## Overview

| Experiment | Authors | Stage | Paper mAP | Configs |
|---|---|---|---|---|
| [Paper method](#paper-method) | both | train + test | **.164** | `configs/paper/` |
| [Class-aware post-processing](#class-aware-post-processing) | Faiz | test | part of .164 | `configs/paper/test.yml` |
| [CLS fusion + part tokens](#cls-fusion-and-part-tokens) | Narmeen | test | .139 → .164 | `configs/paper/test.yml`, `narmyn_multilayer_part_tokens` |
| [Stochastic augmentation + AdamW](#stochastic-augmentation-and-adamw) | Narmeen | train | .148 → .160 | `configs/paper/train.yml` |
| [Merged dataset, gradient checkpointing](#merged-dataset-gradient-checkpointing-and-accumulation) | Narmeen | train | .101 → .120 | `narmyn_datamerge` |
| [Augmentation changes, re-ranking grid search](#augmentation-changes-two-stage-training-re-ranking-grid-search) | Faiz | train + tools | — | `faiz_augmentation_gridsearch` |
| [Class-conditioned attention](#class-conditioned-attention-queries) | Narmeen | train | .086 (negative) | `narmyn_class_conditioned_attention` |
| [Query Majority Voting](#query-majority-voting) | Narmeen | test | .087 (negative) | `narmyn_qmv` |
| [Camera feature normalisation](#camera-feature-normalisation-and-resample-tta) | Narmeen | test | .118 (negative) | `narmyn_camera_norm` |
| [Traffic-sign prototype filtering](#traffic-sign-prototype--sign-type-filtering) | Narmeen | test | .140 (negative) | `narmyn_sign_prototype` |
| [Traffic-sign classifier, domain-adversarial training, TTA](#traffic-sign-classifier-domain-adversarial-training-retrieval-tta) | Faiz | train + test | .154 (negative) | `faiz_gradient_reversal` |
| [Multi-scale TTA](#multi-scale-tta) | Faiz | test | — | `faiz_multiscale_tta` |
| [Traffic / non-traffic model split](#traffic--non-traffic-model-split) | Faiz | train + test | — | `faiz_traffic_split` |
| [Class-specific augmentation (policies)](#class-specific-augmentation-policies) | Faiz | train | — | `faiz_class_specific_aug` |
| [Class-wise augmentation](#class-wise-augmentation-and-sign-classifier-pipeline) | Narmeen | train | — | `narmyn_class_wise_aug` |
| [Class-wise PAT part masks](#class-wise-pat-part-masks) | Narmeen | train | — | `narmyn_classwise_part_masks` |
| Model ensemble (SGD + AdamW) | — | test | .144 (negative) | no code in the repository |

Config paths without a prefix are folders in `configs/experiments/`. Each holds runnable
`train.yml` / `test.yml` (the paper config plus the experiment's keys) and `original/`, the configs
exactly as they were on the branch (absolute paths from our machines).

## Paper method

The paper's system (Sec. 2): PAT with ViT-L/16, trained on Urban2026 + UrbAM-ReID with stochastic
augmentation, AdamW and 320×192 inputs; at inference CLS fusion (K = 8) + final-layer part tokens,
flip TTA, per-class k-reciprocal re-ranking, CAJ scaling and the hard class filter.

```bash
python train.py  --config_file configs/paper/train.yml
python update.py --config_file configs/paper/test.yml --track outputs/paper.txt
```

`configs/paper/ablation/` has one config per Table 1 row. Rows 01, 02 and 05 change the training
recipe and come with a `_train.yml`; the other rows are inference-only. These ablation configs are
reconstructed from the paper text and the trunk history — the repository does not record the exact
configuration of every leaderboard submission (e.g. whether class-based re-ranking / CAJ were on for
rows 01–05, which epoch was submitted). `configs/paper/negative/` has the negative results from
Table 1.

## Class-aware post-processing

*Faiz — trunk (`faizyab/post-hoc-things`, merged into every later branch)*

Hard class filter (+1e6 on cross-class pairs), per-class k-reciprocal re-ranking with its own
(k1, k2, λ) and simple camera-aware scaling (same camera ×1.1, cross camera ×0.95 on the re-ranked
distances). Per-class parameters were tuned with the staged grid search on UrbAM-ReID.

| Key | Meaning |
|---|---|
| `TEST.DO_CLASS_FILTER` | hard class filter (needs `query_classes.csv`, `test_classes.csv` in `DATASETS.ROOT_DIR`) |
| `TEST.DO_CLASS_BASED_RERANKING`, `TEST.CLASS_BASED_RERANKING_PARAMS.<class>.{K1,K2,LAMBDA}` | per-class re-ranking |
| `TEST.DO_CAJ_ADJUSTMENT`, `TEST.CAJ_SAME_CAM_PENALTY`, `TEST.CAJ_CROSS_CAM_SCALE` | CAJ scaling |
| `TEST.RE_RANKING` | unused by `update.py`; when class-based re-ranking is off, plain k-reciprocal re-ranking (k1 20, k2 6, λ 0.3) is always applied |

Tools: `python -m tools.evaluate_uam_gridsearch` (staged search over K, per-class k1/k2/λ and CAJ on
the labelled UAM_Unified split), `python -m tools.analyze_uam_submission` (per-class / per-query
statistics of a submission).

Also on the trunk, both inactive:
- `DATALOADER.CLASS_BALANCE` (class-balanced identity sampler) and class-aware triplet mining
  (`loss/triplet_loss.py`, uses `others['class']`). The train loader never sets `'class'` — on the
  trunk the class lookup used the pid instead of the image path, and the class-split merge replaced it
  with `'class_name'` — so both fall back to the standard sampler / triplet loss.
- `utils/caj_re_ranking.py` (Specker's CAJ re-ranking with asymmetric intra/inter-camera k1) was
  added and removed again on `narmyn/updatedposthocthings`; the file is kept but nothing calls it.

## CLS fusion and part tokens

*Narmeen — `narmyn/updatedposthocthings` (trunk commits `74eb24aa`, `325bd3be`, `be8dcb7d`, `aada2855`)*

Inference-only: the embedding concatenates the CLS tokens of the last K transformer layers and,
optionally, the three part tokens of the last P layers (Eq. 1). Zero retraining cost.

| Key | Paper value |
|---|---|
| `TEST.CLS_FUSION`, `TEST.CLS_FUSION_LAST` | True, 8 |
| `TEST.USE_PART_TOKENS`, `TEST.PART_TOKENS_LAST` | True, 1 |

`configs/experiments/narmyn_multilayer_part_tokens/test.yml` uses part tokens from the last 4
layers (the branch's last commit). The branch's shipped test YAML had `USE_PART_TOKENS` off; the paper
result uses it on.

## Stochastic augmentation and AdamW

*Narmeen — `narmyn/stochasticaugmentation`*

Each image receives 0, 1 or 2 operations from {colour jitter, perspective, rotation, resized crop}
(Díaz Benito and Sequeiro González, ICIPW 2025); RPT and colour jitter are applied on top, LGT and
REA are off; AdamW 5e-4, 320×192, batch 64, 60 epochs.

Keys: `INPUT.STOCHASTIC_AUG.{ENABLED,PROB_NONE,PROB_ONE,PROB_TWO}`; the rest is in
`configs/paper/train.yml`.

## Merged dataset, gradient checkpointing and accumulation

*Narmeen — `narmyn/datamerge`*

`python -m tools.merge` builds `Urban2026_UAM_MERGED/` from Urban2026 and UrbAM-ReID (Table 1:
+ External data, .101 → .120). The branch also adds gradient checkpointing and accumulation for
ViT-L at large batch sizes; `tools/debug.py` checks the merged dataset.

| Key | Branch value |
|---|---|
| `MODEL.GRAD_CHECKPOINTING` | True |
| `SOLVER.GRAD_ACCUM_STEPS` | 4 (batch 16) |

Known issues: gradient accumulation has no effect — the loop still calls `optimizer.zero_grad()`
every iteration, so only the last micro-batch's gradient is used.

## Augmentation changes, two-stage training, re-ranking grid search

*Faiz — `feat/change-in-augmentation`*

Lighter augmentation (no flip, REA p 0.25, CJ p 0.25, LGT off), 100 epochs on `MergedDataset/`,
one- and two-stage training (UAM_Unified first, then Urban2026), and a brute-force re-ranking search.
The runnable `train.yml` is the branch config with the pretrained path made relative.

- `scripts/train_urban_two_stage.sh`, `scripts/train_one_stage.sh` (Kaggle paths)
- `python -m tools.grid_search_rerank` — sweeps k1 × k2 × λ over saved `qf.npy` / `gf.npy`
- `tools.evaluate_csv.evaluate_from_indices` — importable evaluation

Known issues:
- The branch changed the `re_ranking()` defaults to k1 15, k2 3; `main` keeps 20 / 6 (the values
  the paper pipeline used).
- `train_urban_two_stage.sh` checks for `models/model/part_attention_vit_20.pth`, but stage 1 writes
  to `models/urban_two_stage/uam_unified/`, so stage 2 never starts. Both scripts and
  `grid_search_rerank.py` use `/kaggle/working` paths.
- `evaluate_from_indices` computes non-interpolated AP, so its mAP is not comparable with the
  `tools/evaluate_csv.py` command line.
- The branch also added `MergedDataset/`, a second copy of the merged images.

## Class-conditioned attention queries

*Narmeen — `AttQueryCond` · paper: class-conditioned training .086 (negative)*

A learned per-class embedding c is added to the part-attention queries during training,
Q = (X + 0.1 c) W_Q.

Key: `MODEL.CLASS_COND_QUERY` (merge change: on the branch it was always on; the flag also controls
whether the class-embedding parameter exists, so other checkpoints are unchanged).

Known issues:
- The branch's class map uses `'Trashbin'` and `'Trafficsign'`, which do not match the CSV labels
  (`RubbishBins`, `TrafficSign` / `trafficsignal`); rubbish bins and traffic signs therefore get class
  id 0 (Crosswalk). Kept as on the branch (`others['cond_class_id']`).
- Inference never passes class ids, so the model is trained with conditioning and evaluated without it.
- The 0.1 scale is hardcoded in `part_Attention`.

## Query Majority Voting

*Narmeen — `narmyn/vitlarge16` · paper: query expansion .087 (negative)*

Query images that are adjacent frames of the same object (≥ 50 ORB matches) are grouped with
union-find; every group shares one ranking built from rank-decayed (exp(−rank / 10)) scores of its
members' top-100 lists.

Keys: `TEST.QMV.{ENABLED,MATCH_THRESHOLD,TOP_K,DECAY,ORB_FEATURES}`. The branch shipped this as a
separate `update2.py` with plain re-ranking + class filter; `configs/experiments/narmyn_qmv/test.yml`
reproduces that pipeline, `configs/paper/negative/query_expansion_qmv_test.yml` applies QMV on top of
the paper pipeline. ORB matching is quadratic in the number of queries.

## Camera feature normalisation and resample TTA

*Narmeen — `narmyn/moreposthocthings` · paper: camera feature normalisation .118 (negative)*

Subtracts each camera's mean query / gallery feature and re-normalises. The branch also added a
"multi-resolution" TTA that up-samples ×1.1 and resizes back (effectively a slight blur).

Keys: `TEST.CAM_FEAT_NORM`, `TEST.RESAMPLE_TTA` (both hardcoded on the branch; merge change: flags,
off by default).

## Traffic-sign prototype / sign-type filtering

*Narmeen — `narmyn/trafficsignprototype` · paper: prototype filtering (GTSRB ViT) .140 (negative)*

Traffic signs are labelled by sign type / viewpoint (`tools/signs/signlabeler.py`: GTSRB ViT +
back-view detection and prototypes; `tools/signs/shape_sign_grouping.py`: shape classifier; guide in
`tools/signs/SIGNLABELER_GUIDE.md`). At inference, traffic-sign gallery items whose sign type differs
from the query's get a distance penalty (back views and unknown types are skipped).

Key: `TEST.SIGN_PROTOTYPE_FILTER` (merge change: on the branch the filter ran whenever the files
below existed). Inputs, looked up in the working directory or `DATASETS.ROOT_DIR`, or given with
`--prototype_bank / --query_sign_types / --gallery_sign_types`: `Urban2026/query_sign_types.csv`,
`Urban2026/test_sign_types.csv`, `Urban2026/sign_type_prototypes.pkl`. Training-set labels are in
`Urban2026/sign_labels/`. Requires `pandas`.

Known issues: `apply_prototype_filter` ignores the prototype embeddings and margin — it is a
shape-label mismatch penalty of 0.05 — but the prototype bank must exist; errors are caught and the
filter is skipped silently.

## Traffic-sign classifier, domain-adversarial training, retrieval TTA

*Faiz — `faizyab/traffic-classifier`, `faizyab/gradient-reversal` · paper: traffic classifier .154 (negative)*

- **Domain-adversarial training:** a camera classifier behind a gradient reversal layer
  (`model/domain_adaptation.py`) makes features camera-invariant.
  Keys `MODEL.DOMAIN_ADV.{ENABLED,LAMBDA,HIDDEN_DIM,DROPOUT}`.
- **Traffic-sign classifier:** a ResNet-18 over 15 visually grouped sign categories; traffic-sign
  pairs get +α(1 − p_q·p_g) (softmax) or a fixed penalty on argmax mismatch.
  Keys `TEST.TRAFFIC_SIGN_CLASSIFIER.{ENABLED,WEIGHT,MODE,ALPHA,ARGMAX_PENALTY,TEMPERATURE,INPUT_SIZE}`.
  The classifier weights are not in the repository.
- **Retrieval TTA:** every image at each of `TEST.RETRIEVAL_TTA_SIZES` × flips, each view
  L2-normalised and averaged. Key `TEST.USE_TTA`.
- Runtime positional-embedding resizing in the backbone, so the model accepts other input sizes
  (also what makes multi-scale TTA work).

When `TEST.USE_TTA` is on, or the classifier is enabled, `update.py` runs this branch's inference
pipeline (`run_tta_classifier_pipeline`), which differs from the paper pipeline: no part tokens,
CAJ before re-ranking, then the classifier filter. `original/train_one.yml` / `train_two.yml` are
the Kaggle two-stage configs.

Known issues:
- The classifier path is also selected when only `TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT` is set.
- CAJ is applied to the similarity matrices before re-ranking, so same-camera similarities are
  multiplied by 1.1 — the opposite of the intended penalty.
- With `TEST.USE_TTA` off this path still resizes to every `RETRIEVAL_TTA_SIZES` entry (without flips).
- The classifier receives ReID-normalised images (mean / std 0.5), not ImageNet normalisation.
- The dataset parsers subtract 1 from every camera id (for the domain classifier); camera ids are only
  compared for equality elsewhere, so other experiments are unaffected.

## Multi-scale TTA

*Faiz — `faizyab/post-hoc-multi-res` (Narmeen's multi-scale TTA on `narmyn/updatedposthocthings`
`4208f8a8` used sizes 320×192 / 352×208 / 288×176, i.e. the same idea)*

Every image is embedded at several scale factors (and flipped) and the embeddings are averaged or
max-pooled. Keys: `TEST.SCALES` (e.g. `[1.0, 0.9, 1.1]`), `TEST.SCALE_COMBINE` (`avg` | `max`);
`TEST.DO_MULTI_SCALE` is defined but unused. On the branch any scale ≠ 1.0 failed the patch-embedding
size check; on `main` it works through the runtime positional-embedding resizing.

## Traffic / non-traffic model split

*Faiz — `faizyab/traffic-class-things`*

Trains one model on traffic signs and one on the other classes; at inference traffic-sign queries
are ranked by the traffic model and the rest by the other model.

| Key | Meaning |
|---|---|
| `MODEL.CLASS_SPLIT_MODE` | `all` (default), `traffic`, `non_traffic`, `dual` (trains both; writes `models/<LOG_NAME>_traffic`, `models/<LOG_NAME>_non_traffic`) |
| `TEST.TRAFFIC_WEIGHT`, `TEST.OTHER_WEIGHT` | both set → dual-model inference (also `--traffic_weight / --other_weight`) |

Known issues:
- Dual-model inference uses plain k-reciprocal re-ranking only (no class re-ranking, CAJ or class
  filter).
- During split training the validation images get their class from the query / test CSVs by file
  name, which does not match the training images, and `num_query` is not filtered; validation in
  split mode is not meaningful and can fail when only some images get a class. The branch configs only
  validate at the last epoch.
- **Merge change:** dual-model inference now runs under `torch.no_grad()` (it raised on `.numpy()`
  otherwise); class annotation of test items only runs in split mode (it broke normal validation).

## Class-specific augmentation (policies)

*Faiz — `faizyab/class-specific-augmentations`*

Per-class augmentation policies (perspective, JPEG compression, motion blur, noise, small-patch
erasing, …) in `data/transforms/class_specific_aug.py`, applied through `ClassAwareTransformWrapper`.
Key: `INPUT.ENABLE_CLASS_SPECIFIC_AUG` (branch default True; merge change: False). Checks:
`python -m tools.validate_class_aug_static`, `python -m tools.validate_class_aug`.

Known issues:
- The policies are keyed by `TrafficSign`, `RubbishBins`, `Crosswalk`, `Container`, but the class name
  reaching the wrapper never matches (on the branch the lookup used the pid; on `main` it is the
  lowercase name from the class-split code), so only the base transforms are applied.
- If a policy did match: `T.RandomAffine(p=...)` is not a valid argument; crosswalks skip the final
  resize (variable batch shapes); the size is hardcoded to 256×128.
- Enabling it replaces the whole transform pipeline with one that has no stochastic augmentation.

## Class-wise augmentation and sign classifier pipeline

*Narmeen — `narmyn/class-wise-aug`, `narmyn/trafficsignalhardfiltering`*

Per-class tensor augmentation after the global transform, and a separate global transform for
traffic signs (no flip, no colour jitter, no REA). Key: `INPUT.CLASS_SPECIFIC_AUG.ENABLED`.
`tools/signs/sign_classifier_pipeline.py` pseudo-labels sign subtypes (HSV / shape heuristics),
fine-tunes a MobileNetV2 and predicts `query_sign_types.csv` / `test_sign_types.csv`.

Known issues:
- The class keys are the Urban2026 labels (`trafficsignal`); `Urban2026_UAM_MERGED` labels signs
  `TrafficSign`, so on the merged data traffic signs get neither the sign augmentation nor the
  global-transform override.
- Enabling it turns horizontal flip off for every class.
- **Merge change:** the plain dataset is built when the flag is off (the branch raised a NameError).

## Class-wise PAT part masks

*Narmeen — `narmyn/trainingvitlarge64`*

Crosswalks (stripes) use vertical left / centre / right part bands instead of PAT's horizontal ones
during training. Key: `MODEL.CLASSWISE_PART_MASK` (merge change: always on for labelled data on the
branch).

Known issues:
- Inference never passes class ids, so crosswalks are trained with vertical masks and evaluated with
  horizontal ones.
- **Merge change:** the train loader reuses the `add_info` dict the dataset now returns instead of
  appending a second one (the cause of the "too many values to unpack" crash in an earlier run).

## Config keys added by the experiments

| Key | Default | Experiment |
|---|---|---|
| `INPUT.STOCHASTIC_AUG.*` | off | stochastic augmentation |
| `INPUT.ENABLE_CLASS_SPECIFIC_AUG` | False | class-specific augmentation (Faiz) |
| `INPUT.CLASS_SPECIFIC_AUG.ENABLED` | False | class-wise augmentation (Narmeen) |
| `DATALOADER.CLASS_BALANCE` | False | class-balanced sampler (inactive) |
| `MODEL.CLASS_SPLIT_MODE` | `all` | traffic / non-traffic split |
| `MODEL.DOMAIN_ADV.*` | off | domain-adversarial training |
| `MODEL.GRAD_CHECKPOINTING`, `SOLVER.GRAD_ACCUM_STEPS` | False, 1 | merged-dataset training |
| `MODEL.CLASSWISE_PART_MASK` | False | class-wise part masks |
| `MODEL.CLASS_COND_QUERY`, `MODEL.NUM_OBJECT_CLASSES` | False, 4 | class-conditioned attention |
| `TEST.CLS_FUSION`, `TEST.CLS_FUSION_LAST` | False, 6 | CLS fusion |
| `TEST.USE_PART_TOKENS`, `TEST.PART_TOKENS_LAST` | False, 1 | part tokens |
| `TEST.DO_CLASS_FILTER`, `TEST.DO_CLASS_BASED_RERANKING`, `TEST.CLASS_BASED_RERANKING_PARAMS.*` | off | class-aware post-processing |
| `TEST.DO_CAJ_ADJUSTMENT`, `TEST.CAJ_*` | off | CAJ scaling |
| `TEST.SCALES`, `TEST.SCALE_COMBINE`, `TEST.DO_MULTI_SCALE` | None, avg, False | multi-scale TTA |
| `TEST.RESAMPLE_TTA`, `TEST.CAM_FEAT_NORM` | False | camera feature normalisation |
| `TEST.SIGN_PROTOTYPE_FILTER` | False | prototype filtering |
| `TEST.QMV.*` | off | Query Majority Voting |
| `TEST.USE_TTA`, `TEST.RETRIEVAL_TTA_SIZES` | False | Faiz's retrieval TTA |
| `TEST.TRAFFIC_SIGN_CLASSIFIER.*` | off | traffic-sign classifier |
| `TEST.TRAFFIC_WEIGHT`, `TEST.OTHER_WEIGHT` | empty | dual-model inference |
