from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'vit'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ""

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

# ID/triplet
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

_C.MODEL.DOMAIN_ADV = CN()
_C.MODEL.DOMAIN_ADV.ENABLED = False
_C.MODEL.DOMAIN_ADV.LAMBDA = 1.0
_C.MODEL.DOMAIN_ADV.HIDDEN_DIM = 256
_C.MODEL.DOMAIN_ADV.DROPOUT = 0.5
_C.MODEL.DOMAIN_ADV.NUM_DOMAINS = 0

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# patch_embed_type: resnet/ibn/None
_C.MODEL.PATCH_EMBED_TYPE = ''
# fixed patch embed or not
_C.MODEL.FREEZE_PATCH_EMBED = True

# part views
_C.MODEL.PC_SCALE = 0.02
_C.MODEL.PC_LOSS = True
_C.MODEL.PC_LR = 1.0

# soft label
_C.MODEL.SOFT_LABEL = True
_C.MODEL.CLUSTER_K = 10 # num of clusters
_C.MODEL.SOFT_WEIGHT = 0.5
_C.MODEL.SOFT_LAMBDA = 0.5
_C.MODEL.GRAD_CHECKPOINTING = False

# Split training into a traffic-only model and a non-traffic model.
_C.MODEL.CLASS_SPLIT_MODE = 'all'  # options: 'all', 'traffic', 'non_traffic', 'dual'
_C.MODEL.TRAFFIC_CLASS_NAME = 'traffic'

#-----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [123.675, 116.28, 103.53]
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
# Augmix
_C.INPUT.DO_AUGMIX = False
# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 1.0
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1
# Local Grayscale Transfomation
_C.INPUT.LGT = CN()
_C.INPUT.LGT.DO_LGT = False
_C.INPUT.LGT.PROB = 0.2
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5
# Stochastic Augmentation (Díaz Benito et al., ICIPW 2025)
_C.INPUT.STOCHASTIC_AUG = CN()
_C.INPUT.STOCHASTIC_AUG.ENABLED = False
_C.INPUT.STOCHASTIC_AUG.PROB_NONE = 0.15
_C.INPUT.STOCHASTIC_AUG.PROB_ONE = 0.45
_C.INPUT.STOCHASTIC_AUG.PROB_TWO = 0.40
# Class-specific augmentation
_C.INPUT.ENABLE_CLASS_SPECIFIC_AUG = False  # faizyab/class-specific-augmentations (branch default: True)
# Class-wise augmentation (narmyn/class-wise-aug)
_C.INPUT.CLASS_SPECIFIC_AUG = CN()
_C.INPUT.CLASS_SPECIFIC_AUG.ENABLED = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ('Market1501',)
_C.DATASETS.TEST = ('DukeMTMC',)
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# combine both train and test sets
_C.DATASETS.COMBINEALL = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = True
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
_C.DATALOADER.INDIVIDUAL = False
# camera as domain
_C.DATALOADER.CAMERA_TO_DOMAIN = False # True when single-source
# drop last incomplete batch
_C.DATALOADER.DROP_LAST = False
_C.DATALOADER.DELETE_REM = False # if true, remain idx lower than num_instance
# Whether to enforce class-balanced sampling (equal images per semantic class) per batch
_C.DATALOADER.CLASS_BALANCE = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.GRAD_ACCUM_STEPS = 1

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = True

# -----------------------------------------------------------------------------
# Multi-scale test-time augmentation (TTA) defaults
# -----------------------------------------------------------------------------
# List of scale multipliers to evaluate at test-time (e.g. [1.0, 0.9, 1.1]).
# If None, multi-scale TTA is disabled and a single scale (1.0) is used.
_C.TEST.SCALES = None
# How to combine features from different scales: 'avg' or 'max'.
_C.TEST.SCALE_COMBINE = 'avg'
# Master switch to explicitly enable multi-scale testing (keeps default off).
_C.TEST.DO_MULTI_SCALE = False

# Resample TTA: also embed each view up-sampled x1.1 and resized back (narmyn/moreposthocthings)
_C.TEST.RESAMPLE_TTA = False
# Subtract the per-camera mean feature from query / gallery features and re-normalise
_C.TEST.CAM_FEAT_NORM = False

# Traffic-sign prototype / sign-type soft filter (narmyn/trafficsignprototype); uses
# sign_type_prototypes.pkl, query_sign_types.csv, test_sign_types.csv (cwd or DATASETS.ROOT_DIR)
# or the --prototype_bank / --query_sign_types / --gallery_sign_types arguments
_C.TEST.SIGN_PROTOTYPE_FILTER = False

# Query Majority Voting: ORB-matched adjacent query frames share one voted ranking (narmyn/vitlarge16)
_C.TEST.QMV = CN()
_C.TEST.QMV.ENABLED = False
_C.TEST.QMV.MATCH_THRESHOLD = 50
_C.TEST.QMV.TOP_K = 100
_C.TEST.QMV.DECAY = 10.0
_C.TEST.QMV.ORB_FEATURES = 500

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False

# Concatenate CLS tokens from last K blocks at inference
_C.TEST.CLS_FUSION = False
_C.TEST.CLS_FUSION_LAST = 6

# Dual-model inference weights. When both are set, the inference scripts route
# traffic queries to TEST.TRAFFIC_WEIGHT and all other queries to TEST.OTHER_WEIGHT.
_C.TEST.TRAFFIC_WEIGHT = ''
_C.TEST.OTHER_WEIGHT = ''

#class filtering
_C.TEST.DO_CLASS_FILTER = False

# Camera-Aware Jaccard (CAJ) adjustment for multi-camera scenarios — simple distance scaling
_C.TEST.DO_CAJ_ADJUSTMENT = False
_C.TEST.CAJ_SAME_CAM_PENALTY = 1.1  # >1 penalizes same-camera similarity
_C.TEST.CAJ_CROSS_CAM_SCALE = 0.95  # <1 favors cross-camera matches

# Concatenate part tokens (positions 1:4) onto the fused CLS feature
_C.TEST.USE_PART_TOKENS = False
_C.TEST.PART_TOKENS_LAST = 1  # number of trailing layers to draw part tokens from

# Optional traffic-sign classifier-based soft filtering
_C.TEST.TRAFFIC_SIGN_CLASSIFIER = CN()
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.ENABLED = False
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT = ""
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.MODE = "softmax"  # "softmax" or "argmax"
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.ALPHA = 0.35
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.ARGMAX_PENALTY = 0.75
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.TEMPERATURE = 1.0
_C.TEST.TRAFFIC_SIGN_CLASSIFIER.INPUT_SIZE = [224, 224]

# Retrieval test-time augmentation sizes
_C.TEST.RETRIEVAL_TTA_SIZES = [
	[224, 224],
	[224, 192],
	[192, 224],
	[256, 224],
]

# Toggle to enable/disable all retrieval TTA (resizes + flips). Defaults to False.
_C.TEST.USE_TTA = False

# Class-based re-ranking with per-class k1/k2/lambda settings
_C.TEST.DO_CLASS_BASED_RERANKING = False
_C.TEST.CLASS_BASED_RERANKING_PARAMS = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.DEFAULT = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.DEFAULT.K1 = 20
_C.TEST.CLASS_BASED_RERANKING_PARAMS.DEFAULT.K2 = 6
_C.TEST.CLASS_BASED_RERANKING_PARAMS.DEFAULT.LAMBDA = 0.3

_C.TEST.CLASS_BASED_RERANKING_PARAMS.container = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.container.K1 = 20
_C.TEST.CLASS_BASED_RERANKING_PARAMS.container.K2 = 6
_C.TEST.CLASS_BASED_RERANKING_PARAMS.container.LAMBDA = 0.3

_C.TEST.CLASS_BASED_RERANKING_PARAMS.crosswalk = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.crosswalk.K1 = 20
_C.TEST.CLASS_BASED_RERANKING_PARAMS.crosswalk.K2 = 6
_C.TEST.CLASS_BASED_RERANKING_PARAMS.crosswalk.LAMBDA = 0.3

_C.TEST.CLASS_BASED_RERANKING_PARAMS.rubbishbins = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.rubbishbins.K1 = 20
_C.TEST.CLASS_BASED_RERANKING_PARAMS.rubbishbins.K2 = 6
_C.TEST.CLASS_BASED_RERANKING_PARAMS.rubbishbins.LAMBDA = 0.3

_C.TEST.CLASS_BASED_RERANKING_PARAMS.traffic = CN()
_C.TEST.CLASS_BASED_RERANKING_PARAMS.traffic.K1 = 20
_C.TEST.CLASS_BASED_RERANKING_PARAMS.traffic.K2 = 6
_C.TEST.CLASS_BASED_RERANKING_PARAMS.traffic.LAMBDA = 0.3

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# root Path to checkpoint and saved log of trained model
_C.LOG_ROOT = ""
# root path to save tensorboard results
_C.TB_LOG_ROOT = ""
# log dir name
_C.LOG_NAME = ""