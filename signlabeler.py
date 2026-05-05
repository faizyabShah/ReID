"""
sign_labeler_for_prototypes.py
===============================
Practical traffic sign type labeling pipeline for prototype-based ReID.

REALITY CHECK ON MTSD:
    - MTSD dataset exists (100K images, 400+ classes, global coverage)
    - MTSD *does* natively label sign backs
    - But NO pretrained MTSD classifier weights are publicly downloadable
    - Training one from scratch requires 42GB download + GPU training

WHAT WE ACTUALLY USE:
    1. GTSRB ViT classifier (bazyl/gtsrb-model on HuggingFace)
       → 43 classes → merged into 6 coarse Vienna Convention types
       → Works on Spanish signs because same visual grammar
    
    2. DINOv2 features + simple back-detector
       → MVV-2025 benchmark showed DINOv2 outperforms all VLMs on signs
       → We use DINOv2 features for a lightweight front/back classifier
       → Back views have distinctly different feature distributions

HOW THIS FITS WITH PROTOTYPING:
    
    PROTOTYPE BUILDING (training time):
        - For each coarse sign type, compute prototype = average embedding
          of all FRONT-VIEW images classified as that type
        - Back views do NOT contribute to prototype computation
          (they'd pollute the prototype with gray metallic features)
        - Identity label comes from majority vote over front views
        - Back views inherit the identity's type via objectID
    
    PROTOTYPE MATCHING (inference time):
        - Front query → match to nearest prototype → search within that type
        - Back query  → SKIP prototype matching → flat search (wildcard)
        - Front gallery stays in its assigned type subset
        - Back gallery stays in ALL subsets (wildcard)
    
    This means the prototype filter is:
        front_prohibitory query → searches only prohibitory gallery signs + all backs
        back query → searches entire gallery (no filtering)
    
    The back-view query will rely purely on your PAT model's embedding
    quality to find the match — which is correct, because the PAT model
    is the only thing that can match a front to its back (same structure,
    post, surroundings).

Requirements:
    pip install torch torchvision transformers pillow pandas tqdm opencv-python scikit-learn

Usage:
    # Step 1: Label training set with objectID propagation
    python sign_labeler_for_prototypes.py label_train \
        --csv train_classes.csv \
        --image_dir image_train \
        --output train_sign_types.csv

    # Step 2: Label gallery / query (no objectID)
    python sign_labeler_for_prototypes.py label_test \
        --csv test_classes.csv \
        --image_dir image_test \
        --output test_sign_types.csv

    # Step 3: Build prototypes using your PAT model
    python sign_labeler_for_prototypes.py build_prototypes \
        --csv train_sign_types.csv \
        --image_dir image_train \
        --reid_checkpoint path/to/pat_model.pth \
        --reid_backbone vit_large \
        --output prototype_bank.pt

    # Step 4: Sanity check
    python sign_labeler_for_prototypes.py sanity_check \
        --csv train_sign_types.csv \
        --image_dir image_train \
        --output_dir sanity_sheets
"""

import os
import argparse
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COL_CLASS = "Class"
COL_ID = "Corresponding Indexes"
TRAFFIC_SIGN_LABEL = "TrafficSign"


def get_sign_mask(df):
    if COL_CLASS not in df.columns:
        raise ValueError(f"CSV missing required column: {COL_CLASS}")
    return df[COL_CLASS] == TRAFFIC_SIGN_LABEL


def ensure_object_id(df):
    if COL_ID not in df.columns:
        raise ValueError(f"CSV missing required column: {COL_ID}")
    if "objectID" not in df.columns:
        df["objectID"] = df[COL_ID]
    return df


# =============================================================================
# 1. FRONT / BACK DETECTOR
# =============================================================================
# Two-stage: fast saturation heuristic first, then GTSRB confidence as backup.
# If GTSRB classifier is very confused (low confidence on all 43 classes),
# the image is likely a back view.

def detect_back_saturation(image_path, sat_threshold=35, min_color_frac=0.12):
    """
    Fast heuristic: sign backs are gray/metallic with very low saturation.
    Returns True if image is likely a back view.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    color_frac = np.sum(sat > sat_threshold) / sat.size
    return color_frac < min_color_frac


def detect_back_combined(image_path, gtsrb_confidence, gtsrb_entropy,
                          sat_threshold=35, min_color_frac=0.12,
                          min_gtsrb_conf=0.3, max_gtsrb_entropy=3.0):
    """
    Combined back detection:
        1. Low saturation (gray image) → likely back
        2. GTSRB very uncertain (low conf, high entropy) → likely back
        3. Both signals together → high confidence back detection

    Returns: (is_back: bool, confidence: float)
    """
    sat_is_back = detect_back_saturation(image_path, sat_threshold, min_color_frac)

    # GTSRB confusion signal: if classifier is confused, image may not be a front
    gtsrb_confused = (gtsrb_confidence < min_gtsrb_conf) or (gtsrb_entropy > max_gtsrb_entropy)

    if sat_is_back and gtsrb_confused:
        return True, 0.95     # high confidence back
    elif sat_is_back:
        return True, 0.70     # saturation says back, classifier somewhat confident
    elif gtsrb_confused:
        return False, 0.40    # classifier confused but colorful — might be unusual front
    else:
        return False, 0.05    # colorful + classifier confident → front


# =============================================================================
# 2. GTSRB CLASSIFIER
# =============================================================================

GTSRB_TO_COARSE = {
    # Prohibitory (red circle)
    0: "prohibitory", 1: "prohibitory", 2: "prohibitory", 3: "prohibitory",
    4: "prohibitory", 5: "prohibitory", 6: "prohibitory", 7: "prohibitory",
    8: "prohibitory", 9: "prohibitory", 10: "prohibitory",
    15: "prohibitory", 16: "prohibitory", 17: "prohibitory",

    # Warning / Danger (red triangle)
    11: "warning", 18: "warning", 19: "warning", 20: "warning",
    21: "warning", 22: "warning", 23: "warning", 24: "warning",
    25: "warning", 26: "warning", 27: "warning", 28: "warning",
    29: "warning", 30: "warning", 31: "warning",

    # Mandatory (blue circle)
    33: "mandatory", 34: "mandatory", 35: "mandatory", 36: "mandatory",
    37: "mandatory", 38: "mandatory", 39: "mandatory", 40: "mandatory",

    # Special
    12: "priority", 13: "yield", 14: "stop",

    # Derestriction / end-of-restriction
    32: "derestriction", 41: "derestriction", 42: "derestriction",
}

# The coarse types that prototypes will be built for
PROTOTYPE_TYPES = [
    "prohibitory", "warning", "mandatory", "stop", "yield", "priority",
    "derestriction",
]

# Types that bypass prototype filtering entirely at inference time.
# "sign_back" = back view detected by combined saturation + GTSRB-uncertainty signal.
# "unknown"   = front view that GTSRB couldn't classify with confidence.
# Both search the full gallery (flat retrieval), which is correct:
#   - A back-view query relies on PAT's instance-level features to find its match.
#   - An unknown-type query has no reliable prototype to filter against.
WILDCARD_TYPES = frozenset({"sign_back", "unknown"})


def load_gtsrb_classifier(device):
    """Load bazyl/gtsrb-model from HuggingFace."""
    from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor

    model_name = "bazyl/gtsrb-model"
    processor = ViTImageProcessor.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "id2label", None):
        sanitized = {}
        for k, v in config.id2label.items():
            idx = int(k)
            sanitized[idx] = v if v is not None else f"unknown_{idx}"
        config.id2label = sanitized
        config.label2id = {label: idx for idx, label in sanitized.items()}
        config.num_labels = max(getattr(config, "num_labels", 0), len(sanitized))

    model = ViTForImageClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded: {model_name}")
    return model, processor


def classify_batch(model, processor, image_paths, device, batch_size=32):
    """
    Classify images with GTSRB ViT.
    Returns: class_ids, confidences, entropies (lists)
    """
    all_ids, all_confs, all_entropies = [], [], []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        valid = []

        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid.append(j)
            except Exception:
                pass

        # Default values for failed images
        batch_ids = [-1] * len(batch_paths)
        batch_confs = [0.0] * len(batch_paths)
        batch_ents = [10.0] * len(batch_paths)

        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)
                # Entropy: high = uncertain, low = confident
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)

            for k, idx in enumerate(valid):
                batch_ids[idx] = preds[k].item()
                batch_confs[idx] = confs[k].item()
                batch_ents[idx] = entropy[k].item()

        all_ids.extend(batch_ids)
        all_confs.extend(batch_confs)
        all_entropies.extend(batch_ents)

    return all_ids, all_confs, all_entropies


# =============================================================================
# 3. LABEL TRAINING SET
# =============================================================================

def label_train(csv_path, image_dir, output_path):
    """
    Training set labeling with objectID propagation.

    Pipeline per image:
        1. Run GTSRB classifier → get class_id, confidence, entropy
        2. Run combined back detector (saturation + GTSRB uncertainty)
        3. If back → viewpoint="back", sign_type inherited from identity
        4. If front → viewpoint="front", sign_type from GTSRB coarse mapping

    Pipeline per identity (objectID):
        5. Majority vote over front-view predictions → identity type
        6. ALL images (front + back) get the identity's type
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(csv_path)
    df = ensure_object_id(df)
    sign_mask = get_sign_mask(df)
    signs_df = df[sign_mask].copy()
    n_signs = len(signs_df)
    n_ids = signs_df["objectID"].nunique()
    print(f"Traffic signs: {n_signs} images, {n_ids} identities")

    # ── Step 1: Classify all images with GTSRB ──
    print("\n[1/4] Running GTSRB classifier on all sign images...")
    model, processor = load_gtsrb_classifier(device)
    paths = [os.path.join(image_dir, n) for n in signs_df["imageName"]]
    class_ids, confs, entropies = classify_batch(model, processor, paths, device)

    signs_df["gtsrb_class_id"] = class_ids
    signs_df["gtsrb_confidence"] = confs
    signs_df["gtsrb_entropy"] = entropies

    # ── Step 2: Detect front/back using combined signal ──
    print("\n[2/4] Detecting front vs back views...")
    viewpoints = []
    back_confs = []
    for idx, (_, row) in enumerate(tqdm(signs_df.iterrows(), total=n_signs)):
        path = os.path.join(image_dir, row["imageName"])
        is_back, back_conf = detect_back_combined(
            path, confs[idx], entropies[idx]
        )
        viewpoints.append("back" if is_back else "front")
        back_confs.append(back_conf)

    signs_df["viewpoint"] = viewpoints
    signs_df["back_confidence"] = back_confs

    n_front = viewpoints.count("front")
    n_back = viewpoints.count("back")
    print(f"  Front: {n_front}, Back: {n_back}")

    # Map GTSRB class to coarse type (only meaningful for fronts)
    signs_df["raw_coarse"] = signs_df["gtsrb_class_id"].map(
        lambda x: GTSRB_TO_COARSE.get(x, "unknown")
    )

    # ── Step 3: Majority vote per identity using front views ──
    print("\n[3/4] Majority vote per identity...")
    identity_types = {}
    review_items = []

    for oid, group in signs_df.groupby("objectID"):
        fronts = group[group["viewpoint"] == "front"]
        front_types = fronts["raw_coarse"].tolist()
        front_types = [t for t in front_types if t != "unknown"]

        if not front_types:
            # Identity has NO classifiable front views
            identity_types[oid] = "unknown"
            review_items.append({
                "objectID": oid,
                "type": "unknown",
                "reason": "no_front_views",
                "n_total": len(group),
                "n_front": len(fronts),
                "n_back": len(group) - len(fronts),
            })
            continue

        counter = Counter(front_types)
        top, top_n = counter.most_common(1)[0]
        conf = top_n / len(front_types)
        identity_types[oid] = top

        if conf < 0.5:
            review_items.append({
                "objectID": oid,
                "type": top,
                "reason": "low_vote_confidence",
                "confidence": round(conf, 3),
                "votes": dict(counter),
                "n_total": len(group),
                "n_front": len(fronts),
                "n_back": len(group) - len(fronts),
            })

    # ── Step 4: Propagate to all images ──
    print("\n[4/4] Propagating labels...")
    signs_df["sign_type"] = signs_df["objectID"].map(identity_types)

    # Merge into full dataframe
    for col in ["sign_type", "viewpoint", "gtsrb_confidence", "gtsrb_entropy",
                "back_confidence"]:
        df[col] = None

    for col in ["sign_type", "viewpoint", "gtsrb_confidence", "gtsrb_entropy",
                "back_confidence"]:
        df.loc[sign_mask, col] = signs_df[col].values

    df.to_csv(output_path, index=False)

    # ── Summary ──
    print(f"\nSaved → {output_path}")
    print("\nType distribution (by identity):")
    for t, c in Counter(identity_types.values()).most_common():
        print(f"  {t:20s}: {c} identities")

    n_unknown = sum(1 for v in identity_types.values() if v == "unknown")
    print(f"\nUnknown identities (no front views): {n_unknown}")
    print(f"Identities needing review: {len(review_items)}")

    if review_items:
        review_path = output_path.replace(".csv", "_review.csv")
        pd.DataFrame(review_items).to_csv(review_path, index=False)
        print(f"Review file → {review_path}")


# =============================================================================
# 4. LABEL GALLERY / QUERY
# =============================================================================

def label_test(csv_path, image_dir, output_path):
    """
    Gallery/query labeling (no objectID).

    Front views → classified by GTSRB → coarse type
    Back views  → labeled "sign_back" → WILDCARD at inference

    The wildcard means:
        - A back query will search the ENTIRE gallery (no prototype filter)
        - A front query will see back gallery images in its search space
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(csv_path)
    df = ensure_object_id(df)
    sign_mask = get_sign_mask(df)
    signs = df[sign_mask].copy()
    print(f"Traffic signs: {len(signs)} images")

    # Classify with GTSRB
    print("\n[1/2] Running GTSRB classifier...")
    model, processor = load_gtsrb_classifier(device)
    paths = [os.path.join(image_dir, n) for n in signs["imageName"]]
    class_ids, confs, entropies = classify_batch(model, processor, paths, device)

    # Detect front/back
    print("\n[2/2] Detecting viewpoints...")
    sign_types = []
    viewpoints = []

    for idx, (_, row) in enumerate(tqdm(signs.iterrows(), total=len(signs))):
        path = os.path.join(image_dir, row["imageName"])
        is_back, _ = detect_back_combined(path, confs[idx], entropies[idx])

        if is_back:
            viewpoints.append("back")
            sign_types.append("sign_back")  # WILDCARD
        else:
            viewpoints.append("front")
            coarse = GTSRB_TO_COARSE.get(class_ids[idx], "unknown")
            sign_types.append(coarse)

    signs["sign_type"] = sign_types
    signs["viewpoint"] = viewpoints
    signs["gtsrb_confidence"] = confs

    # Merge back
    for col in ["sign_type", "viewpoint", "gtsrb_confidence"]:
        df[col] = None
    df.loc[sign_mask, "sign_type"] = signs["sign_type"].values
    df.loc[sign_mask, "viewpoint"] = signs["viewpoint"].values
    df.loc[sign_mask, "gtsrb_confidence"] = signs["gtsrb_confidence"].values

    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    print("\nType distribution:")
    for t, c in pd.Series(sign_types).value_counts().items():
        print(f"  {t:20s}: {c}")


# =============================================================================
# 5. BUILD PROTOTYPES (using your PAT model)
# =============================================================================

def _detect_model_type(state_dict):
    """Auto-detect model type (vit_large, vit_base) from checkpoint."""
    if 'base.blocks.23.norm1.weight' in state_dict:
        return 'vit_large'
    elif 'base.blocks.11.norm1.weight' in state_dict:
        return 'vit_base'
    elif 'classifier.weight' in state_dict:
        hidden_dim = state_dict['classifier.weight'].shape[1]
        return 'vit_large' if hidden_dim == 1024 else 'vit_base'
    elif 'base.pos_embed' in state_dict:
        hidden_dim = state_dict['base.pos_embed'].shape[2]
        return 'vit_large' if hidden_dim == 1024 else 'vit_base'
    return 'vit_large'

def _detect_num_classes(state_dict):
    """Auto-detect number of classes from checkpoint."""
    if 'classifier.weight' in state_dict:
        return state_dict['classifier.weight'].shape[0]
    return 1000

def build_prototypes(csv_path, image_dir, reid_checkpoint, reid_backbone,
                     output_path, min_samples=10):
    """
    Build one prototype vector per coarse sign type using your PAT model.

    CRITICAL DESIGN CHOICES:
        - Only FRONT views contribute to prototypes
        - Back views are excluded (they'd add gray metallic noise)
        - "unknown" type doesn't get a prototype (it's a wildcard)
        - Each prototype = L2-normalized mean of all front-view embeddings

    Types in WILDCARD_TYPES ('sign_back', 'unknown') are skipped here —
    they contribute no prototype at build time and trigger flat retrieval
    at inference time (see prototype_filter / retrieve_ranked).

    Output: prototype_bank.pt containing:
        - prototypes: dict {sign_type: tensor of shape [D]}
        - metadata: {sign_type: n_images, n_identities}
    """
    from torchvision import transforms, models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(csv_path)
    df = ensure_object_id(df)
    signs = df[get_sign_mask(df)].copy()

    # Only front views with classifiable types — wildcards are excluded
    front_signs = signs[
        (signs["viewpoint"] == "front") &
        (signs["sign_type"].notna()) &
        (~signs["sign_type"].isin(WILDCARD_TYPES))
    ].copy()

    print(f"Front-view signs for prototypes: {len(front_signs)}")

    # ── Load ReID backbone ──
    # Load your custom PAT model (from make_model.py)
    ckpt = torch.load(reid_checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    num_classes = _detect_num_classes(state)

    # Auto-detect if not provided
    if reid_backbone is None:
        reid_backbone = _detect_model_type(state)
        print(f"\nAuto-detected backbone: {reid_backbone}")
    else:
        print(f"\nLoading ReID backbone: {reid_backbone}")

    if reid_backbone == "vit_large":
        from model.make_model import build_part_attention_vit
        from model.backbones.vit_pytorch import part_attention_vit_large

        class DummyConfig:
            class MODEL:
                TRANSFORMER_TYPE = 'vit_large_patch16_224_TransReID'
                STRIDE_SIZE = 16
                DROP_PATH = 0.1
                DROP_OUT = 0.0
                ATT_DROP_RATE = 0.0
                PRETRAIN_PATH = ""
                PRETRAIN_CHOICE = 'imagenet'
                COS_LAYER = False
                NECK = 'bnneck'
            class INPUT:
                SIZE_TRAIN = [256, 128]
            class TEST:
                NECK_FEAT = 'after'
                CLS_FUSION = False

        dummy_cfg = DummyConfig()
        factory = {'vit_large_patch16_224_TransReID': part_attention_vit_large}

        backbone = build_part_attention_vit(num_classes=num_classes, cfg=dummy_cfg,
                                             factory=factory, pretrain_tag='imagenet')
        feat_dim = 1024
        backbone.load_state_dict(state, strict=False)
        print(f"✅ Loaded PAT checkpoint ({num_classes} classes)")

    elif reid_backbone == "vit_base":
        from model.make_model import build_part_attention_vit
        from model.backbones.vit_pytorch import part_attention_vit_base

        class DummyConfig:
            class MODEL:
                TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
                STRIDE_SIZE = 16
                DROP_PATH = 0.1
                DROP_OUT = 0.0
                ATT_DROP_RATE = 0.0
                PRETRAIN_PATH = ""
                PRETRAIN_CHOICE = 'imagenet'
                COS_LAYER = False
                NECK = 'bnneck'
            class INPUT:
                SIZE_TRAIN = [256, 128]
            class TEST:
                NECK_FEAT = 'after'
                CLS_FUSION = False

        dummy_cfg = DummyConfig()
        factory = {'vit_base_patch16_224_TransReID': part_attention_vit_base}

        backbone = build_part_attention_vit(num_classes=num_classes, cfg=dummy_cfg,
                                             factory=factory, pretrain_tag='imagenet')
        feat_dim = 768
        backbone.load_state_dict(state, strict=False)
        print(f"✅ Loaded PAT checkpoint ({num_classes} classes)")

    elif reid_backbone == "resnet50":
        backbone = models.resnet50(weights=None)
        feat_dim = 2048
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v
                   for k, v in state.items() if "classifier" not in k}
        backbone.load_state_dict(cleaned, strict=False)
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {reid_backbone}. "
                        f"Choose from: vit_large, vit_base, resnet50")

    backbone = backbone.to(device)
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # PAT training size
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # PAT normalization
    ])

    # ── Extract features per type ──
    prototypes = {}
    metadata = {}

    for sign_type, group in front_signs.groupby("sign_type"):
        if len(group) < min_samples:
            print(f"  Skipping {sign_type}: only {len(group)} samples "
                  f"(need {min_samples})")
            continue

        print(f"  Extracting features for {sign_type} ({len(group)} images)...")
        features = []

        paths = [os.path.join(image_dir, n) for n in group["imageName"]]
        batch_size = 64

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(transform(img))
                except Exception:
                    imgs.append(torch.zeros(3, 256, 128))  # Match transform size

            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                out = backbone(batch)
                # PAT model returns: (cls_score, layerwise_cls_tokens, layerwise_part_tokens)
                # or just layerwise tokens depending on training mode
                if isinstance(out, tuple):
                    # Training mode: (cls_score, layerwise_cls_tokens, layerwise_part_tokens)
                    feats = out[1][-1] if len(out) > 1 else out[0]  # Last layer CLS token
                elif isinstance(out, list):
                    # Inference mode: list of layerwise tokens
                    feats = out[-1][:, 0]  # Last layer CLS token
                else:
                    # Single tensor output
                    if out.dim() == 3:
                        feats = out[:, 0, :]  # CLS token
                    else:
                        feats = out
            features.append(feats.cpu())
        # L2 normalize each feature, then average, then normalize again
        all_feats = F.normalize(all_feats, dim=1)
        prototype = all_feats.mean(dim=0)
        prototype = F.normalize(prototype, dim=0)

        prototypes[sign_type] = prototype
        metadata[sign_type] = {
            "n_images": len(group),
            "n_identities": group["objectID"].nunique() if "objectID" in group.columns else "N/A",
        }

    # ── Validate: check inter-prototype distances ──
    print("\n" + "=" * 60)
    print("PROTOTYPE VALIDATION")
    print("=" * 60)

    types = list(prototypes.keys())
    if len(types) >= 2:
        print("\nInter-prototype cosine similarities:")
        print("  (< 0.7 = well separated, > 0.9 = dangerously close)")
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                sim = torch.dot(prototypes[types[i]], prototypes[types[j]]).item()
                flag = " ⚠️ OVERLAP" if sim > 0.85 else ""
                print(f"  {types[i]:15s} ↔ {types[j]:15s}: {sim:.3f}{flag}")

    # ── Save ──
    torch.save({
        "prototypes": prototypes,
        "metadata": metadata,
        "feature_dim": feat_dim,
    }, output_path)

    print(f"\nSaved prototype bank → {output_path}")
    print(f"Types with prototypes: {list(prototypes.keys())}")
    print(f"\nTypes WITHOUT prototypes (wildcard flat-retrieval at inference):")
    for wt in sorted(WILDCARD_TYPES):
        print(f"  - '{wt}'")
    print(f"  - Any PROTOTYPE_TYPE with < {min_samples} samples")


# =============================================================================
# 6. PROTOTYPE-BASED RETRIEVAL FILTER
# =============================================================================
#
# The skip logic has three paths:
#
#   Query type         → Action
#   ─────────────────────────────────────────────────────────────────────────
#   sign_back          → SKIP filter, search entire gallery  (wildcard)
#   unknown            → SKIP filter, search entire gallery  (wildcard)
#   known front type,
#     but ambiguous    → SKIP filter, search entire gallery  (fallback)
#   known front type,
#     confident        → FILTER gallery to matching type + all back gallery
#                        images (backs in the gallery are always wildcards —
#                        a front query must be able to match the back of the
#                        same sign instance)
#
# Why backs are included on the GALLERY side for front queries:
#   A front-view query for a stop sign must still be able to retrieve the
#   back-view image of that same stop sign (same objectID), because back
#   views exist in the gallery. Excluding gallery backs would break cross-
#   viewpoint Re-ID entirely.
#
# Why back QUERIES skip the filter:
#   Back views were never seen by GTSRB during training, so the classifier
#   is uninformative. More importantly, the PAT model's embedding is the
#   only signal that can match a back view to its front — the prototype
#   filter would only add noise and reduce the search space arbitrarily.
# =============================================================================

_MIN_FILTERED_GALLERY = 50   # safety floor before padding kicks in
_PAD_TO = 100                 # pad up to this many candidates


def prototype_filter(query_embedding, query_sign_type,
                     gallery_embeddings, gallery_sign_types,
                     prototype_bank, confidence_margin=0.15):
    """
    Return the gallery indices a query should search against.

    Args:
        query_embedding:    tensor [D], L2-normalized query feature
        query_sign_type:    str — output of label_test ('sign_back', 'unknown',
                            'prohibitory', 'warning', etc.)
        gallery_embeddings: tensor [N, D]
        gallery_sign_types: list[str] length N
        prototype_bank:     dict loaded from build_prototypes output (.pt file)
        confidence_margin:  min gap between top-2 prototype cosine similarities
                            required to trust the type assignment; if below
                            this, fall back to flat retrieval

    Returns:
        filtered_indices: list[int] — indices into gallery to search
        filter_mode:      'wildcard' | 'fallback' | 'prototype'
    """
    prototypes = prototype_bank["prototypes"]

    # ── Path 1: wildcard — back views and unclassifiable fronts ──────────────
    # Skip the prototype filter entirely; let PAT's embeddings do the work.
    if query_sign_type in WILDCARD_TYPES:
        return list(range(len(gallery_sign_types))), "wildcard"

    # ── Path 2: no prototype available for this type ──────────────────────────
    if query_sign_type not in prototypes:
        return list(range(len(gallery_sign_types))), "wildcard"

    # ── Path 3: compare query to all prototypes ───────────────────────────────
    # We use the prototype comparison rather than the raw classifier label
    # because the prototype is computed from PAT features — it lives in the
    # same embedding space as the query, making it more reliable than a
    # per-image GTSRB prediction.
    q_norm = F.normalize(query_embedding, dim=0)
    sims = {ptype: torch.dot(q_norm, pvec).item()
            for ptype, pvec in prototypes.items()}

    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_type, best_sim = sorted_sims[0]
    second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

    # ── Path 4: ambiguous — top-2 prototypes too close ───────────────────────
    if best_sim - second_sim < confidence_margin:
        return list(range(len(gallery_sign_types))), "fallback"

    # ── Path 5: confident prototype match — filter gallery ───────────────────
    # Keep gallery items that match the assigned type, PLUS all gallery backs
    # (backs are wildcards on the gallery side — see section header above).
    filtered = [
        idx for idx, gtype in enumerate(gallery_sign_types)
        if gtype == best_type or gtype in WILDCARD_TYPES
    ]

    # Safety floor: if the filtered set is very small, pad with the nearest
    # gallery items by embedding distance so retrieval isn't starved.
    if len(filtered) < _MIN_FILTERED_GALLERY:
        filtered_set = set(filtered)
        remaining = [i for i in range(len(gallery_sign_types))
                     if i not in filtered_set]
        if remaining:
            q_exp = F.normalize(query_embedding.unsqueeze(0), dim=1)
            g_rem = F.normalize(gallery_embeddings[remaining], dim=1)
            nearest = torch.mm(q_exp, g_rem.T).squeeze(0).argsort(descending=True)
            filtered.extend(remaining[i] for i in nearest.tolist()
                            if len(filtered) < _PAD_TO)

    return filtered, "prototype"


def retrieve_ranked(query_embedding, query_sign_type,
                    gallery_embeddings, gallery_sign_types,
                    prototype_bank, top_k=100):
    """
    End-to-end retrieval: prototype filter → cosine ranking.

    This is the single entry point for inference.  The skip logic is
    fully contained inside prototype_filter; here we just rank whatever
    gallery subset it returns.

    Args:
        query_embedding:    tensor [D]
        query_sign_type:    str ('sign_back' triggers wildcard skip)
        gallery_embeddings: tensor [N, D]
        gallery_sign_types: list[str] length N
        prototype_bank:     dict from build_prototypes
        top_k:              how many ranked results to return

    Returns:
        ranked_indices: list[int] length ≤ top_k, gallery indices by similarity
        filter_mode:    str — 'wildcard' | 'fallback' | 'prototype'
    """
    indices, filter_mode = prototype_filter(
        query_embedding, query_sign_type,
        gallery_embeddings, gallery_sign_types,
        prototype_bank,
    )

    q_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
    g_sub = F.normalize(gallery_embeddings[indices], dim=1)
    sims = torch.mm(q_norm, g_sub.T).squeeze(0)

    order = sims.argsort(descending=True).tolist()
    ranked = [indices[i] for i in order[:top_k]]
    return ranked, filter_mode


# =============================================================================
# 7. SANITY CHECK
# =============================================================================

def sanity_check(csv_path, image_dir, output_dir, n_samples=8):
    """Visual grid per type × viewpoint for verification."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = ensure_object_id(df)
    signs = df[get_sign_mask(df)].copy()

    for (sign_type, viewpoint), group in signs.groupby(["sign_type", "viewpoint"]):
        if pd.isna(sign_type):
            continue

        sample = group.sample(min(n_samples, len(group)), random_state=42)
        n = len(sample)
        cols = min(4, n)
        rows = max(1, (n + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
        fig.suptitle(
            f"{sign_type} | {viewpoint} | {len(group)} total",
            fontsize=12, fontweight="bold"
        )

        axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
        for i, (_, row) in enumerate(sample.iterrows()):
            path = os.path.join(image_dir, row["imageName"])
            if os.path.exists(path):
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
            conf = row.get("gtsrb_confidence", "?")
            axes[i].set_title(f"{row['imageName']}\nconf={conf}", fontsize=6)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        safe_type = str(sign_type).replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_type}_{viewpoint}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()

    print(f"Sanity check images → {output_dir}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("label_train")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--image_dir", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("label_test")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--image_dir", required=True)
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("build_prototypes")
    p3.add_argument("--csv", required=True)
    p3.add_argument("--image_dir", required=True)
    p3.add_argument("--reid_checkpoint", required=True)
    p3.add_argument("--reid_backbone", required=False, default=None,
                    choices=["vit_large", "vit_base", "resnet50"],
                    help="Auto-detected from checkpoint if not provided")
    p3.add_argument("--output", required=True)
    p3.add_argument("--min_samples", type=int, default=10)

    p4 = sub.add_parser("sanity_check")
    p4.add_argument("--csv", required=True)
    p4.add_argument("--image_dir", required=True)
    p4.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.cmd == "label_train":
        label_train(args.csv, args.image_dir, args.output)
    elif args.cmd == "label_test":
        label_test(args.csv, args.image_dir, args.output)
    elif args.cmd == "build_prototypes":
        build_prototypes(args.csv, args.image_dir, args.reid_checkpoint,
                         args.reid_backbone, args.output, args.min_samples)
    elif args.cmd == "sanity_check":
        sanity_check(args.csv, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
