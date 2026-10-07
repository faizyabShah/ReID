import os
import csv
import torch
import torch.nn.functional as F
import argparse
from itertools import combinations
from collections import defaultdict
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from data.build_DG_dataloader import fast_batch_collator
from data.common import CommDataset
from data.transforms.build import build_transforms
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from utils.class_split import is_traffic_class, normalize_class_name as shared_normalize_class_name, read_class_map, resolve_image_path

#from torch.backends import cudnn

def read_classes_from_csv(csv_path):
    """Reads the CSV and returns a list of classes in order.
    
    Dynamically finds the 'Class' column to support different CSV formats.
    """
    classes = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header
        class_index = header.index('Class')  # Find the Class column dynamically
        for row in reader:
            classes.append(row[class_index])
    return classes

def apply_class_penalty(dist_mat, query_classes, gallery_classes):
    """Applies a huge penalty to the distance matrix for mismatched classes."""
    # Convert lists to numpy arrays and reshape for broadcasting
    q_classes = np.array(query_classes)[:, np.newaxis]  # Shape: (num_query, 1)
    g_classes = np.array(gallery_classes)[np.newaxis, :]  # Shape: (1, num_gallery)
    
    # Create a boolean mask (True where classes do NOT match)
    mismatch_mask = (q_classes != g_classes)
    
    # Add a massive penalty to the mismatched pairs
    penalized_dist = np.copy(dist_mat)
    penalized_dist[mismatch_mask] += 1e6
    
    return penalized_dist

def apply_caj(q_g_dist, q_q_dist, g_g_dist, q_camids, g_camids,
              same_cam_penalty=1.1, cross_cam_scale=0.95):
    """Camera-Aware Jaccard (CAJ) adjustment for multi-camera retrieval.
    
    Args:
        q_g_dist: (num_query, num_gallery)
        q_q_dist: (num_query, num_query)
        g_g_dist: (num_gallery, num_gallery)
        q_camids: list/array (num_query,)
        g_camids: list/array (num_gallery,)
        same_cam_penalty: >1 → penalize same-camera similarity
        cross_cam_scale: <1 → slightly favor cross-camera matches
    
    Returns:
        adjusted q_g_dist, q_q_dist, g_g_dist
    """
    q_camids = np.array(q_camids)
    g_camids = np.array(g_camids)

    # ---- adjust q-q (same camera bias) ----
    qq_same = q_camids[:, None] == q_camids[None, :]
    q_q_dist = q_q_dist.copy()
    q_q_dist[qq_same] *= same_cam_penalty

    # ---- adjust g-g (same camera bias) ----
    gg_same = g_camids[:, None] == g_camids[None, :]
    g_g_dist = g_g_dist.copy()
    g_g_dist[gg_same] *= same_cam_penalty

    # ---- adjust q-g (optional cross-camera boost) ----
    qg_same = q_camids[:, None] == g_camids[None, :]
    q_g_dist = q_g_dist.copy()

    # penalize same cam
    q_g_dist[qg_same] *= same_cam_penalty

    # boost cross-camera
    q_g_dist[~qg_same] *= cross_cam_scale

    return q_g_dist, q_q_dist, g_g_dist

def normalize_class_name(class_name):
    return shared_normalize_class_name(class_name)

def read_sign_types_from_csv(csv_path):
    """Return {imageName: (sign_type, viewpoint)}."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "imageName" not in df.columns:
        raise ValueError(f"CSV missing required column: imageName ({csv_path})")
    sign_col = "sign_type" if "sign_type" in df.columns else None
    view_col = "viewpoint" if "viewpoint" in df.columns else None

    mapping = {}
    for _, row in df.iterrows():
        key = str(row["imageName"]).replace("\\", "/").lstrip("./")
        sign_type = row[sign_col] if sign_col else None
        viewpoint = row[view_col] if view_col else None
        sign_type = None if pd.isna(sign_type) else str(sign_type)
        viewpoint = None if pd.isna(viewpoint) else str(viewpoint)
        mapping[key] = (sign_type, viewpoint)
    return mapping

def resolve_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

def _safe_rel_image_key(img_path, base_dir):
    img_path = os.path.abspath(img_path)
    if base_dir:
        try:
            base_dir = os.path.abspath(base_dir)
            if os.path.commonpath([base_dir, img_path]) == base_dir:
                rel = os.path.relpath(img_path, base_dir)
                return rel.replace("\\", "/")
        except Exception:
            pass
    return os.path.basename(img_path)

def load_prototypes(bank_path):
    bank = torch.load(bank_path, map_location="cpu")
    prototypes = bank.get("prototypes", {})
    if not prototypes:
        raise ValueError(f"No prototypes found in {bank_path}")
    types = list(prototypes.keys())
    proto_mat = torch.stack([prototypes[t].cpu() for t in types], dim=0).numpy()
    proto_norm = np.linalg.norm(proto_mat, axis=1, keepdims=True)
    proto_mat = proto_mat / np.maximum(proto_norm, 1e-12)
    return types, proto_mat

def apply_prototype_filter(dist_mat, qf, q_keys, g_keys, query_sign_map, gallery_sign_map,
                           proto_types, proto_mat, proto_margin=0.05, penalty=0.1,
                           query_classes=None):
    """
    Soft shape-label penalty. No prototype embeddings used.
    Just compares shape labels from the classifier and adds a small
    distance penalty for mismatched shapes.
    """
    q_sign_types = []
    q_viewpoints = []
    for key in q_keys:
        item = query_sign_map.get(key) or query_sign_map.get(os.path.basename(key))
        if not item:
            q_sign_types.append(None)
            q_viewpoints.append(None)
        else:
            q_sign_types.append(item[0])
            q_viewpoints.append(item[1])

    g_sign_types = []
    g_viewpoints = []
    for key in g_keys:
        item = gallery_sign_map.get(key) or gallery_sign_map.get(os.path.basename(key))
        if not item:
            g_sign_types.append(None)
            g_viewpoints.append(None)
        else:
            g_sign_types.append(item[0])
            g_viewpoints.append(item[1])

    g_sign_types = np.array(g_sign_types, dtype=object)
    g_viewpoints = np.array(g_viewpoints, dtype=object)

    filtered = 0
    skipped = 0

    for i in range(dist_mat.shape[0]):
        # Only apply to traffic sign queries
        if query_classes is not None:
            if normalize_class_name(query_classes[i]) != "traffic":
                continue

        q_type = q_sign_types[i]
        q_view = q_viewpoints[i]

        # Skip back views and unknowns — flat retrieval for these
        if q_view == "back" or q_type in {"sign_back", "unknown", "other", None}:
            skipped += 1
            continue

        # Vectorized: penalize gallery items with different shape
        # but skip backs and wildcards on the gallery side
        mismatch = (
            (g_sign_types != q_type) &
            (g_viewpoints != "back") &
            (g_sign_types != "sign_back") &
            (g_sign_types != "unknown") &
            (g_sign_types != "other") &
            (g_sign_types != None)
        )
        dist_mat[i, mismatch] += penalty
        filtered += 1

    print(f"[shape] Soft penalty applied to {filtered} queries "
          f"(skipped {skipped} back/unknown), penalty={penalty}")
    return dist_mat

def _extract_rerank_params(param_node):
    return {
        "k1": int(param_node.K1),
        "k2": int(param_node.K2),
        "lambda_value": float(param_node.LAMBDA),
    }

def apply_class_based_reranking(q_g_dist, q_q_dist, g_g_dist, query_classes, gallery_classes,
                                class_rerank_cfg):
    """Apply per-class reranking overrides on top of a base reranked matrix."""
    query_classes = np.array([normalize_class_name(cls) for cls in query_classes])
    gallery_classes = np.array([normalize_class_name(cls) for cls in gallery_classes])

    default_params = _extract_rerank_params(class_rerank_cfg.DEFAULT)
    final_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, **default_params)

    all_classes = sorted(set(query_classes.tolist()) | set(gallery_classes.tolist()))
    for class_name in all_classes:
        q_indices = np.where(query_classes == class_name)[0]
        g_indices = np.where(gallery_classes == class_name)[0]
        if len(q_indices) == 0 or len(g_indices) == 0:
            continue

        class_params = class_rerank_cfg[class_name] if class_name in class_rerank_cfg else class_rerank_cfg.DEFAULT
        class_params = _extract_rerank_params(class_params)
        total_num = len(q_indices) + len(g_indices)
        if total_num < 2:
            continue

        class_params["k1"] = max(1, min(class_params["k1"], total_num - 1))
        class_params["k2"] = max(1, min(class_params["k2"], total_num))

        class_dist = re_ranking(
            q_g_dist[np.ix_(q_indices, g_indices)],
            q_q_dist[np.ix_(q_indices, q_indices)],
            g_g_dist[np.ix_(g_indices, g_indices)],
            **class_params,
        )
        final_dist[np.ix_(q_indices, g_indices)] = class_dist

    return final_dist

def extract_feature(model, dataloaders, num_query, last_k=6, use_part_tokens=False, part_tokens_last=1,
                    scales=None, combine_method='avg', resample_tta=False):
    """Extract features with CLS fusion (concat of last K layers' CLS tokens) plus flip TTA.

    If use_part_tokens is True, concatenates the 3 part tokens (positions 1:4) from
    the last `part_tokens_last` layers onto the fused CLS feature.

    Multi-resolution TTA (faizyab/post-hoc-multi-res): `scales` is a list of
    scale multipliers relative to the input size (e.g. [1.0, 0.9, 1.1]); every
    scaled image and its horizontal flip are embedded and combined with
    `combine_method` 'avg' (default) or 'max'.

    Resample TTA (narmyn/moreposthocthings): with `resample_tta`, each view is
    also up-sampled by 1.1 and resized back to its input size.
    """
    features = []
    camids = []
    count = 0
    backbone = model.base

    # default scales fallback
    if scales is None:
        scales = getattr(cfg.TEST, 'SCALES', None) or [1.0]
    resample_scales = [1.0, 1.1] if resample_tta else [1.0]

    for data in dataloaders:
        img = data['images']
        camid = data['camid']
        n, c, h, w = img.size()
        count += n

        agg_feats = None
        max_feats = []

        for scale in scales:
            if scale == 1.0:
                scaled = img
            else:
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))
                scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

            for resample in resample_scales:
                for flip_i in range(2):
                    input_img = scaled.cuda()
                    if resample != 1.0:
                        input_img = torch.nn.functional.interpolate(
                            input_img, scale_factor=resample, mode='bilinear', align_corners=False
                        )
                        input_img = torch.nn.functional.interpolate(
                            input_img, size=scaled.shape[-2:], mode='bilinear', align_corners=False
                        )
                    if flip_i == 1:
                        input_img = torch.flip(input_img, dims=[-1])

                    layerwise_tokens = backbone(input_img)
                    cls_tokens = torch.stack([layer[:, 0] for layer in layerwise_tokens[-last_k:]], dim=1)
                    f = cls_tokens.reshape(n, -1)
                    if use_part_tokens:
                        part_tokens = torch.cat(
                            [layer[:, 1:4].reshape(n, -1) for layer in layerwise_tokens[-part_tokens_last:]],
                            dim=1,
                        )
                        f = torch.cat([f, part_tokens], dim=1)
                    f = f.float()

                    if agg_feats is None:
                        agg_feats = torch.zeros(n, f.shape[1], device=f.device)

                    if combine_method == 'avg':
                        agg_feats = agg_feats + f
                    elif combine_method == 'max':
                        max_feats.append(f)
                    else:
                        agg_feats = agg_feats + f

        if combine_method == 'avg':
            ff = agg_feats / (len(scales) * len(resample_scales) * 2)
        elif combine_method == 'max':
            ff = torch.stack(max_feats, dim=0).max(dim=0)[0]
        else:
            ff = agg_feats / (len(scales) * len(resample_scales) * 2)

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
        camids.extend(np.asarray(camid))

    features = torch.cat(features, 0)
    camids = np.array(camids)

    # query
    qf = features[:num_query]
    q_camids = camids[:num_query]
    # gallery
    gf = features[num_query:]
    g_camids = camids[num_query:]
    return qf, gf, q_camids, g_camids


def build_loader(cfg, items):
    transforms = build_transforms(cfg, is_train=False)
    dataset = CommDataset(items, transforms, relabel=False)
    return DataLoader(
        dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=fast_batch_collator,
    )


def read_split_records(split_csv_path, class_csv_path, image_root, image_subdir):
    with open(split_csv_path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {split_csv_path}")

        image_col = next((c for c in ['imageName', 'imagename', 'image_name'] if c in reader.fieldnames), None)
        pid_col = next((c for c in ['objectID', 'Corresponding Indexes', 'object', 'pid'] if c in reader.fieldnames), None)
        cam_col = next((c for c in ['cameraID', 'camid', 'camera'] if c in reader.fieldnames), None)
        if image_col is None or cam_col is None:
            raise ValueError(f"Could not find required columns in {split_csv_path}")

        class_map = read_class_map(class_csv_path)
        records = []
        for idx, row in enumerate(reader):
            image_name = str(row[image_col]).strip()
            # PID is optional (for unlabeled test sets); use index if not present
            pid = int(row[pid_col]) if pid_col else idx
            camid = int(str(row[cam_col]).strip().lstrip('cC'))
            class_name = class_map.get(image_name, class_map.get(os.path.basename(image_name)))
            records.append({
                'image_name': image_name,
                'image_path': resolve_image_path(image_root, image_subdir, image_name),
                'pid': pid,
                'camid': camid,
                'class_name': shared_normalize_class_name(class_name) if class_name is not None else None,
            })
    return records


def build_combined_items(query_records, gallery_records, class_name):
    class_name = shared_normalize_class_name(class_name)
    if class_name in {'nontraffic', 'other'}:
        selected_queries = [record for record in query_records if shared_normalize_class_name(record.get('class_name')) != 'traffic']
    elif class_name in {'all', 'any', 'none'}:
        selected_queries = list(query_records)
    else:
        selected_queries = [record for record in query_records if shared_normalize_class_name(record.get('class_name')) == class_name]
    combined_items = []
    combined_items.extend(
        (
            record['image_path'],
            record['pid'],
            record['camid'],
            {'class_name': record.get('class_name'), 'q_or_g': 'query'},
        )
        for record in selected_queries
    )
    combined_items.extend(
        (
            record['image_path'],
            record['pid'],
            record['camid'],
            {'class_name': record.get('class_name'), 'q_or_g': 'gallery'},
        )
        for record in gallery_records
    )
    return selected_queries, combined_items


def run_split_model(model, cfg, query_records, gallery_records, class_name):
    selected_queries, combined_items = build_combined_items(query_records, gallery_records, class_name)
    if not selected_queries:
        return {}

    loader = build_loader(cfg, combined_items)
    last_k = cfg.TEST.CLS_FUSION_LAST if cfg.TEST.CLS_FUSION else 1
    qf, gf, _, _ = extract_feature(model, loader, len(selected_queries), last_k=last_k,
                                   use_part_tokens=cfg.TEST.USE_PART_TOKENS,
                                   part_tokens_last=cfg.TEST.PART_TOKENS_LAST,
                                   scales=cfg.TEST.SCALES, combine_method=cfg.TEST.SCALE_COMBINE,
                                   resample_tta=cfg.TEST.RESAMPLE_TTA)
    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()
    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    return {record['image_name']: indices[row_idx] + 1 for row_idx, record in enumerate(selected_queries)}


# ---------------------------------------------------------------------------
# narmyn/vitlarge16 (update2.py): Query Majority Voting
# ---------------------------------------------------------------------------
def find_adjacent_query_pairs(query_image_paths, match_threshold=50, orb_features=500):
    """
    Uses ORB feature matching to find pairs of query images
    that are likely adjacent frames of the same object.
    Returns list of (idx_a, idx_b) pairs.
    """
    import cv2
    orb = cv2.ORB_create(nfeatures=orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    descriptors = []
    for path in query_image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, des = orb.detectAndCompute(img, None)
        descriptors.append(des)

    adjacent_pairs = []
    for i, j in combinations(range(len(query_image_paths)), 2):
        if descriptors[i] is None or descriptors[j] is None:
            continue
        matches = bf.match(descriptors[i], descriptors[j])
        if len(matches) >= match_threshold:
            adjacent_pairs.append((i, j))

    print(f"Found {len(adjacent_pairs)} adjacent query pairs")
    return adjacent_pairs


def build_query_groups(adjacent_pairs, num_queries):
    """
    Uses union-find to group transitively connected adjacent frames.
    e.g. if A-B and B-C are pairs, group = {A, B, C}
    """
    parent = list(range(num_queries))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, j in adjacent_pairs:
        union(i, j)

    groups = defaultdict(list)
    for idx in range(num_queries):
        groups[find(idx)].append(idx)

    multi_groups = [g for g in groups.values() if len(g) > 1]
    singleton_groups = [g for g in groups.values() if len(g) == 1]

    print(f"Multi-frame groups: {len(multi_groups)}")
    print(f"Singleton queries:  {len(singleton_groups)}")
    return multi_groups, singleton_groups


def apply_qmv(score_mat, query_image_paths, match_threshold=50, top_k=100, decay=10.0, orb_features=500):
    """
    QMV pipeline. Returns final rankings for all queries.
    score_mat: [Q, N] higher is better
    """
    num_queries, num_gallery = score_mat.shape

    individual_rankings = np.argsort(score_mat, axis=1)[:, ::-1]

    pairs = find_adjacent_query_pairs(query_image_paths, match_threshold, orb_features)
    multi_groups, _ = build_query_groups(pairs, num_queries)

    final_rankings = individual_rankings.copy()

    for group in multi_groups:
        gallery_scores = np.zeros(num_gallery, dtype=np.float32)
        for query_idx in group:
            ranked_indices = individual_rankings[query_idx]
            top_indices = ranked_indices[:top_k]
            for rank, gal_idx in enumerate(top_indices):
                weight = float(np.exp(-rank / decay))
                gallery_scores[gal_idx] += weight * score_mat[query_idx, gal_idx]

        voted_ranking = np.argsort(gallery_scores)[::-1]
        for query_idx in group:
            final_rankings[query_idx] = voted_ranking

    return final_rankings


# ---------------------------------------------------------------------------
# faizyab/gradient-reversal: retrieval TTA + traffic-sign classifier soft filter
# ---------------------------------------------------------------------------
TRAFFIC_SIGN_CLASS_NAMES = [
    "Bus_stop",
    "Not_allowed_double",
    "Upside_down_triangle",
    "Stop",
    "Bump",
    "Round_about",
    "Circle_minus",
    "Crossing",
    "Parking",
    "misc_circles",
    "Rectangle_roadside",
    "Warning_triangle",
    "Not_allowed_single",
    "Arrow_to_side",
    "Blue_rectangle",
]

def _checkpoint_to_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise ValueError("Unsupported checkpoint format for traffic sign classifier")

def _strip_state_dict_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "model.", "net."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    return cleaned

def _resize_batch(img, size):
    return F.interpolate(img, size=size, mode="bilinear", align_corners=False)

def _cfg_size_list_to_tuples(size_list):
    return [tuple(int(dim) for dim in size) for size in size_list]

def build_traffic_sign_classifier(cfg):
    classifier = tv_models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(TRAFFIC_SIGN_CLASS_NAMES))

    weight_path = cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT
    if not weight_path:
        raise ValueError("TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT must be set when the classifier is enabled")

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = _strip_state_dict_prefix(_checkpoint_to_state_dict(checkpoint))
    missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print("Traffic-sign classifier checkpoint loaded with partial key matching")
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    classifier.cuda()
    classifier.eval()
    return classifier

def apply_traffic_sign_soft_filter(dist_mat, query_probs, gallery_probs, mode="softmax",
                                   alpha=0.35, argmax_penalty=0.75,
                                   query_mask=None, gallery_mask=None):
    query_probs = np.asarray(query_probs)
    gallery_probs = np.asarray(gallery_probs)

    if query_mask is None:
        query_mask = np.ones(query_probs.shape[0], dtype=bool)
    else:
        query_mask = np.asarray(query_mask, dtype=bool)

    if gallery_mask is None:
        gallery_mask = np.ones(gallery_probs.shape[0], dtype=bool)
    else:
        gallery_mask = np.asarray(gallery_mask, dtype=bool)

    pair_mask = query_mask[:, np.newaxis] & gallery_mask[np.newaxis, :]
    if not np.any(pair_mask):
        return dist_mat

    if mode == "argmax":
        query_pred = np.argmax(query_probs, axis=1)[:, np.newaxis]
        gallery_pred = np.argmax(gallery_probs, axis=1)[np.newaxis, :]
        mismatch = (query_pred != gallery_pred).astype(np.float32)
        adjusted = np.copy(dist_mat)
        adjusted[pair_mask] += argmax_penalty * mismatch[pair_mask]
        return adjusted

    if mode != "softmax":
        raise ValueError(f"Unsupported classifier filtering mode: {mode}")

    class_affinity = np.matmul(query_probs, gallery_probs.T)
    adjusted = np.copy(dist_mat)
    adjusted[pair_mask] += alpha * (1.0 - class_affinity)[pair_mask]
    return adjusted

def extract_feature_tta(model, dataloaders, num_query, classifier_model=None,
                    classifier_temperature=1.0, semantic_classes=None,
                    retrieval_tta_sizes=None, classifier_input_size=None,
                    use_tta=False):
    """Extract features and collect camera IDs for CAJ adjustment."""
    features = []
    camids = []
    img_path = []
    class_probs = []
    traffic_mask = []
    normalized_classes = None
    if semantic_classes is not None:
        normalized_classes = [normalize_class_name(cls) for cls in semantic_classes]

    if classifier_input_size is None:
        classifier_input_size = (224, 224)
    if retrieval_tta_sizes is None:
        default_sizes = [
            (224, 224),
            (224, 192),
            (192, 224),
            (256, 224),
        ]
        retrieval_tta_sizes = default_sizes if use_tta else [tuple(cfg.INPUT.SIZE_TEST)]

    sample_offset = 0

    for data in dataloaders:
        img = data['images']
        camid = data['camid']
        batch_paths = data['img_path']
        n, c, h, w = img.size()

        feature_maps = []
        for size in retrieval_tta_sizes:
            flips = [False, True] if use_tta else [False]
            for flip in flips:
                if flip:
                    in_img = torch.flip(img, dims=[-1])
                else:
                    in_img = img
                input_img = _resize_batch(in_img, size).cuda()
                outputs = model(input_img)
                f = F.normalize(outputs.float(), dim=1)
                feature_maps.append(f)

        ff = torch.stack(feature_maps, dim=0).mean(dim=0)
        ff = F.normalize(ff, dim=1)
        features.append(ff)
        camids.extend(np.asarray(camid))
        img_path.extend(list(batch_paths))

        if normalized_classes is not None:
            batch_classes = normalized_classes[sample_offset:sample_offset + n]
            batch_traffic_mask = np.array([cls == "traffic" for cls in batch_classes], dtype=bool)
        else:
            batch_traffic_mask = np.ones(n, dtype=bool)
        traffic_mask.extend(batch_traffic_mask.tolist())

        if classifier_model is not None:
            batch_probs = torch.zeros((n, len(TRAFFIC_SIGN_CLASS_NAMES)), dtype=torch.float32)
            if np.any(batch_traffic_mask):
                traffic_mask_tensor = torch.from_numpy(batch_traffic_mask)
                traffic_imgs = _resize_batch(img[traffic_mask_tensor], classifier_input_size)
                cls_logits = classifier_model(traffic_imgs.cuda())
                if classifier_temperature != 1.0:
                    cls_logits = cls_logits / float(classifier_temperature)
                cls_probs = torch.softmax(cls_logits, dim=1).detach().cpu()
                batch_probs[traffic_mask_tensor] = cls_probs
            class_probs.append(batch_probs)

        sample_offset += n
    features = torch.cat(features, 0)
    camids = np.array(camids)
    class_probs = torch.cat(class_probs, 0).numpy() if class_probs else None
    traffic_mask = np.array(traffic_mask, dtype=bool)

    # query
    qf = features[:num_query]
    q_camids = camids[:num_query]
    q_img_paths = img_path[:num_query]
    q_class_probs = class_probs[:num_query] if class_probs is not None else None
    q_traffic_mask = traffic_mask[:num_query]
    # gallery
    gf = features[num_query:]
    g_camids = camids[num_query:]
    g_img_paths = img_path[num_query:]
    g_class_probs = class_probs[num_query:] if class_probs is not None else None
    g_traffic_mask = traffic_mask[num_query:]
    return qf, gf, q_camids, g_camids, q_img_paths, g_img_paths, q_class_probs, g_class_probs, q_traffic_mask, g_traffic_mask

def run_tta_classifier_pipeline(cfg, args, model):
    """Inference path of faizyab/gradient-reversal (TEST.USE_TTA / TEST.TRAFFIC_SIGN_CLASSIFIER).

    Kept verbatim: resize+flip TTA over TEST.RETRIEVAL_TTA_SIZES with per-view L2 norm,
    CAJ applied *before* re-ranking, then the traffic-sign classifier soft filter.
    """
    traffic_sign_classifier = None
    if cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ENABLED or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT:
        traffic_sign_classifier = build_traffic_sign_classifier(cfg)

    query_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "query_classes.csv")
    gallery_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "test_classes.csv")
    need_classes = (
        cfg.TEST.DO_CLASS_FILTER
        or cfg.TEST.DO_CLASS_BASED_RERANKING
        or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ENABLED
        or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT
    )
    query_classes = gallery_classes = None
    if need_classes:
        query_classes = read_classes_from_csv(query_csv_path)
        gallery_classes = read_classes_from_csv(gallery_csv_path)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
    with torch.no_grad():
        qf, gf, q_camids, g_camids, q_img_paths, g_img_paths, q_class_probs, g_class_probs, q_traffic_mask, g_traffic_mask = extract_feature_tta(
            model,
            val_loader,
            num_query,
            classifier_model=traffic_sign_classifier,
            classifier_temperature=cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.TEMPERATURE,
            semantic_classes=(query_classes + gallery_classes) if query_classes is not None and gallery_classes is not None else None,
            retrieval_tta_sizes=_cfg_size_list_to_tuples(cfg.TEST.RETRIEVAL_TTA_SIZES),
            classifier_input_size=tuple(cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.INPUT_SIZE),
            use_tta=getattr(cfg.TEST, 'USE_TTA', False),
        )

    # save feature
    qf=qf.cpu().numpy()
    gf=gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    # Apply Camera-Aware Jaccard adjustment first if enabled (influence reranking topology)
    if cfg.TEST.DO_CAJ_ADJUSTMENT:
        q_g_dist, q_q_dist, g_g_dist = apply_caj(
            q_g_dist,
            q_q_dist,
            g_g_dist,
            q_camids,
            g_camids,
            same_cam_penalty=cfg.TEST.CAJ_SAME_CAM_PENALTY,
            cross_cam_scale=cfg.TEST.CAJ_CROSS_CAM_SCALE,
        )

    if cfg.TEST.DO_CLASS_BASED_RERANKING:
        re_rank_dist = apply_class_based_reranking(
            q_g_dist,
            q_q_dist,
            g_g_dist,
            query_classes,
            gallery_classes,
            cfg.TEST.CLASS_BASED_RERANKING_PARAMS,
        )
    else:
        re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    if (cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ENABLED or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT) and q_class_probs is not None and g_class_probs is not None:
        re_rank_dist = apply_traffic_sign_soft_filter(
            re_rank_dist,
            q_class_probs,
            g_class_probs,
            mode=cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.MODE,
            alpha=cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ALPHA,
            argmax_penalty=cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ARGMAX_PENALTY,
            query_mask=q_traffic_mask,
            gallery_mask=g_traffic_mask,
        )

    if cfg.TEST.DO_CLASS_FILTER:
        re_rank_dist = apply_class_penalty(re_rank_dist, query_classes, gallery_classes)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    m, n = indices.shape
    # # print('m: {}  n: {}'.format(m, n))
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())


    lista_nombres = [os.path.basename(path) for path in q_img_paths]
    output_path = args.track.split(".txt")[0] + "_submission.csv"

    with open(output_path, 'w', newline='') as archivo_csv:
        csv_writter = csv.writer(archivo_csv)
        csv_writter.writerow(['imageName', 'Corresponding Indexes'])
        for numero, track in zip(lista_nombres, indices):
            track_str = ' '.join(map(str, track + 1))
            csv_writter.writerow([numero, track_str])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./configs/paper/test.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./outputs/track.txt", help="ranking output; also writes <track>_submission.csv", type=str
    )
    parser.add_argument(
        "--traffic_weight", default="", help="path to traffic model weights", type=str
    )
    parser.add_argument(
        "--other_weight", default="", help="path to non-traffic model weights", type=str
    )
    parser.add_argument("--prototype_bank", default=None, type=str,
                        help="Path to sign_type_prototypes.pkl")
    parser.add_argument("--query_sign_types", default=None, type=str,
                        help="Path to query_sign_types.csv")
    parser.add_argument("--gallery_sign_types", default=None, type=str,
                        help="Path to test_sign_types.csv")
    parser.add_argument("--proto_margin", default=0.05, type=float,
                        help="Prototype confidence margin for filtering")
    parser.add_argument("--proto_use_unfused", action="store_true",
                        help="Use non-fused CLS features for prototype routing")
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    os.makedirs(os.path.dirname(os.path.abspath(args.track)), exist_ok=True)

    traffic_weight = args.traffic_weight or cfg.TEST.TRAFFIC_WEIGHT
    other_weight = args.other_weight or cfg.TEST.OTHER_WEIGHT
    dual_mode = bool(traffic_weight and other_weight)
    if dual_mode:
        query_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "query.csv")
        gallery_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "test.csv")
        query_class_csv = os.path.join(cfg.DATASETS.ROOT_DIR, "query_classes.csv")
        gallery_class_csv = os.path.join(cfg.DATASETS.ROOT_DIR, "test_classes.csv")

        query_records = read_split_records(query_csv_path, query_class_csv, cfg.DATASETS.ROOT_DIR, "image_query")
        gallery_records = read_split_records(gallery_csv_path, gallery_class_csv, cfg.DATASETS.ROOT_DIR, "image_test")

        traffic_model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        traffic_model.load_param(traffic_weight)
        traffic_model.eval()
        traffic_model.cuda()
        
        other_model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        other_model.load_param(other_weight)
        other_model.eval()
        other_model.cuda()

        with torch.no_grad():
            traffic_predictions = run_split_model(traffic_model, cfg, query_records, gallery_records, cfg.MODEL.TRAFFIC_CLASS_NAME)
            other_predictions = run_split_model(other_model, cfg, query_records, gallery_records, 'non_traffic')

        output_path = args.track.split(".txt")[0] + "_submission.csv"
        with open(output_path, 'w', newline='') as archivo_csv:
            csv_writter = csv.writer(archivo_csv)
            csv_writter.writerow(['imageName', 'Corresponding Indexes'])
            for record in query_records:
                image_name = record['image_name']
                if is_traffic_class(record.get('class_name')):
                    track = traffic_predictions.get(image_name)
                else:
                    track = other_predictions.get(image_name)
                if track is None:
                    continue
                csv_writter.writerow([image_name, ' '.join(map(str, track.tolist()))])
    elif cfg.TEST.USE_TTA or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.ENABLED or cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.WEIGHT:
        model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
        model.load_param(cfg.TEST.WEIGHT)
        run_tta_classifier_pipeline(cfg, args, model)
    else:
        model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
        model.load_param(cfg.TEST.WEIGHT)
        model.eval()
        model.cuda()

        for testname in cfg.DATASETS.TEST:
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            if cfg.MODEL.NAME == 'part_attention_vit':
                do_inf_pat(cfg, model, val_loader, num_query)
            else:
                do_inf(cfg, model, val_loader, num_query)
        last_k = cfg.TEST.CLS_FUSION_LAST if cfg.TEST.CLS_FUSION else 1
        with torch.no_grad():
            qf, gf, q_camids, g_camids = extract_feature(
                model, val_loader, num_query,
                last_k=last_k,
                use_part_tokens=cfg.TEST.USE_PART_TOKENS,
                part_tokens_last=cfg.TEST.PART_TOKENS_LAST,
                scales=cfg.TEST.SCALES,
                combine_method=cfg.TEST.SCALE_COMBINE,
                resample_tta=cfg.TEST.RESAMPLE_TTA,
            )

        # save feature
        qf=qf.cpu().numpy()
        gf=gf.cpu().numpy()
        np.save("./qf.npy", qf)
        np.save("./gf.npy", gf)

        if cfg.TEST.CAM_FEAT_NORM:
            # Camera feature normalization (narmyn/moreposthocthings)
            print("[cam_norm] Normalizing features per camera...")
            for cam_id in np.unique(q_camids):
                mask = q_camids == cam_id
                centroid = qf[mask].mean(axis=0)
                qf[mask] = qf[mask] - centroid
            for cam_id in np.unique(g_camids):
                mask = g_camids == cam_id
                centroid = gf[mask].mean(axis=0)
                gf[mask] = gf[mask] - centroid

            # Re-normalize after subtraction
            qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
            gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)

        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))

        query_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "query_classes.csv")
        gallery_csv_path = os.path.join(cfg.DATASETS.ROOT_DIR, "test_classes.csv")
        need_classes = cfg.TEST.DO_CLASS_FILTER or cfg.TEST.DO_CLASS_BASED_RERANKING
        query_classes = gallery_classes = None
        if need_classes:
            query_classes = read_classes_from_csv(query_csv_path)
            gallery_classes = read_classes_from_csv(gallery_csv_path)

        if cfg.TEST.DO_CLASS_BASED_RERANKING:
            re_rank_dist = apply_class_based_reranking(
                q_g_dist,
                q_q_dist,
                g_g_dist,
                query_classes,
                gallery_classes,
                cfg.TEST.CLASS_BASED_RERANKING_PARAMS,
            )
        else:
            re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        # Apply Camera-Aware Jaccard adjustment if enabled
        if cfg.TEST.DO_CAJ_ADJUSTMENT:
            re_rank_dist, _, _ = apply_caj(
                re_rank_dist,
                q_q_dist,
                g_g_dist,
                q_camids,
                g_camids,
                same_cam_penalty=cfg.TEST.CAJ_SAME_CAM_PENALTY,
                cross_cam_scale=cfg.TEST.CAJ_CROSS_CAM_SCALE,
            )

        if cfg.TEST.SIGN_PROTOTYPE_FILTER:
            # Prototype / sign-type filtering for traffic signs (narmyn/trafficsignprototype)
            q_paths = [item[0] for item in val_loader.dataset.img_items[:num_query]]
            g_paths = [item[0] for item in val_loader.dataset.img_items[num_query:]]
            proto_bank = resolve_existing_path(
                args.prototype_bank,
                os.path.join(os.getcwd(), "sign_type_prototypes.pkl"),
                os.path.join(cfg.DATASETS.ROOT_DIR, "sign_type_prototypes.pkl"),
            )
            query_sign_csv = resolve_existing_path(
                args.query_sign_types,
                os.path.join(os.getcwd(), "query_sign_types.csv"),
                os.path.join(cfg.DATASETS.ROOT_DIR, "query_sign_types.csv"),
            )
            gallery_sign_csv = resolve_existing_path(
                args.gallery_sign_types,
                os.path.join(os.getcwd(), "test_sign_types.csv"),
                os.path.join(cfg.DATASETS.ROOT_DIR, "test_sign_types.csv"),
            )

            if proto_bank and query_sign_csv and gallery_sign_csv:
                try:
                    proto_types, proto_mat = load_prototypes(proto_bank)
                    query_sign_map = read_sign_types_from_csv(query_sign_csv)
                    gallery_sign_map = read_sign_types_from_csv(gallery_sign_csv)
                    query_dir = os.path.join(cfg.DATASETS.ROOT_DIR, "image_query")
                    gallery_dir = os.path.join(cfg.DATASETS.ROOT_DIR, "image_test")
                    q_keys = [_safe_rel_image_key(p, query_dir) for p in q_paths]
                    g_keys = [_safe_rel_image_key(p, gallery_dir) for p in g_paths]

                    qf_proto = qf
                    if args.proto_use_unfused and cfg.TEST.CLS_FUSION:
                        print("[proto] Using non-fused CLS features for prototype routing.")
                        proto_cfg = cfg.clone()
                        proto_cfg.defrost()
                        proto_cfg.TEST.CLS_FUSION = False
                        proto_cfg.TEST.CLS_FUSION_LAST = 1
                        proto_cfg.freeze()

                        proto_model = make_model(proto_cfg, proto_cfg.MODEL.NAME, 0, 0, 0)
                        proto_model.load_param(proto_cfg.TEST.WEIGHT)
                        proto_model.to(cfg.MODEL.DEVICE)
                        proto_model.eval()
                        with torch.no_grad():
                            qf_proto, _, _, _ = extract_feature(proto_model, val_loader, num_query)
                        qf_proto = qf_proto.cpu().numpy()
                        del proto_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    re_rank_dist = apply_prototype_filter(
                        re_rank_dist,
                        qf_proto,
                        q_keys,
                        g_keys,
                        query_sign_map,
                        gallery_sign_map,
                        proto_types,
                        proto_mat,
                        proto_margin=args.proto_margin,
                        penalty=0.05,
                        query_classes=query_classes,
                    )
                except Exception as e:
                    print(f"[proto] Skipping prototype filter: {e}")
            else:
                print("[proto] Prototype filter disabled (missing files).")


        if cfg.TEST.DO_CLASS_FILTER:
            re_rank_dist = apply_class_penalty(re_rank_dist, query_classes, gallery_classes)

        if cfg.TEST.QMV.ENABLED:
            query_image_paths = [item[0] for item in val_loader.dataset.img_items[:num_query]]
            score_mat = -re_rank_dist
            indices = apply_qmv(
                score_mat,
                query_image_paths,
                match_threshold=cfg.TEST.QMV.MATCH_THRESHOLD,
                top_k=cfg.TEST.QMV.TOP_K,
                decay=cfg.TEST.QMV.DECAY,
                orb_features=cfg.TEST.QMV.ORB_FEATURES,
            )[:, :100]
        else:
            indices = np.argsort(re_rank_dist, axis=1)[:, :100]

        m, n = indices.shape
        # # print('m: {}  n: {}'.format(m, n))
        with open(args.track, 'wb') as f_w:
            for i in range(m):
                write_line = indices[i] + 1
                write_line = ' '.join(map(str, write_line.tolist())) + '\n'
                f_w.write(write_line.encode())


        lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
        output_path = args.track.split(".txt")[0] + "_submission.csv"

        with open(output_path, 'w', newline='') as archivo_csv:
            csv_writter = csv.writer(archivo_csv)
            csv_writter.writerow(['imageName', 'Corresponding Indexes'])
            for numero, track in zip(lista_nombres, indices):
                track_str = ' '.join(map(str, track + 1))
                csv_writter.writerow([numero, track_str])
