import os
import csv
import torch
import argparse
import torch.nn as nn

import numpy as np
from torchvision import models as tv_models
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat

#from torch.backends import cudnn

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

def normalize_class_name(class_name):
    """Normalize class names so dataset-specific aliases share one config key."""
    normalized = str(class_name).strip().lower().replace(" ", "").replace("_", "")
    if normalized in {"trafficsign", "trafficsignal", "signal", "sign"}:
        return "traffic"
    return normalized

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

def extract_feature(model, dataloaders, num_query, classifier_model=None,
                    classifier_temperature=1.0, semantic_classes=None):
    """Extract features and collect camera IDs for CAJ adjustment."""
    features = []
    camids = []
    img_path = []
    class_probs = []
    traffic_mask = []
    normalized_classes = None
    if semantic_classes is not None:
        normalized_classes = [normalize_class_name(cls) for cls in semantic_classes]

    sample_offset = 0

    for data in dataloaders:
        img = data['images']
        camid = data['camid']
        batch_paths = data['img_path']
        #obtain values form dict data
        n, c, h, w = img.size()
        # ff = torch.FloatTensor(n, 1024).zero_().cuda()  # 2048 is pool5 of resnet
        for i in range(2):
            input_img = img.cuda()
            if i == 1:
                input_img = torch.flip(input_img, dims=[-1])
            outputs = model(input_img)
            f = outputs.float()
            if i == 0:
                ff = torch.zeros(n, f.shape[1], device=f.device)
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
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
                traffic_imgs = img[traffic_mask_tensor]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./config/PAT.yml", help="path to config file", type=str
    )
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

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(cfg.TEST.WEIGHT)

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
        qf, gf, q_camids, g_camids, q_img_paths, g_img_paths, q_class_probs, g_class_probs, q_traffic_mask, g_traffic_mask = extract_feature(
            model,
            val_loader,
            num_query,
            classifier_model=traffic_sign_classifier,
            classifier_temperature=cfg.TEST.TRAFFIC_SIGN_CLASSIFIER.TEMPERATURE,
            semantic_classes=(query_classes + gallery_classes) if query_classes is not None and gallery_classes is not None else None,
        )

    # save feature
    qf=qf.cpu().numpy()
    gf=gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

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
