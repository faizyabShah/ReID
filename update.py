import os
import csv
import torch
import argparse
import pandas as pd

import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat

#from torch.backends import cudnn

def read_classes_from_csv(csv_path):
    """Reads the CSV and returns a list of classes in order."""
    classes = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "Class" in reader.fieldnames:
            for row in reader:
                classes.append(row.get("Class"))
        else:
            f.seek(0)
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                classes.append(row[2] if len(row) > 2 else None)
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
    """Normalize class names so dataset-specific aliases share one config key."""
    normalized = str(class_name).strip().lower().replace(" ", "").replace("_", "")
    if normalized in {"trafficsign", "trafficsignal", "signal", "sign"}:
        return "traffic"
    return normalized

def read_sign_types_from_csv(csv_path):
    """Return {imageName: (sign_type, viewpoint)}."""
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

def extract_feature(model, dataloaders, num_query):
    """Extract features and collect camera IDs for CAJ adjustment."""
    features = []
    camids = []
    count = 0
    img_path = []

    for data in dataloaders:
        img = data['images']
        camid = data['camid']
        img_path.extend(data.get("img_path", []))
        #obtain values form dict data
        n, c, h, w = img.size()
        count += n
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
    features = torch.cat(features, 0)
    camids = np.array(camids)

    # query
    qf = features[:num_query]
    q_camids = camids[:num_query]
    q_paths = img_path[:num_query]
    # gallery
    gf = features[num_query:]
    g_camids = camids[num_query:]
    g_paths = img_path[num_query:]
    return qf, gf, q_camids, g_camids, q_paths, g_paths

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

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
    with torch.no_grad():
        qf, gf, q_camids, g_camids, q_paths, g_paths = extract_feature(model, val_loader, num_query)

    # save feature
    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

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

    # Prototype-based filtering for traffic signs (optional)
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
                    qf_proto, _, _, _, _, _ = extract_feature(proto_model, val_loader, num_query)
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
