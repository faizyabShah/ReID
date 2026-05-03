import argparse
import csv
import json
import os
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from data.build_DG_dataloader import fast_batch_collator
from data.common import CommDataset
from data.transforms.build import build_transforms
from model import make_model
from update import apply_caj, normalize_class_name
from utils.logger import setup_logger
from utils.re_ranking import re_ranking


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def find_column(fieldnames, candidates):
    lookup = {name.strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def read_split_csv(csv_path, include_class=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        image_col = find_column(reader.fieldnames, ["imageName", "imagename", "image_name"])
        pid_col = find_column(reader.fieldnames, ["objectID", "Corresponding Indexes", "object", "pid"])
        cam_col = find_column(reader.fieldnames, ["cameraID", "camid", "camera"])
        class_col = find_column(reader.fieldnames, ["Class", "class"])

        if image_col is None:
            raise ValueError(f"imageName column not found in {csv_path}")
        if pid_col is None:
            raise ValueError(f"objectID column not found in {csv_path}")
        if cam_col is None:
            raise ValueError(f"cameraID column not found in {csv_path}")
        if include_class and class_col is None:
            raise ValueError(f"Class column not found in {csv_path}")

        records = []
        for row in reader:
            record = {
                "image_name": row[image_col].strip(),
                "pid": int(row[pid_col]),
                "camid": int(str(row[cam_col]).strip().lstrip("cC")),
            }
            if include_class:
                record["class_name"] = normalize_class_name(row[class_col])
            records.append(record)
    return records


def index_by_image_name(records):
    return {record["image_name"]: record for record in records}


def resolve_image_path(root, subdir, image_name):
    candidates = [
        Path(root) / subdir / image_name,
        Path(root) / subdir / Path(image_name).name,
        Path(root) / image_name,
        Path(root) / Path(image_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_loader(cfg, items):
    transforms = build_transforms(cfg, is_train=False)
    dataset = CommDataset(items, transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    num_workers = cfg.DATALOADER.NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fast_batch_collator,
    )


def get_backbone(model):
    return model.module.base if isinstance(model, torch.nn.DataParallel) else model.base


def extract_layerwise_cls_tokens(model, loader, device):
    backbone = get_backbone(model)
    batch_tokens = []
    pids = []
    camids = []

    backbone.eval()
    for batch in loader:
        images = batch["images"].to(device)
        with torch.no_grad():
            try:
                layerwise_tokens = backbone(images, return_layerwise=True)
            except TypeError:
                # Some backbones return layerwise tokens by default and do not expose return_layerwise.
                layerwise_tokens = backbone(images)
            cls_tokens = torch.stack([layer[:, 0] for layer in layerwise_tokens], dim=1)
        batch_tokens.append(cls_tokens.cpu())
        pids.extend(np.asarray(batch["targets"]))
        camids.extend(np.asarray(batch["camid"]))

    return torch.cat(batch_tokens, dim=0), np.asarray(pids), np.asarray(camids)


def fuse_last_k_cls_tokens(layerwise_cls_tokens, last_k):
    last_k = max(1, min(int(last_k), layerwise_cls_tokens.shape[1]))
    fused = layerwise_cls_tokens[:, -last_k:, :].reshape(layerwise_cls_tokens.shape[0], -1)
    return torch.nn.functional.normalize(fused, dim=1, p=2)


def compute_similarity_mats(query_feat, gallery_feat):
    query_np = query_feat.cpu().numpy()
    gallery_np = gallery_feat.cpu().numpy()
    q_g_sim = np.dot(query_np, gallery_np.T)
    q_q_sim = np.dot(query_np, query_np.T)
    g_g_sim = np.dot(gallery_np, gallery_np.T)
    return q_g_sim, q_q_sim, g_g_sim


def evaluate_distmat(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    valid_queries = 0

    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        valid_queries += 1

        num_rel = orig_cmc.sum()
        tmp = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
        ap = (tmp * orig_cmc).sum() / num_rel
        all_ap.append(ap)

    if valid_queries == 0:
        raise RuntimeError("No valid query identities found in the gallery")

    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / valid_queries
    mAP = float(np.mean(all_ap))
    return all_cmc, mAP


def build_submission(distmat, query_names, gallery_names, topk=100):
    indices = np.argsort(distmat, axis=1)[:, :topk]
    rows = []
    for q_name, row in zip(query_names, indices):
        rows.append((q_name, " ".join(str(i + 1) for i in row.tolist())))
    return rows, indices


def normalize_class_list(class_list):
    return [normalize_class_name(cls) for cls in class_list]


def build_class_param_map(class_names, default_params):
    return {class_name: dict(default_params) for class_name in class_names}


def apply_classwise_reranking(q_g_sim, q_q_sim, g_g_sim, query_classes, gallery_classes,
                              class_param_map, default_params):
    base_dist = re_ranking(q_g_sim, q_q_sim, g_g_sim, **default_params)
    final_dist = base_dist.copy()

    unique_classes = sorted(set(query_classes) | set(gallery_classes))
    for class_name in unique_classes:
        q_idx = np.where(np.asarray(query_classes) == class_name)[0]
        g_idx = np.where(np.asarray(gallery_classes) == class_name)[0]
        if len(q_idx) == 0 or len(g_idx) == 0:
            continue

        params = dict(class_param_map.get(class_name, default_params))
        total_num = len(q_idx) + len(g_idx)
        if total_num < 2:
            continue

        params["k1"] = max(1, min(int(params["k1"]), total_num - 1))
        params["k2"] = max(1, min(int(params["k2"]), total_num))

        class_dist = re_ranking(
            q_g_sim[np.ix_(q_idx, g_idx)],
            q_q_sim[np.ix_(q_idx, q_idx)],
            g_g_sim[np.ix_(g_idx, g_idx)],
            **params,
        )
        final_dist[np.ix_(q_idx, g_idx)] = class_dist

    return final_dist


def better_score(candidate, best):
    if best is None:
        return True
    if candidate[1] != best[1]:
        return candidate[1] > best[1]
    return candidate[2] > best[2]


def search_last_k(layerwise_cls_tokens, q_count, q_pids, g_pids, k_grid, default_rerank_params):
    best = None
    for last_k in k_grid:
        fused = fuse_last_k_cls_tokens(layerwise_cls_tokens, last_k)
        query_feat = fused[:q_count]
        gallery_feat = fused[q_count:]
        q_g_sim, q_q_sim, g_g_sim = compute_similarity_mats(query_feat, gallery_feat)
        distmat = re_ranking(q_g_sim, q_q_sim, g_g_sim, **default_rerank_params)
        cmc, mAP = evaluate_distmat(distmat, q_pids, g_pids)
        candidate = {
            "last_k": int(last_k),
            "fused": fused,
            "q_g_sim": q_g_sim,
            "q_q_sim": q_q_sim,
            "g_g_sim": g_g_sim,
            "distmat": distmat,
            "cmc": cmc,
            "mAP": mAP,
        }
        if better_score((last_k, mAP, float(cmc[0])), None if best is None else (best["last_k"], best["mAP"], float(best["cmc"][0]))):
            best = candidate
    return best


def search_classwise_params(q_g_sim, q_q_sim, g_g_sim, query_classes, gallery_classes,
                            class_order, candidate_params, default_rerank_params,
                            q_pids, g_pids):
    class_param_map = build_class_param_map(class_order, candidate_params[0])
    best_dist = apply_classwise_reranking(
        q_g_sim,
        q_q_sim,
        g_g_sim,
        query_classes,
        gallery_classes,
        class_param_map,
        default_rerank_params,
    )
    best_cmc, best_map = evaluate_distmat(best_dist, q_pids, g_pids)

    for class_name in class_order:
        current_best_params = dict(class_param_map[class_name])
        current_best_score = (best_map, float(best_cmc[0]))

        for params in candidate_params:
            trial_map = deepcopy(class_param_map)
            trial_map[class_name] = dict(params)
            trial_dist = apply_classwise_reranking(
                q_g_sim,
                q_q_sim,
                g_g_sim,
                query_classes,
                gallery_classes,
                trial_map,
                default_rerank_params,
            )
            trial_cmc, trial_map_score = evaluate_distmat(trial_dist, q_pids, g_pids)
            trial_score = (trial_map_score, float(trial_cmc[0]))
            if trial_score > current_best_score:
                current_best_score = trial_score
                current_best_params = dict(params)
                best_dist = trial_dist
                best_cmc = trial_cmc
                best_map = trial_map_score

        class_param_map[class_name] = current_best_params

    return best_dist, class_param_map, best_cmc, best_map


def search_caj_params(base_dist, q_q_sim, g_g_sim, q_camids, g_camids, candidate_pairs, q_pids, g_pids):
    best = None
    for same_penalty, cross_scale in candidate_pairs:
        trial_dist, _, _ = apply_caj(
            base_dist,
            q_q_sim,
            g_g_sim,
            q_camids,
            g_camids,
            same_cam_penalty=same_penalty,
            cross_cam_scale=cross_scale,
        )
        trial_cmc, trial_map = evaluate_distmat(trial_dist, q_pids, g_pids)
        candidate = {
            "same_cam_penalty": float(same_penalty),
            "cross_cam_scale": float(cross_scale),
            "distmat": trial_dist,
            "cmc": trial_cmc,
            "mAP": trial_map,
        }
        if better_score((same_penalty, trial_map, float(trial_cmc[0])), None if best is None else (best["same_cam_penalty"], best["mAP"], float(best["cmc"][0]))):
            best = candidate
    return best


def save_submission_csv(output_path, rows):
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="UAM Unified grid-search evaluation")
    parser.add_argument("--config_file", default="./config/PAT.yml", type=str)
    parser.add_argument("--data_root", default="./UAM_Unified", type=str)
    parser.add_argument("--output", default="./uam_gridsearch_submission.csv", type=str)
    parser.add_argument("--summary", default="./uam_gridsearch_best.json", type=str)
    parser.add_argument("--k_grid", default="4,6,8,10,12", type=str)
    parser.add_argument("--class_k1_grid", default="14,18,20,24,26,28,30,32", type=str)
    parser.add_argument("--class_k2_grid", default="1,2,4,6,8", type=str)
    parser.add_argument("--class_lambda_grid", default="0.25,0.30,0.35,0.40,0.45,0.50", type=str)
    parser.add_argument("--caj_same_grid", default="1.00,1.05,1.10,1.15", type=str)
    parser.add_argument("--caj_cross_grid", default="0.80,0.85,0.90,0.95,1.00", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    cfg.defrost()
    cfg.TEST.CLS_FUSION = True
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME) if cfg.LOG_ROOT else ""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("UAM_GRIDSEARCH", output_dir, if_train=False)
    logger.info(args)
    logger.info("Running with config:\n%s", cfg)

    root_dir = Path(args.data_root)
    query_records = read_split_csv(root_dir / "query.csv", include_class=False)
    gallery_records = read_split_csv(root_dir / "test.csv", include_class=False)
    query_class_records = read_split_csv(root_dir / "query_classes.csv", include_class=True)
    gallery_class_records = read_split_csv(root_dir / "test_classes.csv", include_class=True)

    query_classes_by_name = index_by_image_name(query_class_records)
    gallery_classes_by_name = index_by_image_name(gallery_class_records)

    missing_query_classes = [record["image_name"] for record in query_records if record["image_name"] not in query_classes_by_name]
    missing_gallery_classes = [record["image_name"] for record in gallery_records if record["image_name"] not in gallery_classes_by_name]
    if missing_query_classes:
        raise ValueError(f"Missing query class labels for {len(missing_query_classes)} images, e.g. {missing_query_classes[:3]}")
    if missing_gallery_classes:
        raise ValueError(f"Missing gallery class labels for {len(missing_gallery_classes)} images, e.g. {missing_gallery_classes[:3]}")

    query_names = [record["image_name"] for record in query_records]
    gallery_names = [record["image_name"] for record in gallery_records]
    query_pids = np.asarray([record["pid"] for record in query_records])
    gallery_pids = np.asarray([record["pid"] for record in gallery_records])
    query_camids = np.asarray([record["camid"] for record in query_records])
    gallery_camids = np.asarray([record["camid"] for record in gallery_records])
    query_classes = normalize_class_list([query_classes_by_name[record["image_name"]]["class_name"] for record in query_records])
    gallery_classes = normalize_class_list([gallery_classes_by_name[record["image_name"]]["class_name"] for record in gallery_records])

    test_items = [
        (
            resolve_image_path(root_dir, "image_query", record["image_name"]),
            record["pid"],
            record["camid"],
        )
        for record in query_records
    ] + [
        (
            resolve_image_path(root_dir, "image_test", record["image_name"]),
            record["pid"],
            record["camid"],
        )
        for record in gallery_records
    ]
    test_loader = build_loader(cfg, test_items)
    gallery_offset = len(query_records)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    if torch.cuda.device_count() > 1 and device == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    with torch.no_grad():
        layerwise_cls_tokens, pids, camids = extract_layerwise_cls_tokens(model, test_loader, device)

    if not np.array_equal(pids[:gallery_offset], query_pids):
        logger.warning("Loaded query PIDs do not match query.csv order exactly; using CSV order for scoring.")

    k_grid = parse_int_list(args.k_grid)
    class_k1_grid = parse_int_list(args.class_k1_grid)
    class_k2_grid = parse_int_list(args.class_k2_grid)
    class_lambda_grid = parse_float_list(args.class_lambda_grid)
    caj_same_grid = parse_float_list(args.caj_same_grid)
    caj_cross_grid = parse_float_list(args.caj_cross_grid)

    default_rerank_params = {"k1": 20, "k2": 6, "lambda_value": 0.3}
    logger.info("Stage 1: searching CLS fusion last-k over %s", k_grid)
    best_k = search_last_k(
        layerwise_cls_tokens,
        gallery_offset,
        query_pids,
        gallery_pids,
        k_grid,
        default_rerank_params,
    )

    logger.info("Best CLS fusion last-k: %s | mAP %.6f | Rank-1 %.6f", best_k["last_k"], best_k["mAP"], float(best_k["cmc"][0]))

    class_candidates = [
        {"k1": k1, "k2": k2, "lambda_value": lam}
        for k1, k2, lam in product(class_k1_grid, class_k2_grid, class_lambda_grid)
    ]
    class_order = sorted(set(query_classes) | set(gallery_classes))
    logger.info("Stage 2: searching classwise reranking params over %d candidates per class", len(class_candidates))
    best_class_dist, best_class_map, best_class_cmc, best_class_map_score = search_classwise_params(
        best_k["q_g_sim"],
        best_k["q_q_sim"],
        best_k["g_g_sim"],
        query_classes,
        gallery_classes,
        class_order,
        class_candidates,
        default_rerank_params,
        query_pids,
        gallery_pids,
    )
    logger.info("Best classwise rerank mAP %.6f | Rank-1 %.6f", best_class_map_score, float(best_class_cmc[0]))

    caj_candidates = list(product(caj_same_grid, caj_cross_grid))
    logger.info("Stage 3: searching CAJ params over %d combinations", len(caj_candidates))
    best_caj = search_caj_params(
        best_class_dist,
        best_k["q_q_sim"],
        best_k["g_g_sim"],
        query_camids,
        gallery_camids,
        caj_candidates,
        query_pids,
        gallery_pids,
    )
    logger.info(
        "Best CAJ same_penalty %.3f cross_scale %.3f | mAP %.6f | Rank-1 %.6f",
        best_caj["same_cam_penalty"],
        best_caj["cross_cam_scale"],
        best_caj["mAP"],
        float(best_caj["cmc"][0]),
    )

    submission_rows, indices = build_submission(best_caj["distmat"], query_names, gallery_names, topk=100)
    save_submission_csv(args.output, submission_rows)

    summary = {
        "best_last_k": best_k["last_k"],
        "best_last_k_mAP": best_k["mAP"],
        "best_last_k_rank1": float(best_k["cmc"][0]),
        "best_classwise_params": best_class_map,
        "best_classwise_mAP": best_class_map_score,
        "best_classwise_rank1": float(best_class_cmc[0]),
        "best_caj_same_cam_penalty": best_caj["same_cam_penalty"],
        "best_caj_cross_cam_scale": best_caj["cross_cam_scale"],
        "best_caj_mAP": best_caj["mAP"],
        "best_caj_rank1": float(best_caj["cmc"][0]),
        "output_csv": args.output,
    }

    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("=" * 40)
    print("BEST CONFIG")
    print("=" * 40)
    print(f"CLS_FUSION_LAST: {best_k['last_k']}")
    print(f"Classwise params: {json.dumps(best_class_map, sort_keys=True)}")
    print(f"CAJ: same_cam_penalty={best_caj['same_cam_penalty']:.3f}, cross_cam_scale={best_caj['cross_cam_scale']:.3f}")
    print(f"mAP: {best_caj['mAP']:.6f}")
    print(f"Rank-1: {float(best_caj['cmc'][0]):.6f}")
    print(f"Saved submission: {args.output}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()