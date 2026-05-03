import os
import cv2
import csv
import torch
import argparse

import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from itertools import combinations
from collections import defaultdict
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat

#from torch.backends import cudnn

def read_classes_from_csv(csv_path):
    """Reads the CSV and returns a list of classes in order."""
    classes = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header (cameraID, imageName, Class)
        for row in reader:
            classes.append(row[2])  # The Class is the 3rd column
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

def find_adjacent_query_pairs(query_image_paths, match_threshold=50, orb_features=500):
    """
    Uses ORB feature matching to find pairs of query images
    that are likely adjacent frames of the same object.
    Returns list of (idx_a, idx_b) pairs.
    """
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

def extract_feature(model, dataloaders, num_query):
    features = []
    count = 0
    img_path = []

    for data in dataloaders:
        img, a, b,_,_ = data.values()
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
    features = torch.cat(features, 0)

    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    return qf, gf

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
    parser.add_argument("--enable_qmv", action="store_true", help="enable Query Majority Voting")
    parser.add_argument("--qmv_match_threshold", type=int, default=50, help="ORB match threshold")
    parser.add_argument("--qmv_top_k", type=int, default=100, help="top-k for QMV voting")
    parser.add_argument("--qmv_decay", type=float, default=10.0, help="exponential decay in QMV")
    parser.add_argument("--qmv_orb_features", type=int, default=500, help="ORB features for QMV")
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
    query_image_paths = [item[0] for item in val_loader.dataset.img_items[:num_query]]
    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    # save feature
    qf=qf.cpu().numpy()
    gf=gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    query_csv_path = "./Urban2026/query_classes.csv"
    gallery_csv_path = "./Urban2026/test_classes.csv"
    
    query_classes = read_classes_from_csv(query_csv_path)
    gallery_classes = read_classes_from_csv(gallery_csv_path)

    re_rank_dist = apply_class_penalty(re_rank_dist, query_classes, gallery_classes)

    score_mat = -re_rank_dist
    if args.enable_qmv:
        indices = apply_qmv(
            score_mat,
            query_image_paths,
            match_threshold=args.qmv_match_threshold,
            top_k=args.qmv_top_k,
            decay=args.qmv_decay,
            orb_features=args.qmv_orb_features,
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
