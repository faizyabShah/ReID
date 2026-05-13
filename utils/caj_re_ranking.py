"""
Camera-Aware Jaccard (CAJ) Re-Ranking

Based on:
- Zhong et al., "Re-ranking Person Re-identification with k-Reciprocal Encoding", CVPR 2017
- Chen et al., "CA-Jaccard: Camera-Aware Jaccard Distance for Person Re-identification", CVPR 2024
- Specker, "Intermediate Features and View-Specific Embeddings...", ICIPW 2025
  (k1_intra=3, k1_inter=30, k2=5, lambda=0.1)

Key difference from standard k-reciprocal:
- Standard: uses one k1 for all neighbor searches
- CAJ: uses k1_intra for same-camera pairs, k1_inter for cross-camera pairs
  This compensates for the different distance scales between intra/inter camera pairs.

Usage:
    from utils.caj_re_ranking import caj_re_ranking_fast

    dist = caj_re_ranking_fast(
        q_g_sim, q_q_sim, g_g_sim,
        q_camids, g_camids,
        k1_intra=3, k1_inter=30, k2=5, lambda_value=0.1
    )
"""

import numpy as np


def k_reciprocal_neigh(initial_rank, i, k1):
    """Find k-reciprocal nearest neighbors."""
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def k_reciprocal_neigh_cam_aware(initial_rank, i, k1_intra, k1_inter, camids_all):
    """
    Camera-aware k-reciprocal nearest neighbors.

    For each candidate neighbor j of query i:
    - If same camera: use k1_intra (smaller neighborhood)
    - If different camera: use k1_inter (larger neighborhood)

    This means we first find the k1_inter nearest neighbors (the larger set),
    then for each candidate, check reciprocity using the appropriate k1.
    """
    cam_i = camids_all[i]

    # Forward: take the larger k1 to get all potential neighbors
    k1_max = max(k1_intra, k1_inter)
    forward_k_neigh = initial_rank[i, :k1_max + 1]

    # For each forward neighbor, check if i is in THEIR k-reciprocal set
    # using the camera-appropriate k1
    reciprocal = []
    for j in forward_k_neigh:
        cam_j = camids_all[j]

        # Determine which k1 to use for the (i, j) pair
        if cam_i == cam_j:
            k1_pair = k1_intra
        else:
            k1_pair = k1_inter

        # Check: was j in i's top-k1_pair?
        if j not in initial_rank[i, :k1_pair + 1]:
            continue

        # Check: is i in j's top-k1_pair? (reciprocity)
        if i in initial_rank[j, :k1_pair + 1]:
            reciprocal.append(j)

    return np.array(reciprocal, dtype=np.int32)


def caj_re_ranking(q_g_sim, q_q_sim, g_g_sim, q_camids, g_camids,
                   k1_intra=3, k1_inter=30, k2=5, lambda_value=0.1):
    """
    Camera-Aware Jaccard re-ranking.

    Args:
        q_g_sim: cosine similarity, query vs gallery [num_q, num_g]
        q_q_sim: cosine similarity, query vs query [num_q, num_q]
        g_g_sim: cosine similarity, gallery vs gallery [num_g, num_g]
        q_camids: camera IDs for queries [num_q]
        g_camids: camera IDs for gallery [num_g]
        k1_intra: k1 for same-camera neighbor search (small, e.g. 3)
        k1_inter: k1 for cross-camera neighbor search (large, e.g. 30)
        k2: expansion parameter
        lambda_value: weight for original distance (e.g. 0.1)

    Returns:
        final_dist: re-ranked distance matrix [num_q, num_g]
    """
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = q_g_sim.shape[0]
    num_g = q_g_sim.shape[1]
    num_all = num_q + num_g

    # Concatenate all camera IDs
    camids_all = np.concatenate([q_camids, g_camids])

    # Build full similarity matrix
    # Layout: [query | gallery] x [query | gallery]
    all_sim = np.zeros((num_all, num_all), dtype=np.float32)
    all_sim[:num_q, :num_q] = q_q_sim
    all_sim[:num_q, num_q:] = q_g_sim
    all_sim[num_q:, :num_q] = q_g_sim.T
    all_sim[num_q:, num_q:] = g_g_sim

    # Convert similarity to distance
    original_dist = 1.0 - all_sim

    # Initial ranking (ascending distance = descending similarity)
    initial_rank = np.argsort(original_dist, axis=1)

    # Compute Jaccard distance with camera-aware k-reciprocal neighbors
    k1_max = max(k1_intra, k1_inter)

    # For each sample, compute its k-reciprocal neighbor set
    all_neighbor_sets = []
    for i in range(num_all):
        k_reciprocal = k_reciprocal_neigh_cam_aware(
            initial_rank, i, k1_intra, k1_inter, camids_all
        )

        # k2 expansion: for each reciprocal neighbor, check THEIR neighbors
        # and add them if they share enough overlap (standard k-reciprocal logic)
        if k2 > 0 and len(k_reciprocal) > 0:
            expanded = set(k_reciprocal.tolist())
            for j in k_reciprocal:
                # Use standard k1_max for expansion step
                j_reciprocal = k_reciprocal_neigh(initial_rank, j, k1_max)
                if len(j_reciprocal) > 0:
                    # Check overlap: if 2/3 of j's reciprocal neighbors are in i's set
                    overlap = len(set(j_reciprocal.tolist()) & set(k_reciprocal.tolist()))
                    if overlap >= 2.0 / 3.0 * len(j_reciprocal):
                        for idx in j_reciprocal:
                            expanded.add(idx)
            k_reciprocal = np.array(sorted(expanded), dtype=np.int32)

        all_neighbor_sets.append(set(k_reciprocal.tolist()))

    # Compute Jaccard distance between each query and gallery item
    jaccard_dist = np.ones((num_q, num_g), dtype=np.float32)

    for i in range(num_q):
        set_i = all_neighbor_sets[i]
        if len(set_i) == 0:
            continue
        for j in range(num_g):
            set_j = all_neighbor_sets[num_q + j]
            if len(set_j) == 0:
                continue
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            if union > 0:
                jaccard_dist[i, j] = 1.0 - float(intersection) / float(union)

    # Final distance: combine original and Jaccard
    original_q_g_dist = 1.0 - q_g_sim
    final_dist = (1.0 - lambda_value) * jaccard_dist + lambda_value * original_q_g_dist

    return final_dist


def caj_re_ranking_fast(q_g_sim, q_q_sim, g_g_sim, q_camids, g_camids,
                        k1_intra=3, k1_inter=30, k2=5, lambda_value=0.1):
    """
    Faster version using sparse vector encoding instead of set operations.
    Better for large galleries. Same result as caj_re_ranking.
    """
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = q_g_sim.shape[0]
    num_g = q_g_sim.shape[1]
    num_all = num_q + num_g

    camids_all = np.concatenate([q_camids, g_camids])

    # Build full similarity and distance matrices
    all_sim = np.zeros((num_all, num_all), dtype=np.float32)
    all_sim[:num_q, :num_q] = q_q_sim
    all_sim[:num_q, num_q:] = q_g_sim
    all_sim[num_q:, :num_q] = q_g_sim.T
    all_sim[num_q:, num_q:] = g_g_sim

    original_dist = 1.0 - all_sim
    initial_rank = np.argsort(original_dist, axis=1)

    # Encode each sample's neighborhood as a sparse binary vector
    # Then Jaccard = 1 - (dot product) / (sum_i + sum_j - dot product)
    V = np.zeros((num_all, num_all), dtype=np.float32)

    for i in range(num_all):
        k_reciprocal = k_reciprocal_neigh_cam_aware(
            initial_rank, i, k1_intra, k1_inter, camids_all
        )

        # k2 expansion
        if k2 > 0 and len(k_reciprocal) > 0:
            expanded = list(k_reciprocal)
            for j in k_reciprocal:
                k1_max = max(k1_intra, k1_inter)
                j_reciprocal = k_reciprocal_neigh(initial_rank, j, k1_max)
                if len(j_reciprocal) > 0:
                    overlap = np.intersect1d(j_reciprocal, k_reciprocal)
                    if len(overlap) >= 2.0 / 3.0 * len(j_reciprocal):
                        expanded = np.union1d(expanded, j_reciprocal)
            k_reciprocal = np.array(expanded, dtype=np.int32)

        # Gaussian-weighted encoding
        if len(k_reciprocal) > 0:
            weight = np.exp(-original_dist[i, k_reciprocal])
            weight = weight / np.sum(weight)
            V[i, k_reciprocal] = weight

    # Local query expansion with k2
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(num_all):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    # Compute Jaccard distance for query-gallery pairs
    jaccard_dist = np.zeros((num_q, num_g), dtype=np.float32)

    for i in range(num_q):
        vi = V[i]
        vi_sum = np.sum(vi)
        if vi_sum == 0:
            jaccard_dist[i, :] = 1.0
            continue
        for j in range(num_g):
            vj = V[num_q + j]
            vj_sum = np.sum(vj)
            if vj_sum == 0:
                jaccard_dist[i, j] = 1.0
                continue
            dot = np.dot(vi, vj)
            jaccard_dist[i, j] = 1.0 - dot / (vi_sum + vj_sum - dot + 1e-10)

    # Combine
    original_q_g_dist = 1.0 - q_g_sim
    final_dist = (1.0 - lambda_value) * jaccard_dist + lambda_value * original_q_g_dist

    return final_dist
