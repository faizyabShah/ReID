import numpy as np
from utils.re_ranking import re_ranking
from evaluate_csv import evaluate_from_indices
import csv

# ---------------- CONFIG ----------------
query_csv = "/kaggle/working/ReID/UAM_Unified/query.csv"
gallery_csv = "/kaggle/working/ReID/UAM_Unified/test.csv"

use_class_penalty = True
query_classes_csv = "/kaggle/working/ReID/UAM_Unified/query_classes.csv"
gallery_classes_csv = "/kaggle/working/ReID/UAM_Unified/test_classes.csv"

# ---------------- LOAD FEATURES ----------------
qf = np.load("qf.npy")
gf = np.load("gf.npy")

# ---------------- PRECOMPUTE ----------------
q_g_dist = np.dot(qf, gf.T)
q_q_dist = np.dot(qf, qf.T)
g_g_dist = np.dot(gf, gf.T)

# ---------------- CLASS PENALTY ----------------
def read_classes(csv_path):
    classes = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            classes.append(row[3])
    return classes

def apply_class_penalty(dist_mat, query_classes, gallery_classes):
    q_classes = np.array(query_classes)[:, None]
    g_classes = np.array(gallery_classes)[None, :]
    mismatch = (q_classes != g_classes)

    penalized = dist_mat.copy()
    penalized[mismatch] += 1e6
    return penalized

if use_class_penalty:
    query_classes = read_classes(query_classes_csv)
    gallery_classes = read_classes(gallery_classes_csv)

# ---------------- GRID ----------------
k1_list = [10, 15, 20, 25, 30]
k2_list = [3, 6, 9]
lambda_list = [0.1, 0.3, 0.5, 0.7]

best_mAP = 0
best_params = None

print("Starting grid search...\n")

for k1 in k1_list:
    for k2 in k2_list:
        for lmb in lambda_list:

            print(f"Testing k1={k1}, k2={k2}, lambda={lmb}")

            dist = re_ranking(
                q_g_dist,
                q_q_dist,
                g_g_dist,
                k1=k1,
                k2=k2,
                lambda_value=lmb
            )

            if use_class_penalty:
                dist = apply_class_penalty(dist, query_classes, gallery_classes)

            indices = np.argsort(dist, axis=1)[:, :100]

            mAP, cmc = evaluate_from_indices(indices, query_csv, gallery_csv)

            print(f"mAP={mAP:.4f}, Rank-1={cmc[0]:.4f}\n")

            if mAP > best_mAP:
                best_mAP = mAP
                best_params = (k1, k2, lmb)

print("=" * 40)
print("BEST RESULT")
print("=" * 40)
print(f"Best params: k1={best_params[0]}, k2={best_params[1]}, lambda={best_params[2]}")
print(f"Best mAP: {best_mAP:.4f}")
print("=" * 40)