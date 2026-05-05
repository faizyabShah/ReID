#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IMAGE_COL_CANDIDATES = ("imageName", "imagename", "image_name")
PID_COL_CANDIDATES = ("objectID", "Corresponding Indexes", "object", "pid")
CAM_COL_CANDIDATES = ("cameraID", "camid", "camera")
CLASS_COL_CANDIDATES = ("Class", "class")
PRED_COL_CANDIDATES = ("Corresponding Indexes", "predictions")


def normalize_class_name(class_name: object) -> str:
    normalized = str(class_name).strip().lower().replace(" ", "").replace("_", "")
    if normalized in {"trafficsign", "trafficsignal", "signal", "sign"}:
        return "traffic"
    return normalized


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(name).strip(): str(name).strip() for name in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def image_sort_key(image_name: str) -> tuple[int, object, str]:
    stem = Path(str(image_name)).stem
    try:
        return (0, int(stem), str(image_name))
    except ValueError:
        return (1, stem, str(image_name))


def load_base_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [str(col).strip() for col in df.columns]

    image_col = find_column(df.columns, IMAGE_COL_CANDIDATES)
    pid_col = find_column(df.columns, PID_COL_CANDIDATES)
    cam_col = find_column(df.columns, CAM_COL_CANDIDATES)

    if image_col is None:
        raise ValueError(f"imageName column not found in {csv_path}")
    if pid_col is None:
        raise ValueError(f"objectID column not found in {csv_path}")
    if cam_col is None:
        raise ValueError(f"cameraID column not found in {csv_path}")

    base = pd.DataFrame()
    base["imageName"] = df[image_col].astype(str).str.strip()
    base["objectID"] = pd.to_numeric(df[pid_col], errors="raise").astype(int)
    base["cameraID"] = df[cam_col].astype(str).str.strip()
    return base


def load_class_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [str(col).strip() for col in df.columns]

    image_col = find_column(df.columns, IMAGE_COL_CANDIDATES)
    pid_col = find_column(df.columns, PID_COL_CANDIDATES)
    cam_col = find_column(df.columns, CAM_COL_CANDIDATES)
    class_col = find_column(df.columns, CLASS_COL_CANDIDATES)

    if image_col is None:
        raise ValueError(f"imageName column not found in {csv_path}")
    if pid_col is None:
        raise ValueError(f"objectID column not found in {csv_path}")
    if cam_col is None:
        raise ValueError(f"cameraID column not found in {csv_path}")
    if class_col is None:
        raise ValueError(f"Class column not found in {csv_path}")

    classes = pd.DataFrame()
    classes["imageName"] = df[image_col].astype(str).str.strip()
    classes["objectID"] = pd.to_numeric(df[pid_col], errors="raise").astype(int)
    classes["cameraID"] = df[cam_col].astype(str).str.strip()
    classes["Class"] = df[class_col].astype(str).map(normalize_class_name)
    return classes


def load_labeled_split(split_csv: Path, class_csv: Path) -> pd.DataFrame:
    base = load_base_split(split_csv)
    classes = load_class_split(class_csv)

    merged = base.merge(classes[["imageName", "Class"]], on="imageName", how="left", validate="one_to_one")
    if merged["Class"].isna().any():
        missing = merged.loc[merged["Class"].isna(), "imageName"].tolist()
        raise ValueError(f"Missing class labels in {class_csv} for {len(missing)} images, e.g. {missing[:3]}")

    merged["Class"] = merged["Class"].map(normalize_class_name)

    check = base.merge(classes, on="imageName", suffixes=("_base", "_class"), validate="one_to_one")
    mismatched_pid = check.loc[check["objectID_base"] != check["objectID_class"], "imageName"].tolist()
    mismatched_cam = check.loc[check["cameraID_base"] != check["cameraID_class"], "imageName"].tolist()
    if mismatched_pid or mismatched_cam:
        issues = []
        if mismatched_pid:
            issues.append(f"{len(mismatched_pid)} objectID mismatches")
        if mismatched_cam:
            issues.append(f"{len(mismatched_cam)} cameraID mismatches")
        raise ValueError(f"Split and class CSVs disagree in {class_csv}: {', '.join(issues)}")

    return merged


def load_submission(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing submission file: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [str(col).strip() for col in df.columns]

    image_col = find_column(df.columns, IMAGE_COL_CANDIDATES)
    pred_col = find_column(df.columns, PRED_COL_CANDIDATES)
    if image_col is None:
        raise ValueError(f"imageName column not found in {csv_path}")
    if pred_col is None:
        raise ValueError(f"Corresponding Indexes column not found in {csv_path}")

    submission = pd.DataFrame()
    submission["imageName"] = df[image_col].astype(str).str.strip()
    submission["predictions_raw"] = df[pred_col].fillna("").astype(str).str.strip()
    return submission


def parse_rank_list(raw_value: str) -> list[int]:
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip()
    if not raw_text:
        return []
    values = []
    for item in raw_text.split():
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def build_gallery_lookup(gallery_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_gallery = gallery_df.copy()
    sorted_gallery["_sort_key"] = sorted_gallery["imageName"].map(image_sort_key)
    sorted_gallery = sorted_gallery.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    sorted_gallery["gallery_rank"] = np.arange(1, len(sorted_gallery) + 1)
    lookup = sorted_gallery.set_index("gallery_rank")
    return sorted_gallery, lookup


def compute_ap(retrieved_matches: np.ndarray, relevant_total: int) -> float:
    if relevant_total <= 0:
        return np.nan
    hits = np.flatnonzero(retrieved_matches)
    if len(hits) == 0:
        return 0.0
    precision_at_hits = np.cumsum(retrieved_matches)[hits] / (hits + 1)
    return float(precision_at_hits.sum() / relevant_total)


def hit_at_k(retrieved_matches: np.ndarray, k: int) -> float:
    if len(retrieved_matches) == 0:
        return 0.0
    return float(np.any(retrieved_matches[: min(k, len(retrieved_matches))]))


def first_hit_rank(retrieved_matches: np.ndarray) -> float:
    hits = np.flatnonzero(retrieved_matches)
    if len(hits) == 0:
        return np.nan
    return float(hits[0] + 1)


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    subset = df.loc[:, columns].copy()
    if max_rows is not None:
        subset = subset.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            values.append(format_float(row[column]).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_groups(per_query: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = per_query.groupby(group_col, dropna=False)
    summary = grouped.agg(
        query_count=("imageName", "count"),
        valid_query_count=("valid_for_id_metrics", "sum"),
        id_map=("id_ap", "mean"),
        id_rank1=("id_hit@1", "mean"),
        id_rank5=("id_hit@5", "mean"),
        id_rank10=("id_hit@10", "mean"),
        id_rank100=("id_hit@100", "mean"),
        class_map=("class_ap", "mean"),
        class_rank1=("class_hit@1", "mean"),
        class_rank5=("class_hit@5", "mean"),
        class_rank10=("class_hit@10", "mean"),
        class_rank100=("class_hit@100", "mean"),
        mean_first_id_rank=("first_id_rank", "mean"),
        mean_first_class_rank=("first_class_rank", "mean"),
        same_class_top1_wrong_rate=("same_class_top1_wrong", "mean"),
        top1_correct_id_rate=("id_hit@1", "mean"),
    ).reset_index()
    return summary


def build_confusion_table(per_query: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    wrong_top1 = per_query.loc[
        (per_query["valid_for_id_metrics"]) & (~per_query["top1_correct_id"]) & (per_query["top1_pred_class"].notna()),
        ["query_class", "top1_pred_class"],
    ]
    if wrong_top1.empty:
        return pd.DataFrame(columns=["query_class", "top1_pred_class", "count", "rate_within_query_class"])

    counts = wrong_top1.groupby(["query_class", "top1_pred_class"]).size().reset_index(name="count")
    totals = wrong_top1.groupby("query_class").size().reset_index(name="query_class_error_count")
    counts = counts.merge(totals, on="query_class", how="left")
    counts["rate_within_query_class"] = counts["count"] / counts["query_class_error_count"]
    counts = counts.sort_values(["count", "rate_within_query_class"], ascending=[False, False]).head(top_n)
    return counts.drop(columns=["query_class_error_count"])


def make_report_text(
    submission_path: Path,
    data_root: Path,
    per_query: pd.DataFrame,
    id_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    confusion_df: pd.DataFrame,
    validation: dict,
) -> str:
    valid_query_df = per_query.loc[per_query["valid_for_id_metrics"]].copy()
    if valid_query_df.empty:
        raise RuntimeError("No valid queries were found in the submission/data combination.")

    overall_id_map = float(valid_query_df["id_ap"].mean())
    overall_id_rank1 = float(valid_query_df["id_hit@1"].mean())
    overall_id_rank5 = float(valid_query_df["id_hit@5"].mean())
    overall_id_rank10 = float(valid_query_df["id_hit@10"].mean())
    overall_class_map = float(valid_query_df["class_ap"].mean())
    overall_class_rank1 = float(valid_query_df["class_hit@1"].mean())
    overall_class_rank5 = float(valid_query_df["class_hit@5"].mean())
    overall_class_rank10 = float(valid_query_df["class_hit@10"].mean())
    exact_id_perfect = all(
        np.isclose(value, 1.0)
        for value in (overall_id_map, overall_id_rank1, overall_id_rank5, overall_id_rank10)
    )
    same_class_wrong_rate = float(valid_query_df.loc[~valid_query_df["top1_correct_id"], "same_class_top1_wrong"].mean()) if (~valid_query_df["top1_correct_id"]).any() else 0.0
    wrong_top1_rate = float((~valid_query_df["top1_correct_id"]).mean())
    top1_class_only_rate = float(valid_query_df["top1_correct_class"].mean())
    recovered_in_top10_rate = float(valid_query_df["id_hit@10"].mean())
    mean_first_id_rank = float(valid_query_df["first_id_rank"].mean())
    mean_first_class_rank = float(valid_query_df["first_class_rank"].mean())

    worst_queries = valid_query_df.sort_values(["id_ap", "id_hit@1", "first_id_rank"], ascending=[True, True, False]).head(15)
    worst_ids = id_summary.sort_values(["id_map", "id_rank1", "query_count"], ascending=[True, True, False]).head(15)
    worst_classes = class_summary.sort_values(["id_map", "id_rank1", "query_count"], ascending=[True, True, False]).head(15)

    dominant_issue = "identity discrimination inside the same class"
    if exact_id_perfect:
        dominant_issue = "no exact-ID failures on the labeled split"
    elif overall_class_map - overall_id_map < 0.05 and wrong_top1_rate > 0.5:
        dominant_issue = "rank ordering on already-near neighbours"
    elif overall_class_map - overall_id_map >= 0.15:
        dominant_issue = "identity discrimination inside the same class"
    elif top1_class_only_rate < 0.5:
        dominant_issue = "class separation as well as identity matching"

    lines: list[str] = []
    lines.append("# UAM submission analysis")
    lines.append("")
    lines.append(f"Submission: [{submission_path.name}]({submission_path})")
    lines.append(f"Data root: [{data_root}]({data_root})")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"The submission contains {validation['submission_rows']} rows for {validation['query_rows']} queries, with {validation['missing_queries']} missing query rows and {validation['extra_submission_rows']} extra submission rows."
    )
    lines.append(
        f"Exact-ID retrieval is at mAP {overall_id_map:.4f}, Rank-1 {overall_id_rank1:.4f}, Rank-5 {overall_id_rank5:.4f}, and Rank-10 {overall_id_rank10:.4f}."
    )
    lines.append(
        f"Class-level retrieval is stronger at mAP {overall_class_map:.4f}, Rank-1 {overall_class_rank1:.4f}, Rank-5 {overall_class_rank5:.4f}, and Rank-10 {overall_class_rank10:.4f}."
    )
    lines.append(
        f"Among valid queries, {top1_class_only_rate:.4%} have the correct class at rank-1, while only {overall_id_rank1:.4%} have the correct ID at rank-1. The average first exact-ID hit is at rank {mean_first_id_rank:.2f}, compared with rank {mean_first_class_rank:.2f} for the first correct class hit."
    )
    lines.append(
        f"The dominant failure mode appears to be {dominant_issue}; among rank-1 ID errors, {same_class_wrong_rate:.4%} are still the correct class but the wrong identity."
    )
    if exact_id_perfect:
        lines.append("Every valid query retrieves the correct identity at Rank-1, so there are no exact-ID failure cases to rank on this split.")
    lines.append("")
    lines.append("## Integrity checks")
    lines.append("")
    lines.append(f"Gallery order matches sorted image-name order: {validation['gallery_order_matches_sorted']}")
    lines.append(f"Duplicate submission rows: {validation['duplicate_submission_rows']}")
    lines.append(f"Duplicate prediction indices in a row: {validation['rows_with_duplicate_indices']}")
    lines.append(f"Rows with out-of-range indices: {validation['rows_with_out_of_range_indices']}")
    lines.append(f"Queries with no relevant gallery ID in the labeled gallery: {validation['queries_without_gallery_match']}")
    lines.append("")
    lines.append("## Worst queries")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            worst_queries,
            [
                "imageName",
                "query_objectID",
                "query_class",
                "id_ap",
                "id_hit@1",
                "id_hit@5",
                "id_hit@10",
                "first_id_rank",
                "class_ap",
                "class_hit@1",
                "top1_pred_image",
                "top1_pred_objectID",
                "top1_pred_class",
            ],
            max_rows=15,
        )
    )
    lines.append("")
    lines.append("## Worst IDs")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            worst_ids,
            [
                "query_objectID",
                "query_count",
                "id_map",
                "id_rank1",
                "id_rank5",
                "id_rank10",
                "mean_first_id_rank",
                "same_class_top1_wrong_rate",
            ],
            max_rows=15,
        )
    )
    lines.append("")
    lines.append("## Worst classes")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            worst_classes,
            [
                "query_class",
                "query_count",
                "id_map",
                "id_rank1",
                "id_rank5",
                "id_rank10",
                "class_map",
                "class_rank1",
                "class_rank5",
                "class_rank10",
                "same_class_top1_wrong_rate",
            ],
            max_rows=15,
        )
    )
    lines.append("")
    lines.append("## Failure pattern breakdown")
    lines.append("")
    lines.append(
        f"Top-1 wrong but correct class: {float(valid_query_df.loc[~valid_query_df['top1_correct_id'], 'same_class_top1_wrong'].mean()):.4%} of valid queries."
    )
    lines.append(
        f"Correct ID recovered in top-5: {float(valid_query_df['id_hit@5'].mean()):.4%}; in top-10: {recovered_in_top10_rate:.4%}; in top-100: {float(valid_query_df['id_hit@100'].mean()):.4%}."
    )
    lines.append(
        f"Correct class recovered in top-5: {float(valid_query_df['class_hit@5'].mean()):.4%}; in top-10: {float(valid_query_df['class_hit@10'].mean()):.4%}; in top-100: {float(valid_query_df['class_hit@100'].mean()):.4%}."
    )
    lines.append("")
    lines.append("## Common confusion pairs")
    lines.append("")
    if confusion_df.empty:
        lines.append("No confusion pairs were found because all rank-1 predictions were correct.")
    else:
        lines.append(dataframe_to_markdown(confusion_df, ["query_class", "top1_pred_class", "count", "rate_within_query_class"], max_rows=20))

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("The report scores exact object ID retrieval and coarse class retrieval separately, so you can distinguish semantic class errors from within-class identity confusion.")
    lines.append("Queries with no relevant gallery ID are excluded from the metric averages but are still listed in the validation summary.")
    lines.append("If the gallery order differs from sorted numeric image-name order, the script still maps submission indices through the sorted order used by the existing evaluator.")

    return "\n".join(lines)


def analyze_submission(submission_path: Path, data_root: Path, output_dir: Path, topk: int) -> dict:
    query_df = load_labeled_split(data_root / "query.csv", data_root / "query_classes.csv")
    gallery_df = load_labeled_split(data_root / "test.csv", data_root / "test_classes.csv")
    submission_df = load_submission(submission_path)

    raw_submission_rows = int(len(submission_df))
    duplicate_submission_rows = int(submission_df["imageName"].duplicated().sum())
    submission_df = submission_df.drop_duplicates(subset=["imageName"], keep="last").reset_index(drop=True)

    gallery_sorted, gallery_lookup = build_gallery_lookup(gallery_df)
    gallery_order_matches_sorted = gallery_df["imageName"].tolist() == gallery_sorted["imageName"].tolist()

    query_lookup = query_df.set_index("imageName")
    submission_lookup = submission_df.set_index("imageName")["predictions_raw"].to_dict()

    query_results: list[dict] = []
    rows_with_duplicate_indices = 0
    rows_with_out_of_range_indices = 0
    missing_queries = []
    queries_without_gallery_match = 0

    gallery_pid_array = gallery_sorted["objectID"].to_numpy()
    gallery_class_array = gallery_sorted["Class"].to_numpy()
    gallery_name_array = gallery_sorted["imageName"].to_numpy()
    gallery_cam_array = gallery_sorted["cameraID"].to_numpy()

    for _, q_row in query_df.iterrows():
        image_name = q_row["imageName"]
        query_pid = int(q_row["objectID"])
        query_class = q_row["Class"]
        query_camid = q_row["cameraID"]

        raw_prediction = submission_lookup.get(image_name)
        if raw_prediction is None:
            missing_queries.append(image_name)
            continue

        prediction_list = parse_rank_list(raw_prediction)
        if len(prediction_list) != len(set(prediction_list)):
            rows_with_duplicate_indices += 1

        if topk > 0:
            prediction_list = prediction_list[:topk]

        valid_indices = [idx for idx in prediction_list if 1 <= idx <= len(gallery_sorted)]
        if len(valid_indices) != len(prediction_list):
            rows_with_out_of_range_indices += 1

        ranking = np.array(valid_indices, dtype=int) - 1 if valid_indices else np.array([], dtype=int)
        retrieved_pids = gallery_pid_array[ranking] if len(ranking) else np.array([], dtype=int)
        retrieved_classes = gallery_class_array[ranking] if len(ranking) else np.array([], dtype=object)
        retrieved_names = gallery_name_array[ranking] if len(ranking) else np.array([], dtype=object)
        retrieved_camids = gallery_cam_array[ranking] if len(ranking) else np.array([], dtype=object)

        relevant_mask_id = retrieved_pids == query_pid
        relevant_mask_class = retrieved_classes == query_class
        query_gallery_match_count = int((gallery_pid_array == query_pid).sum())
        query_gallery_class_count = int((gallery_class_array == query_class).sum())
        if query_gallery_match_count == 0:
            queries_without_gallery_match += 1

        id_ap = compute_ap(relevant_mask_id, query_gallery_match_count)
        class_ap = compute_ap(relevant_mask_class, query_gallery_class_count)

        first_id = first_hit_rank(relevant_mask_id)
        first_class = first_hit_rank(relevant_mask_class)

        top1_pid = int(retrieved_pids[0]) if len(retrieved_pids) else None
        top1_class = str(retrieved_classes[0]) if len(retrieved_classes) else None
        top1_name = str(retrieved_names[0]) if len(retrieved_names) else None
        top1_camid = str(retrieved_camids[0]) if len(retrieved_camids) else None

        top1_correct_id = bool(len(retrieved_pids) and top1_pid == query_pid)
        top1_correct_class = bool(len(retrieved_classes) and top1_class == query_class)
        same_class_top1_wrong = bool((not top1_correct_id) and top1_correct_class)

        result = {
            "imageName": image_name,
            "query_objectID": query_pid,
            "query_class": query_class,
            "query_cameraID": query_camid,
            "submission_length": len(prediction_list),
            "valid_prediction_length": len(valid_indices),
            "duplicate_indices": len(prediction_list) != len(set(prediction_list)),
            "out_of_range_indices": len(valid_indices) != len(prediction_list),
            "query_gallery_match_count": query_gallery_match_count,
            "query_gallery_class_count": query_gallery_class_count,
            "valid_for_id_metrics": query_gallery_match_count > 0,
            "valid_for_class_metrics": query_gallery_class_count > 0,
            "id_ap": id_ap,
            "class_ap": class_ap,
            "id_hit@1": float(hit_at_k(relevant_mask_id, 1)),
            "id_hit@5": float(hit_at_k(relevant_mask_id, 5)),
            "id_hit@10": float(hit_at_k(relevant_mask_id, 10)),
            "id_hit@100": float(hit_at_k(relevant_mask_id, 100)),
            "class_hit@1": float(hit_at_k(relevant_mask_class, 1)),
            "class_hit@5": float(hit_at_k(relevant_mask_class, 5)),
            "class_hit@10": float(hit_at_k(relevant_mask_class, 10)),
            "class_hit@100": float(hit_at_k(relevant_mask_class, 100)),
            "first_id_rank": first_id,
            "first_class_rank": first_class,
            "top1_correct_id": top1_correct_id,
            "top1_correct_class": top1_correct_class,
            "same_class_top1_wrong": same_class_top1_wrong,
            "top1_pred_image": top1_name,
            "top1_pred_objectID": top1_pid,
            "top1_pred_class": top1_class,
            "top1_pred_cameraID": top1_camid,
        }
        query_results.append(result)

    per_query = pd.DataFrame(query_results)
    if per_query.empty:
        raise RuntimeError("No queries were evaluated. Check that the submission names match query.csv.")

    if missing_queries:
        missing_rows = pd.DataFrame(
            {
                "imageName": missing_queries,
                "query_objectID": [int(query_lookup.loc[name, "objectID"]) for name in missing_queries],
                "query_class": [query_lookup.loc[name, "Class"] for name in missing_queries],
                "query_cameraID": [query_lookup.loc[name, "cameraID"] for name in missing_queries],
                "submission_length": 0,
                "valid_prediction_length": 0,
                "duplicate_indices": False,
                "out_of_range_indices": False,
                "query_gallery_match_count": [int((gallery_pid_array == int(query_lookup.loc[name, "objectID"])).sum()) for name in missing_queries],
                "query_gallery_class_count": [int((gallery_class_array == query_lookup.loc[name, "Class"]).sum()) for name in missing_queries],
                "valid_for_id_metrics": [int((gallery_pid_array == int(query_lookup.loc[name, "objectID"])).sum()) > 0 for name in missing_queries],
                "valid_for_class_metrics": [int((gallery_class_array == query_lookup.loc[name, "Class"]).sum()) > 0 for name in missing_queries],
                "id_ap": np.nan,
                "class_ap": np.nan,
                "id_hit@1": np.nan,
                "id_hit@5": np.nan,
                "id_hit@10": np.nan,
                "id_hit@100": np.nan,
                "class_hit@1": np.nan,
                "class_hit@5": np.nan,
                "class_hit@10": np.nan,
                "class_hit@100": np.nan,
                "first_id_rank": np.nan,
                "first_class_rank": np.nan,
                "top1_correct_id": False,
                "top1_correct_class": False,
                "same_class_top1_wrong": False,
                "top1_pred_image": None,
                "top1_pred_objectID": None,
                "top1_pred_class": None,
                "top1_pred_cameraID": None,
            }
        )
        per_query = pd.concat([per_query, missing_rows], ignore_index=True)

    valid_query_df = per_query.loc[per_query["valid_for_id_metrics"]].copy()
    id_summary = summarize_groups(valid_query_df, "query_objectID")
    class_summary = summarize_groups(valid_query_df, "query_class")
    confusion_df = build_confusion_table(valid_query_df, top_n=20)

    validation = {
        "submission_rows": raw_submission_rows,
        "query_rows": int(len(query_df)),
        "matched_rows": int(len(per_query)),
        "missing_queries": int(len(missing_queries)),
        "extra_submission_rows": int(max(raw_submission_rows - len(query_df), 0)),
        "duplicate_submission_rows": duplicate_submission_rows,
        "rows_with_duplicate_indices": int(rows_with_duplicate_indices),
        "rows_with_out_of_range_indices": int(rows_with_out_of_range_indices),
        "queries_without_gallery_match": int(queries_without_gallery_match),
        "gallery_order_matches_sorted": bool(gallery_order_matches_sorted),
    }

    report_text = make_report_text(
        submission_path=submission_path,
        data_root=data_root,
        per_query=per_query,
        id_summary=id_summary,
        class_summary=class_summary,
        confusion_df=confusion_df,
        validation=validation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "per_query_metrics.csv"
    id_summary_path = output_dir / "id_summary.csv"
    class_summary_path = output_dir / "class_summary.csv"
    confusion_path = output_dir / "class_confusions.csv"
    failures_path = output_dir / "worst_queries.csv"
    summary_json_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    per_query.sort_values(["id_ap", "id_hit@1", "first_id_rank"], ascending=[True, True, False]).to_csv(per_query_path, index=False)
    id_summary.sort_values(["id_map", "id_rank1", "query_count"], ascending=[True, True, False]).to_csv(id_summary_path, index=False)
    class_summary.sort_values(["id_map", "id_rank1", "query_count"], ascending=[True, True, False]).to_csv(class_summary_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    per_query.sort_values(["id_ap", "id_hit@1", "first_id_rank"], ascending=[True, True, False]).head(50).to_csv(failures_path, index=False)

    summary = {
        "submission_path": str(submission_path),
        "data_root": str(data_root),
        "topk": int(topk),
        "validation": validation,
        "metrics": {
            "id_map": float(valid_query_df["id_ap"].mean()),
            "id_rank1": float(valid_query_df["id_hit@1"].mean()),
            "id_rank5": float(valid_query_df["id_hit@5"].mean()),
            "id_rank10": float(valid_query_df["id_hit@10"].mean()),
            "class_map": float(valid_query_df["class_ap"].mean()),
            "class_rank1": float(valid_query_df["class_hit@1"].mean()),
            "class_rank5": float(valid_query_df["class_hit@5"].mean()),
            "class_rank10": float(valid_query_df["class_hit@10"].mean()),
        },
        "artifacts": {
            "report_md": str(report_path),
            "summary_json": str(summary_json_path),
            "per_query_metrics_csv": str(per_query_path),
            "id_summary_csv": str(id_summary_path),
            "class_summary_csv": str(class_summary_path),
            "class_confusions_csv": str(confusion_path),
            "worst_queries_csv": str(failures_path),
        },
    }

    report_path.write_text(report_text, encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a UAM Unified retrieval submission")
    parser.add_argument("--submission", default="vit_large_submission.csv", type=str, help="Submission CSV to analyze")
    parser.add_argument("--data-root", default="UAM_Unified", type=str, help="Dataset root containing query/test CSVs")
    parser.add_argument("--output-dir", default="outputs/submission_analysis", type=str, help="Directory for the report artifacts")
    parser.add_argument("--topk", default=100, type=int, help="Submission rank depth to treat as the evaluation cutoff")
    args = parser.parse_args()

    submission_path = Path(args.submission).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    summary = analyze_submission(submission_path, data_root, output_dir, args.topk)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(f"Report written to {summary['artifacts']['report_md']}")


if __name__ == "__main__":
    main()