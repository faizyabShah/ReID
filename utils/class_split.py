from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable


TRAFFIC_CLASS_NAMES = {"traffic", "trafficsign", "trafficsignal", "signal", "sign"}


def normalize_class_name(class_name: object) -> str:
    normalized = str(class_name).strip().lower().replace(" ", "").replace("_", "")
    if normalized in TRAFFIC_CLASS_NAMES:
        return "traffic"
    return normalized


def is_traffic_class(class_name: object) -> bool:
    return normalize_class_name(class_name) == "traffic"


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(name).strip(): str(name).strip() for name in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def read_class_map(csv_path: str | os.PathLike[str]) -> dict[str, str]:
    path = Path(csv_path)
    if not path.exists():
        return {}

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}

        image_col = find_column(reader.fieldnames, ["imageName", "imagename", "image_name"])
        class_col = find_column(reader.fieldnames, ["Class", "class"])
        if image_col is None or class_col is None:
            return {}

        class_map: dict[str, str] = {}
        for row in reader:
            image_name = str(row.get(image_col, "")).strip()
            if not image_name:
                continue
            normalized = normalize_class_name(row.get(class_col, ""))
            class_map[image_name] = normalized
            class_map[Path(image_name).name] = normalized
        return class_map


def resolve_image_path(root: str | os.PathLike[str], subdir: str, image_name: str) -> str:
    root_path = Path(root)
    candidates = [
        root_path / subdir / image_name,
        root_path / subdir / Path(image_name).name,
        root_path / image_name,
        root_path / Path(image_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def attach_class_info(record: tuple, class_name: str | None) -> tuple:
    if len(record) > 3:
        image_path, pid, camid, others = record
        merged = dict(others)
    else:
        image_path, pid, camid = record
        merged = {}
    if class_name is not None:
        merged["class_name"] = normalize_class_name(class_name)
    return image_path, pid, camid, merged


def split_records_by_class(records: list[tuple], class_name: str) -> list[tuple]:
    target = normalize_class_name(class_name)
    filtered = []
    for record in records:
        if len(record) < 4:
            continue
        class_value = record[3].get("class_name")
        if class_value is None:
            continue
        if normalize_class_name(class_value) == target:
            filtered.append(record)
    return filtered
