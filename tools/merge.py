from pathlib import Path
import csv
import shutil

URBAN = Path("Urban2026")          # URVAM-ReID2026 (challenge)
UAM   = Path("UAM_Unified")        # UrbAM-ReID (external)
OUT   = Path("Urban2026_UAM_MERGED")

URBAN_PREFIX = "urban"
UAM_PREFIX   = "uam"


def norm_class(s: str) -> str:
    key = "".join(ch.lower() for ch in s.strip() if ch.isalnum())
    if key in {"container", "containers"}:
        return "Container"
    if key in {"rubbishbins", "rubbishbin", "trashbin", "trashbins", "bin", "bins", "wastebin", "wastebins"}:
        return "RubbishBins"
    if key in {"crosswalk", "crosswalks", "zebracrossing", "zebracrosswalk"}:
        return "Crosswalk"
    if key in {"trafficsign", "trafficsigns", "trafficsignal", "trafficsignals", "roadsign", "roadsigns"}:
        return "TrafficSign"
    return s.strip()


def read_train_3col(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return rows
        for cam, img, pid in r:
            rows.append((cam.strip(), img.strip(), int(pid)))
    return rows


def read_train_classes_4col(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return rows
        for cam, img, pid, cls in r:
            rows.append((cam.strip(), img.strip(), int(pid), norm_class(cls)))
    return rows


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output folder: {OUT}")

    urban_train = read_train_3col(URBAN / "train.csv")
    uam_train   = read_train_3col(UAM / "train.csv")
    if not urban_train:
        raise RuntimeError("Urban2026/train.csv missing or empty")
    if not uam_train:
        raise RuntimeError("UAM_Unified/train.csv missing or empty")

    max_urban_id = max(pid for _, _, pid in urban_train)
    shift = max_urban_id + 1

    OUT.mkdir(parents=True)

    # --- merged train.csv (keep Urban2026 header style) ---
    merged_train = []
    for cam, img, pid in urban_train:
        merged_train.append((cam, f"{URBAN_PREFIX}/{img}", pid))
    for cam, img, pid in uam_train:
        merged_train.append((cam, f"{UAM_PREFIX}/{img}", pid + shift))

    write_csv(
        OUT / "train.csv",
        ["cameraID", "imageName", "Corresponding Indexes"],
        merged_train,
    )

    # --- merged train_classes.csv (if both inputs exist) ---
    urban_cls_path = URBAN / "train_classes.csv"
    uam_cls_path   = UAM / "train_classes.csv"
    if urban_cls_path.exists() and uam_cls_path.exists():
        urban_cls = read_train_classes_4col(urban_cls_path)
        uam_cls   = read_train_classes_4col(uam_cls_path)

        merged_cls = []
        for cam, img, pid, cls in urban_cls:
            merged_cls.append((cam, f"{URBAN_PREFIX}/{img}", pid, cls))
        for cam, img, pid, cls in uam_cls:
            merged_cls.append((cam, f"{UAM_PREFIX}/{img}", pid + shift, cls))

        write_csv(
            OUT / "train_classes.csv",
            ["cameraID", "imageName", "Corresponding Indexes", "Class"],
            merged_cls,
        )

    # --- copy eval metadata from Urban2026 ONLY (keep challenge eval untouched) ---
    for name in ["query.csv", "test.csv", "query_classes.csv", "test_classes.csv", "sample_submission.csv"]:
        src = URBAN / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    # --- copy images with collision-safe subfolders ---
    shutil.copytree(URBAN / "image_train", OUT / "image_train" / URBAN_PREFIX)
    shutil.copytree(UAM   / "image_train", OUT / "image_train" / UAM_PREFIX)

    # eval images: Urban2026 only
    if (URBAN / "image_query").exists():
        shutil.copytree(URBAN / "image_query", OUT / "image_query")
    if (URBAN / "image_test").exists():
        shutil.copytree(URBAN / "image_test", OUT / "image_test")

    print("Done.")
    print(f"Urban max id: {max_urban_id}  |  UAM shift: +{shift}")
    print(f"Urban train: {len(urban_train)}  UAM train: {len(uam_train)}  Merged: {len(merged_train)}")
    print(f"Output: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())