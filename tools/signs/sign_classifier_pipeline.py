"""
sign_classifier_pipeline.py
============================
End-to-end pipeline to train a traffic sign subtype classifier
on the UrbanElementsReID dataset using heuristic pseudo-labels.

Adapted for actual CSV column names:
    - Class column:    "Class"  (values: "TrafficSign", "RubbishBins", etc.)
    - Object ID:       "Corresponding Indexes"  (train CSV only)
    - Image filename:  "imageName"

Sign subtypes produced:
    prohibitory, warning, mandatory, informational,
    stop, yield, pedestrian_info, sign_back, unknown

Steps:
    1. pseudo_label  — heuristic majority-vote per identity
    2. review        — image grids for manual label verification
    3. train         — fine-tune MobileNetV2 on corrected labels
    4. predict       — classify gallery/query signs

Usage (run from repo root: c:\Users\Narmeen\my_codes\ReID\ReID):

    python sign_classifier_pipeline.py pseudo_label \
        --csv Urban2026_UAM_MERGED/train_classes.csv \
        --image_dir Urban2026_UAM_MERGED/image_train \
        --output pseudo_labels.csv

    python sign_classifier_pipeline.py review \
        --csv pseudo_labels.csv \
        --image_dir Urban2026_UAM_MERGED/image_train \
        --output_dir review_sheets --n_identities 50

    # (manually correct pseudo_labels.csv where review sheets show wrong labels)

    python sign_classifier_pipeline.py train \
        --csv pseudo_labels.csv \
        --image_dir Urban2026_UAM_MERGED/image_train \
        --output_dir sign_classifier_model --epochs 30

    python sign_classifier_pipeline.py predict \
        --csv Urban2026_UAM_MERGED/query_classes.csv \
        --image_dir Urban2026_UAM_MERGED/image_query \
        --model_dir sign_classifier_model \
        --output query_sign_types.csv

    python sign_classifier_pipeline.py predict \
        --csv Urban2026_UAM_MERGED/test_classes.csv \
        --image_dir Urban2026_UAM_MERGED/image_test \
        --model_dir sign_classifier_model \
        --output test_sign_types.csv
"""

import os
import argparse
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image


# ---------------------------------------------------------------------------
# CSV column names (matches UrbanElementsReID merged dataset)
# ---------------------------------------------------------------------------
COL_CLASS  = "Class"
COL_ID     = "Corresponding Indexes"
COL_IMAGE  = "imageName"

# train_classes.csv uses "TrafficSign"; query/test use "trafficsignal"
# Normalise by checking the set of known aliases.
_SIGN_ALIASES = {"trafficsign", "trafficsignal", "traffic_sign", "sign", "signal"}


def _is_traffic_sign(class_value: str) -> bool:
    norm = str(class_value).strip().lower().replace("_", "").replace(" ", "")
    return norm in _SIGN_ALIASES


# ---------------------------------------------------------------------------
# HEURISTIC CLASSIFIER
# ---------------------------------------------------------------------------

COLOR_RANGES = {
    "blue":   [((100, 80, 50),  (130, 255, 255))],
    "red":    [((0, 100, 60),   (10, 255, 255)),
               ((165, 100, 60), (180, 255, 255))],
    "yellow": [((15, 70, 70),   (35, 255, 220))],
    "green":  [((40, 50, 50),   (85, 255, 255))],
    "white":  [((0, 0, 180),    (180, 40, 255))],
    "gray":   [((0, 0, 60),     (180, 30, 160))],
}


def _detect_colors(hsv):
    h, w = hsv.shape[:2]
    total = h * w
    fracs = {}
    for name, ranges in COLOR_RANGES.items():
        count = 0
        for lo, hi in ranges:
            count += np.count_nonzero(cv2.inRange(hsv, np.array(lo), np.array(hi)))
        f = count / total
        if f >= 0.08:
            fracs[name] = f
    return fracs


def _detect_shape(img):
    h, w = img.shape[:2]
    aspect = w / h
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        peri = cv2.arcLength(largest, True)
        if peri > 0 and area > (h * w * 0.05):
            circ = 4 * np.pi * area / (peri ** 2)
            approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
            n = len(approx)
            if circ > 0.72:
                return "circle"
            if n == 3:
                pts = approx.reshape(-1, 2)
                top_count = np.sum(pts[:, 1] < (pts[:, 1].min() + pts[:, 1].max()) / 2)
                return "inv_triangle" if top_count >= 2 else "triangle"
            if n == 8:
                return "octagon"

    if aspect > 2.0:    return "wide"
    elif aspect > 1.2:  return "landscape"
    elif aspect > 0.8:  return "square"
    else:               return "portrait"


TYPE_MAP = {
    "red_circle": "prohibitory",   "white_circle": "prohibitory",
    "red_triangle": "warning",     "yellow_triangle": "warning",   "white_triangle": "warning",
    "blue_circle": "mandatory",
    "blue_square": "pedestrian_info",
    "blue_landscape": "informational", "blue_wide": "informational",
    "blue_portrait": "informational",
    "green_square": "informational",   "green_landscape": "informational",
    "green_wide": "informational",     "green_portrait": "informational",
    "white_wide": "informational",     "white_landscape": "informational",
    "yellow_wide": "informational",    "yellow_landscape": "informational",
    "red_octagon": "stop",         "white_octagon": "stop",
    "red_inv_triangle": "yield",   "white_inv_triangle": "yield",
    "gray_square": "sign_back",    "gray_portrait": "sign_back",
    "gray_landscape": "sign_back", "gray_wide": "sign_back",
    "gray_circle": "sign_back",    "gray_triangle": "sign_back",
    "gray_inv_triangle": "sign_back", "gray_octagon": "sign_back",
}


def heuristic_classify(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colors = _detect_colors(hsv)
    if not colors:
        return "unknown"
    dom = max(colors, key=colors.get)
    shape = _detect_shape(img)
    return TYPE_MAP.get(f"{dom}_{shape}", "unknown")


# ---------------------------------------------------------------------------
# STEP 1: GENERATE PSEUDO-LABELS
# ---------------------------------------------------------------------------

def generate_pseudo_labels(csv_path, image_dir, output_path):
    df = pd.read_csv(csv_path)

    sign_mask = df[COL_CLASS].apply(_is_traffic_sign)
    signs = df[sign_mask].copy()

    if signs.empty:
        print(f"No rows matched any traffic-sign alias.  Classes found: {df[COL_CLASS].unique()}")
        return

    print(f"Traffic sign images:   {len(signs)}")
    print(f"Unique identities:     {signs[COL_ID].nunique()}")

    identity_labels = {}
    identity_confidence = {}

    for oid, group in tqdm(signs.groupby(COL_ID), desc="Heuristic labeling"):
        preds = []
        for _, row in group.iterrows():
            path = os.path.join(image_dir, row[COL_IMAGE])
            if os.path.exists(path):
                preds.append(heuristic_classify(path))

        if not preds:
            identity_labels[oid] = "unknown"
            identity_confidence[oid] = 0.0
            continue

        counter = Counter(preds)
        top, top_n = counter.most_common(1)[0]
        identity_labels[oid] = top
        identity_confidence[oid] = round(top_n / len(preds), 3)

    signs["sign_type"] = signs[COL_ID].map(identity_labels)
    signs["confidence"] = signs[COL_ID].map(identity_confidence)

    signs.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    print("\nDistribution (by identity):")
    for t, c in Counter(identity_labels.values()).most_common():
        print(f"  {t:25s}: {c}")
    print(f"\nLow-confidence (<0.5): {sum(1 for v in identity_confidence.values() if v < 0.5)}")


# ---------------------------------------------------------------------------
# STEP 2: REVIEW SHEET GENERATION
# ---------------------------------------------------------------------------

def generate_review_sheet(csv_path, image_dir, output_dir, n_identities=50):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    id_info = df.groupby(COL_ID).agg(
        sign_type=("sign_type", "first"),
        confidence=("confidence", "first"),
        n_images=(COL_IMAGE, "count"),
    ).reset_index()

    id_info = id_info.sort_values("confidence").head(n_identities)
    print(f"Generating review sheets for {len(id_info)} identities...")

    for idx, (_, row) in enumerate(id_info.iterrows()):
        oid = row[COL_ID]
        stype = row["sign_type"]
        conf = row["confidence"]

        sample = df[df[COL_ID] == oid].sample(min(8, len(df[df[COL_ID] == oid])), random_state=42)
        n = len(sample)
        cols = min(4, n)
        rows_count = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows_count, cols, figsize=(3 * cols, 3 * rows_count))
        if rows_count == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        fig.suptitle(
            f"ID: {oid}  |  Label: {stype}  |  Conf: {conf:.2f}  |  N={row['n_images']}",
            fontsize=11, fontweight="bold"
        )

        for i, (_, img_row) in enumerate(sample.iterrows()):
            path = os.path.join(image_dir, img_row[COL_IMAGE])
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(img_row[COL_IMAGE], fontsize=6)
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"review_{idx:03d}_id{oid}_{stype}.png"),
            dpi=120, bbox_inches="tight"
        )
        plt.close()

    correction_path = os.path.join(output_dir, "corrections.csv")
    id_info[[COL_ID, "sign_type", "confidence", "n_images"]].to_csv(correction_path, index=False)
    print(f"\nReview images saved to: {output_dir}/")
    print(f"Correction template:    {correction_path}")
    print("\nWorkflow:")
    print("  1. Open each review_*.png")
    print("  2. Fix sign_type in corrections.csv where the label looks wrong")
    print("  3. Merge corrections back into pseudo_labels.csv (or re-run pseudo_label)")


# ---------------------------------------------------------------------------
# STEP 3: TRAIN CLASSIFIER
# ---------------------------------------------------------------------------

class SignTypeDataset(Dataset):
    def __init__(self, df, image_dir, transform, class_to_idx):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row[COL_IMAGE])
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.class_to_idx[row["sign_type"]]


def train_classifier(csv_path, image_dir, output_dir, epochs=30, batch_size=32,
                     lr_head=1e-3, lr_backbone=1e-4, freeze_epochs=5):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["sign_type"].notna() & (df["sign_type"] != "unknown")].copy()

    classes = sorted(df["sign_type"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    n_classes = len(classes)

    print(f"Classes ({n_classes}): {classes}")
    print(f"Total images: {len(df)}")
    for c in classes:
        print(f"  {c:25s}: {(df['sign_type'] == c).sum()}")

    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    # Stratified split by identity to prevent leakage
    ids = df[COL_ID].unique()
    np.random.seed(42)
    np.random.shuffle(ids)
    split = int(0.8 * len(ids))
    train_df = df[df[COL_ID].isin(ids[:split])].copy()
    val_df   = df[df[COL_ID].isin(ids[split:])].copy()
    print(f"\nTrain: {len(train_df)} images | Val: {len(val_df)} images")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SignTypeDataset(train_df, image_dir, train_tf, class_to_idx)
    val_ds   = SignTypeDataset(val_df,   image_dir, val_tf,   class_to_idx)

    train_labels = [class_to_idx[t] for t in train_df["sign_type"]]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Phase 1: head only (epochs 1-{freeze_epochs})")
    print(f"{'='*60}")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_head)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            print(f"\n{'='*60}")
            print(f"Phase 2: full fine-tune (epochs {freeze_epochs+1}-{epochs})")
            print(f"{'='*60}")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {"params": model.features.parameters(),    "lr": lr_backbone},
                {"params": model.classifier.parameters(),  "lr": lr_head * 0.1},
            ])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs
            )

        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)

        scheduler.step()

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:2d}/{epochs}  "
              f"Train {tr_correct/tr_total:.3f}  Val {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  → saved (val_acc={val_acc:.3f})")

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved → {output_dir}/")


# ---------------------------------------------------------------------------
# STEP 4: PREDICT WITH TRAINED MODEL
# ---------------------------------------------------------------------------

def predict(csv_path, image_dir, model_dir, output_path, batch_size=64):
    with open(os.path.join(model_dir, "class_mapping.json")) as f:
        mapping = json.load(f)
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    n_classes = len(class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best_model.pth"), map_location=device
    ))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(csv_path)
    sign_mask = df[COL_CLASS].apply(_is_traffic_sign)
    signs = df[sign_mask].copy()

    print(f"Classifying {len(signs)} traffic sign images...")

    all_preds, all_confs = [], []
    paths = [os.path.join(image_dir, name) for name in signs[COL_IMAGE]]

    for i in tqdm(range(0, len(paths), batch_size), desc="Predicting"):
        batch_imgs = []
        for p in paths[i:i + batch_size]:
            try:
                batch_imgs.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                batch_imgs.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch_tensor), dim=1)
            confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    signs["sign_type"] = [idx_to_class[p] for p in all_preds]
    signs["sign_type_confidence"] = [round(float(c), 3) for c in all_confs]

    df["sign_type"] = None
    df["sign_type_confidence"] = None
    df.loc[sign_mask, "sign_type"] = signs["sign_type"].values
    df.loc[sign_mask, "sign_type_confidence"] = signs["sign_type_confidence"].values

    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    print("\nPrediction distribution:")
    for t, c in signs["sign_type"].value_counts().items():
        print(f"  {t:25s}: {c}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sign type classifier pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("pseudo_label")
    p1.add_argument("--csv",       required=True)
    p1.add_argument("--image_dir", required=True)
    p1.add_argument("--output",    required=True)

    p2 = sub.add_parser("review")
    p2.add_argument("--csv",          required=True)
    p2.add_argument("--image_dir",    required=True)
    p2.add_argument("--output_dir",   required=True)
    p2.add_argument("--n_identities", type=int, default=50)

    p3 = sub.add_parser("train")
    p3.add_argument("--csv",           required=True)
    p3.add_argument("--image_dir",     required=True)
    p3.add_argument("--output_dir",    required=True)
    p3.add_argument("--epochs",        type=int,   default=30)
    p3.add_argument("--batch_size",    type=int,   default=32)
    p3.add_argument("--lr_head",       type=float, default=1e-3)
    p3.add_argument("--lr_backbone",   type=float, default=1e-4)
    p3.add_argument("--freeze_epochs", type=int,   default=5)

    p4 = sub.add_parser("predict")
    p4.add_argument("--csv",        required=True)
    p4.add_argument("--image_dir",  required=True)
    p4.add_argument("--model_dir",  required=True)
    p4.add_argument("--output",     required=True)
    p4.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "pseudo_label":
        generate_pseudo_labels(args.csv, args.image_dir, args.output)
    elif args.command == "review":
        generate_review_sheet(args.csv, args.image_dir, args.output_dir, args.n_identities)
    elif args.command == "train":
        train_classifier(args.csv, args.image_dir, args.output_dir,
                         args.epochs, args.batch_size,
                         args.lr_head, args.lr_backbone, args.freeze_epochs)
    elif args.command == "predict":
        predict(args.csv, args.image_dir, args.model_dir, args.output, args.batch_size)


if __name__ == "__main__":
    main()
