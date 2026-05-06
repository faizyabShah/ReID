"""
shape_sign_grouping.py
======================
Shape-based traffic sign grouping for prototype retrieval.

WHY SHAPE INSTEAD OF VIENNA CONVENTION TYPE:
    - Shape is the MOST reliable visual axis for sign classification
    - GTSRB fine-grained type labels degrade heavily under domain shift
      (trained on German signs, applied to Spanish city data)
    - Shape is geometrically present in PAT embeddings (even unfused)
    - Shape misclassification rate is much lower than type misclassification
    - Shape groups are ~perfectly correlated with Vienna Convention anyway:
        circular  → prohibitory (C) + mandatory (D)
        triangle  → warning/danger (A)
        rectangle → information (F/G) + directional
        octagon   → stop (B)
        diamond   → priority
        inverted_triangle → yield

USAGE:
    # Option 1: Replace GTSRB coarse types with shape groups
    from shape_sign_grouping import ShapeClassifier, replace_types_with_shapes
    
    shape_clf = ShapeClassifier(device='cuda')
    df = replace_types_with_shapes(df, image_dir, shape_clf)
    
    # Option 2: Use alongside GTSRB (dual-axis prototypes)
    from shape_sign_grouping import add_shape_column
    
    df = add_shape_column(df, image_dir, shape_clf)
    # Now df has both 'sign_type' (Vienna) and 'sign_shape' columns

INTEGRATION WITH PROTOTYPE PIPELINE:
    The key insight: use UNFUSED embeddings (final-layer CLS token only)
    for prototype matching, then FUSED embeddings for the actual retrieval.
    
    Unfused (1×D = 1024): inter-prototype similarity ~0.5 → well-separated
    Fused   (8×D = 8192): inter-prototype similarity >0.8 → collapsed
    
    Pipeline:
        1. Build prototypes from unfused features, grouped by shape
        2. At inference: route query using unfused features vs shape prototypes
        3. Filter gallery to matched shape + all back views
        4. Re-rank within subset using FUSED features (better for instance-level)

Requirements:
    pip install torch torchvision opencv-python numpy pandas tqdm
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
# SHAPE GROUPS
# ═══════════════════════════════════════════════════════════════════════

SHAPE_GROUPS = [
    "circular",           # prohibitory (red border) + mandatory (blue)
    "triangular",         # warning/danger signs
    "rectangular",        # information, directional, street names
    "octagonal",          # stop signs
    "diamond",            # priority signs
    "inverted_triangle",  # yield signs
    "other",              # irregular shapes, damaged, unrecognizable
]

# Map GTSRB classes to shapes (more reliable than mapping to Vienna types)
GTSRB_TO_SHAPE = {
    # Circular — speed limits, no-entry, mandatory blue
    0: "circular", 1: "circular", 2: "circular", 3: "circular",
    4: "circular", 5: "circular", 6: "circular", 7: "circular",
    8: "circular", 9: "circular", 10: "circular", 15: "circular",
    16: "circular", 17: "circular",
    32: "circular", 41: "circular", 42: "circular",  # derestriction (circular)
    33: "circular", 34: "circular", 35: "circular", 36: "circular",
    37: "circular", 38: "circular", 39: "circular", 40: "circular",

    # Triangular — warning/danger
    11: "triangular", 18: "triangular", 19: "triangular", 20: "triangular",
    21: "triangular", 22: "triangular", 23: "triangular", 24: "triangular",
    25: "triangular", 26: "triangular", 27: "triangular", 28: "triangular",
    29: "triangular", 30: "triangular", 31: "triangular",

    # Special shapes
    12: "diamond",              # priority road (diamond)
    13: "inverted_triangle",    # yield (inverted triangle)
    14: "octagonal",            # stop (octagon)
}

WILDCARD_SHAPES = frozenset({"sign_back", "unknown", "other"})


# ═══════════════════════════════════════════════════════════════════════
# SHAPE CLASSIFIER — GEOMETRY-BASED (NO NEURAL NETWORK)
# ═══════════════════════════════════════════════════════════════════════
# This is intentionally simple and robust. It uses contour analysis
# which is invariant to color/texture and works on low-res images.

class ShapeClassifier:
    """
    Classify traffic sign shape using contour analysis.
    
    This works MUCH better than color-based classification because:
    - Contours are invariant to lighting, color degradation, JPEG artifacts
    - Traffic signs are designed to have distinctive shapes
    - Works even on partially occluded signs (largest contour)
    
    The trick: we don't analyze the raw image. We first threshold to
    find the sign region (usually the largest bright/colored blob),
    then analyze that contour's geometry.
    """
    
    def __init__(self, device='cpu'):
        """device param kept for API compatibility but not used — this is CPU-only."""
        self.device = device
    
    def classify_single(self, image_path):
        """
        Classify one image's sign shape.
        
        Returns: (shape: str, confidence: float)
        """
        img = cv2.imread(image_path)
        if img is None:
            return "other", 0.0
        
        h, w = img.shape[:2]
        
        # ── Step 1: Find the sign contour ────────────────────────
        # Convert to multiple color spaces for robust segmentation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Try multiple approaches to find the sign region
        candidates = []
        
        # Approach A: Otsu threshold on grayscale
        _, thresh_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        candidates.append(thresh_otsu)
        
        # Approach B: Saturation threshold (colored signs pop out)
        sat = hsv[:, :, 1]
        _, thresh_sat = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)
        candidates.append(thresh_sat)
        
        # Approach C: Edge-based (Canny → fill)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        candidates.append(edges_dilated)
        
        # Pick the candidate with the best contour
        best_contour = None
        best_area = 0
        
        for thresh in candidates:
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            # Sign should be at least 5% of image area
            if area > h * w * 0.05 and area > best_area:
                best_area = area
                best_contour = largest
        
        if best_contour is None:
            return "other", 0.1
        
        # ── Step 2: Analyze contour geometry ─────────────────────
        return self._classify_contour(best_contour, h * w)
    
    def _classify_contour(self, contour, image_area):
        """Classify shape from contour using geometric features."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "other", 0.1
        
        # Circularity: 4π·area / perimeter² — 1.0 = perfect circle
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Approximate polygon
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        n_vertices = len(approx)
        
        # Bounding rect aspect ratio
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        
        # Extent: area / bounding rect area
        extent = area / max(bw * bh, 1)
        
        # ── Classification rules ─────────────────────────────────
        
        # Circular: high circularity, high solidity
        if circularity > 0.75 and solidity > 0.85:
            return "circular", min(circularity, 0.99)
        
        # Octagonal: 7-9 vertices, high solidity, roughly square
        if 7 <= n_vertices <= 10 and solidity > 0.85 and 0.8 < aspect < 1.25:
            return "octagonal", 0.80
        
        # Triangular: 3 vertices (pointing up)
        if n_vertices == 3 and solidity > 0.7:
            # Check if pointing up or down (inverted)
            moments = cv2.moments(approx)
            if moments["m00"] > 0:
                cy = moments["m01"] / moments["m00"]
                # If centroid is in top half → inverted (yield)
                _, by, _, bh_rect = cv2.boundingRect(approx)
                relative_cy = (cy - by) / max(bh_rect, 1)
                if relative_cy < 0.4:
                    return "inverted_triangle", 0.80
                else:
                    return "triangular", 0.85
            return "triangular", 0.75
        
        # Diamond: 4 vertices, rotated 45° (narrow aspect in bounding box)
        if n_vertices == 4 and solidity > 0.8:
            # Check if it's a rotated square (diamond) or rectangle
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            rect_aspect = min(rect_w, rect_h) / max(rect_w, rect_h, 1)
            angle = rect[2]
            
            # Diamond: roughly square when un-rotated, with ~45° rotation
            if rect_aspect > 0.7 and (30 < abs(angle) < 60 or
                                       30 < abs(angle - 90) < 60):
                return "diamond", 0.75
            
            # Wide rectangle (directional/info sign)
            if aspect > 1.5:
                return "rectangular", 0.80
            # Tall or square rectangle
            return "rectangular", 0.70
        
        # Rectangle: 4-6 vertices (accounting for noise), wider shapes
        if 4 <= n_vertices <= 6 and solidity > 0.75:
            return "rectangular", 0.65
        
        # Fallback: high circularity but noisy contour
        if circularity > 0.6:
            return "circular", 0.55
        
        # Wide aspect → rectangular (directional boards)
        if aspect > 2.0 and solidity > 0.7:
            return "rectangular", 0.60
        
        return "other", 0.3
    
    def classify_batch(self, image_paths):
        """Classify a list of images. Returns (shapes, confidences)."""
        shapes = []
        confs = []
        for path in tqdm(image_paths, desc="Classifying shapes"):
            shape, conf = self.classify_single(path)
            shapes.append(shape)
            confs.append(conf)
        return shapes, confs


# ═══════════════════════════════════════════════════════════════════════
# HYBRID CLASSIFIER: GTSRB + SHAPE GEOMETRY
# ═══════════════════════════════════════════════════════════════════════
# Uses GTSRB prediction mapped to shape, cross-checked with geometry.
# When they agree → high confidence. When they disagree → geometry wins
# (because GTSRB degrades under domain shift but shape is invariant).

class HybridShapeClassifier:
    """
    Combines GTSRB ViT prediction (mapped to shape) with geometric
    contour analysis. Agreement = high confidence. Disagreement =
    trust geometry (more robust to domain shift).
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.geometry_clf = ShapeClassifier(device=device)
        self.gtsrb_model = None
        self.gtsrb_processor = None
    
    def _load_gtsrb(self):
        """Lazy-load GTSRB model."""
        if self.gtsrb_model is not None:
            return
        
        from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor
        
        model_name = "bazyl/gtsrb-model"
        target_num_labels = 44
        self.gtsrb_processor = ViTImageProcessor.from_pretrained(model_name)
        
        config = AutoConfig.from_pretrained(model_name)
        id2label_raw = dict(getattr(config, "id2label", {}) or {})
        sanitized = {}
        for idx in range(target_num_labels):
            label = id2label_raw.get(str(idx), id2label_raw.get(idx))
            sanitized[idx] = label if label is not None else f"unknown_{idx}"
        config.id2label = sanitized
        config.label2id = {label: idx for idx, label in sanitized.items()}
        config.num_labels = target_num_labels
        
        self.gtsrb_model = ViTForImageClassification.from_pretrained(
            model_name, config=config
        ).to(self.device).eval()
        print(f"Loaded GTSRB: {model_name}")
    
    def classify_single(self, image_path):
        """
        Returns: (shape, confidence, viewpoint, gtsrb_class_id)
        
        viewpoint is 'front' or 'back' based on combined signals.
        """
        # ── Geometry-based shape ──
        geo_shape, geo_conf = self.geometry_clf.classify_single(image_path)
        
        # ── GTSRB-based shape ──
        self._load_gtsrb()
        
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.gtsrb_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.gtsrb_model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                gtsrb_conf, gtsrb_pred = probs.max(dim=1)
                gtsrb_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            
            gtsrb_class = gtsrb_pred.item()
            gtsrb_conf_val = gtsrb_conf.item()
            gtsrb_entropy_val = gtsrb_entropy.item()
            gtsrb_shape = GTSRB_TO_SHAPE.get(gtsrb_class, "other")
        except Exception:
            gtsrb_class = -1
            gtsrb_conf_val = 0.0
            gtsrb_entropy_val = 10.0
            gtsrb_shape = "other"
        
        # ── Back-view detection (from original pipeline) ──
        is_back = self._detect_back(image_path, gtsrb_conf_val, gtsrb_entropy_val)
        
        if is_back:
            return "sign_back", 0.85, "back", gtsrb_class
        
        # ── Combine geometry + GTSRB ──
        if geo_shape == gtsrb_shape and geo_shape != "other":
            # Agreement → high confidence
            combined_conf = min(0.95, (geo_conf + gtsrb_conf_val) / 2 + 0.15)
            return geo_shape, combined_conf, "front", gtsrb_class
        
        if geo_conf > 0.6 and gtsrb_conf_val < 0.5:
            # Geometry confident, GTSRB uncertain → trust geometry
            return geo_shape, geo_conf, "front", gtsrb_class
        
        if gtsrb_conf_val > 0.8 and geo_conf < 0.4:
            # GTSRB very confident, geometry uncertain → trust GTSRB
            return gtsrb_shape, gtsrb_conf_val * 0.8, "front", gtsrb_class
        
        # Neither strong → use geometry (more robust to domain shift)
        return geo_shape, max(geo_conf * 0.9, 0.3), "front", gtsrb_class
    
    def _detect_back(self, image_path, gtsrb_conf, gtsrb_entropy,
                     sat_threshold=35, min_color_frac=0.12):
        """Combined saturation + GTSRB confidence back detection."""
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        color_frac = np.sum(sat > sat_threshold) / sat.size
        sat_is_back = color_frac < min_color_frac
        gtsrb_confused = gtsrb_conf < 0.3 or gtsrb_entropy > 3.0
        
        return sat_is_back and gtsrb_confused


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

TRAFFIC_SIGN_LABELS = {
    "trafficsign", "traffic_sign", "traffic sign",
    "trafficsignal", "traffic_signal", "traffic signal",
}

def get_sign_mask(df):
    col = "Class" if "Class" in df.columns else "class"
    return df[col].astype(str).str.strip().str.lower().isin(TRAFFIC_SIGN_LABELS)


def label_train_with_shapes(csv_path, image_dir, output_path, device='cuda'):
    """
    Label training set with shape-based sign groups.
    Uses objectID majority vote (same as original pipeline).
    
    Output CSV will have columns:
        sign_shape:  circular / triangular / rectangular / etc.
        viewpoint:   front / back
        shape_confidence: float
    """
    df = pd.read_csv(csv_path)
    sign_mask = get_sign_mask(df)
    signs = df[sign_mask].copy()
    
    oid_col = "objectID" if "objectID" in signs.columns else "Corresponding Indexes"
    
    n_signs = len(signs)
    n_ids = signs[oid_col].nunique()
    print(f"Traffic signs: {n_signs} images, {n_ids} identities")
    
    # Classify all sign images
    print("\nClassifying shapes...")
    clf = HybridShapeClassifier(device=device)
    
    shapes = []
    confs = []
    viewpoints = []
    
    for _, row in tqdm(signs.iterrows(), total=n_signs, desc="Shape classification"):
        path = os.path.join(image_dir, row["imageName"])
        shape, conf, viewpoint, _ = clf.classify_single(path)
        shapes.append(shape)
        confs.append(conf)
        viewpoints.append(viewpoint)
    
    signs["sign_shape"] = shapes
    signs["shape_confidence"] = confs
    signs["viewpoint"] = viewpoints
    
    n_front = viewpoints.count("front")
    n_back = viewpoints.count("back")
    print(f"Front: {n_front}, Back: {n_back}")
    
    # Majority vote per identity (front views only)
    print("\nMajority vote per identity...")
    identity_shapes = {}
    
    for oid, group in signs.groupby(oid_col):
        fronts = group[group["viewpoint"] == "front"]
        front_shapes = [s for s in fronts["sign_shape"].tolist() 
                       if s not in WILDCARD_SHAPES]
        
        if not front_shapes:
            identity_shapes[oid] = "unknown"
            continue
        
        counter = Counter(front_shapes)
        top_shape, _ = counter.most_common(1)[0]
        identity_shapes[oid] = top_shape
    
    # Propagate identity shape to all images (including backs)
    signs["sign_type"] = signs[oid_col].map(identity_shapes)
    
    # Merge back into full dataframe
    for col in ["sign_type", "sign_shape", "viewpoint", "shape_confidence"]:
        df[col] = None
        df.loc[sign_mask, col] = signs[col].values
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    
    print("\nShape distribution (by identity):")
    for shape, count in Counter(identity_shapes.values()).most_common():
        print(f"  {shape:25s}: {count} identities")


def label_test_with_shapes(csv_path, image_dir, output_path, device='cuda'):
    """
    Label gallery/query with shape-based sign groups.
    Per-image classification (no objectID available).
    """
    df = pd.read_csv(csv_path)
    sign_mask = get_sign_mask(df)
    signs = df[sign_mask].copy()
    
    print(f"Traffic signs: {len(signs)} images")
    
    clf = HybridShapeClassifier(device=device)
    
    sign_types = []
    viewpoints = []
    confs = []
    
    for _, row in tqdm(signs.iterrows(), total=len(signs), desc="Shape classification"):
        path = os.path.join(image_dir, row["imageName"])
        shape, conf, viewpoint, _ = clf.classify_single(path)
        
        if viewpoint == "back":
            sign_types.append("sign_back")
        else:
            sign_types.append(shape)
        viewpoints.append(viewpoint)
        confs.append(conf)
    
    for col in ["sign_type", "viewpoint", "shape_confidence"]:
        df[col] = None
    df.loc[sign_mask, "sign_type"] = sign_types
    df.loc[sign_mask, "viewpoint"] = viewpoints
    df.loc[sign_mask, "shape_confidence"] = confs
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    
    print("\nType distribution:")
    for t, c in pd.Series(sign_types).value_counts().items():
        print(f"  {t:25s}: {c}")


# ═══════════════════════════════════════════════════════════════════════
# PROTOTYPE BUILDING WITH UNFUSED FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_shape_prototypes_unfused(
    csv_path, image_dir, 
    reid_checkpoint, config_file,
    output_path, min_samples=5
):
    """
    Build shape-based prototypes using UNFUSED (final-layer-only) PAT features.
    
    CRITICAL: Prototypes MUST be built from unfused features because:
    - Unfused (1024-dim): inter-prototype cosine sim ~0.5 → well-separated
    - Fused (8192-dim): inter-prototype cosine sim >0.8 → collapsed
    
    At inference time:
    - Use UNFUSED features to route query → shape prototype
    - Use FUSED features for actual retrieval within the filtered subset
    
    This is already supported by the --proto_use_unfused flag in your
    inference script (update.py / inference.py).
    """
    from torchvision import transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(csv_path)
    sign_mask = get_sign_mask(df)
    signs = df[sign_mask].copy()
    
    # Only front views with valid shapes
    front_signs = signs[
        (signs["viewpoint"] == "front") &
        (signs["sign_type"].notna()) &
        (~signs["sign_type"].isin(WILDCARD_SHAPES))
    ].copy()
    
    print(f"Front-view signs for prototypes: {len(front_signs)}")
    print(f"Shape distribution:")
    for shape, count in front_signs["sign_type"].value_counts().items():
        print(f"  {shape:20s}: {count}")
    
    # ── Load PAT model WITHOUT CLS fusion ──
    # This is the key: prototypes are built from unfused features
    from config import cfg as pat_cfg
    
    if config_file and os.path.exists(config_file):
        pat_cfg.merge_from_file(config_file)
    
    # Force CLS fusion OFF for prototype extraction
    pat_cfg.defrost()
    pat_cfg.TEST.CLS_FUSION = False
    pat_cfg.TEST.CLS_FUSION_LAST = 1
    if reid_checkpoint:
        pat_cfg.TEST.WEIGHT = reid_checkpoint
    pat_cfg.freeze()
    
    from model import make_model
    model = make_model(pat_cfg, pat_cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(pat_cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()
    print(f"Loaded PAT model (CLS fusion DISABLED for prototypes)")
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # ── Extract unfused features per shape ──
    prototypes = {}
    metadata = {}
    
    for shape, group in front_signs.groupby("sign_type"):
        if len(group) < min_samples:
            print(f"  Skipping {shape}: only {len(group)} samples")
            continue
        
        print(f"  Extracting for {shape} ({len(group)} images)...")
        features = []
        
        paths = [os.path.join(image_dir, n) for n in group["imageName"]]
        batch_size = 16
        
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(transform(img))
                except Exception:
                    imgs.append(torch.zeros(3, 256, 128))
            
            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                # Test-time flip augmentation (same as inference.py)
                ff = None
                for flip in [False, True]:
                    inp = torch.flip(batch, dims=[-1]) if flip else batch
                    out = model(inp)
                    f = out.float()
                    if ff is None:
                        ff = torch.zeros_like(f)
                    ff = ff + f
                
                # L2 normalize
                ff = F.normalize(ff, dim=1, p=2)
            
            features.append(ff.cpu())
        
        if not features:
            continue
        
        all_feats = torch.cat(features, dim=0)
        all_feats = F.normalize(all_feats, dim=1)
        
        # Prototype = mean of normalized features, then re-normalize
        prototype = all_feats.mean(dim=0)
        prototype = F.normalize(prototype, dim=0)
        
        prototypes[shape] = prototype
        
        oid_col = "objectID" if "objectID" in group.columns else "Corresponding Indexes"
        metadata[shape] = {
            "n_images": len(group),
            "n_identities": group[oid_col].nunique() if oid_col in group.columns else "N/A",
            "feature_dim": prototype.shape[0],
        }
    
    # ── Validate separation ──
    print("\n" + "=" * 60)
    print("PROTOTYPE VALIDATION (UNFUSED FEATURES)")
    print("=" * 60)
    
    types = list(prototypes.keys())
    if len(types) >= 2:
        print("\nInter-prototype cosine similarities:")
        print("  (< 0.5 = excellent, < 0.7 = good, > 0.8 = problematic)")
        max_sim = 0
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                sim = torch.dot(prototypes[types[i]], prototypes[types[j]]).item()
                flag = ""
                if sim > 0.8:
                    flag = " ⚠️  OVERLAP"
                elif sim < 0.5:
                    flag = " ✅ WELL SEPARATED"
                print(f"  {types[i]:20s} ↔ {types[j]:20s}: {sim:.4f}{flag}")
                max_sim = max(max_sim, sim)
        
        if max_sim < 0.7:
            print(f"\n✅ All prototypes well-separated (max sim: {max_sim:.4f})")
            print("   Shape-based filtering should work effectively!")
        else:
            print(f"\n⚠️  Some prototypes overlap (max sim: {max_sim:.4f})")
            print("   Consider merging overlapping shape groups.")
    
    # ── Save ──
    torch.save({
        "prototypes": prototypes,
        "metadata": metadata,
        "grouping": "shape",  # flag that this uses shape groups
        "cls_fusion": False,  # flag that these are unfused features
    }, output_path)
    
    print(f"\nSaved shape prototype bank → {output_path}")
    print(f"Groups: {list(prototypes.keys())}")
    print(f"Feature dim: {list(metadata.values())[0]['feature_dim'] if metadata else 'N/A'}")
    print(f"\nWildcard types (flat retrieval at inference):")
    for wt in sorted(WILDCARD_SHAPES):
        print(f"  - '{wt}'")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shape-based sign grouping")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    p1 = sub.add_parser("label_train", help="Label training signs with shape groups")
    p1.add_argument("--csv", required=True, help="train_classes.csv")
    p1.add_argument("--image_dir", required=True, help="image_train/")
    p1.add_argument("--output", required=True, help="Output CSV path")
    p1.add_argument("--device", default="cuda")
    
    p2 = sub.add_parser("label_test", help="Label gallery/query signs with shape groups")
    p2.add_argument("--csv", required=True, help="test_classes.csv or query_classes.csv")
    p2.add_argument("--image_dir", required=True, help="image_test/ or image_query/")
    p2.add_argument("--output", required=True, help="Output CSV path")
    p2.add_argument("--device", default="cuda")
    
    p3 = sub.add_parser("build_prototypes", help="Build shape prototypes from unfused PAT features")
    p3.add_argument("--csv", required=True, help="train_sign_types.csv (output of label_train)")
    p3.add_argument("--image_dir", required=True, help="image_train/")
    p3.add_argument("--reid_checkpoint", required=True, help="PAT model .pth")
    p3.add_argument("--config_file", default=None, help="PAT config .yml")
    p3.add_argument("--output", required=True, help="Output .pt path")
    p3.add_argument("--min_samples", type=int, default=5)
    
    p4 = sub.add_parser("test_shape", help="Test shape classifier on a single image")
    p4.add_argument("--image", required=True)
    
    args = parser.parse_args()
    
    if args.cmd == "label_train":
        label_train_with_shapes(args.csv, args.image_dir, args.output, args.device)
    elif args.cmd == "label_test":
        label_test_with_shapes(args.csv, args.image_dir, args.output, args.device)
    elif args.cmd == "build_prototypes":
        build_shape_prototypes_unfused(
            args.csv, args.image_dir, args.reid_checkpoint,
            args.config_file, args.output, args.min_samples
        )
    elif args.cmd == "test_shape":
        clf = ShapeClassifier()
        shape, conf = clf.classify_single(args.image)
        print(f"Shape: {shape} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()