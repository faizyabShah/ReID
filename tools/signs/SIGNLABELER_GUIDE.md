# SignLabeler Pipeline Guide

## ✅ Code Updated for ViT-Large-Patch16

Your PAT model backbone has been fully integrated into `signlabeler.py`:

**Backbone Specifications:**
- **Architecture**: vit_large_patch16_224_TransReID
- **Feature Dimension**: 1024
- **Input Size**: 256 × 128 (PAT training resolution)
- **Normalization**: Mean [0.5, 0.5, 0.5], Std [0.5, 0.5, 0.5]
- **Feature Output**: CLS token (first token from ViT sequence)

**Checkpoint Handling:**
- Supports `'model'` or `'state_dict'` keys
- Automatically removes DDP prefix (`'module.'`)
- Filters out classifier and bottleneck layers

---

## 📋 Pipeline Overview: 4 Steps

```
Step 1: Label Training Set
  ↓
Step 2: Label Gallery/Query
  ↓
Step 3: Build Prototypes (using your PAT model)
  ↓
Step 4: Sanity Check Visualizations
```

---

## 🚀 Step 1: Label Your Training Set

**Purpose**: Classify all training images (fronts/backs) + propagate identity labels

**Command**:
```bash
python signlabeler.py label_train \
    --csv train_classes.csv \
    --image_dir image_train \
    --output train_sign_types.csv
```

**Output**: 
- `train_sign_types.csv`: Contains columns:
  - `imageName`: path to image
  - `objectID`: identity (from "Corresponding Indexes")
  - `Class`: original class label
  - `sign_type`: coarse type (prohibitory/warning/mandatory/stop/yield/priority/derestriction/unknown)
  - `viewpoint`: "front" or "back"
  - `gtsrb_confidence`: GTSRB classifier confidence
  - `gtsrb_entropy`: classifier uncertainty
  - `back_confidence`: confidence of back detection

- `train_sign_types_review.csv`: Identities with low confidence or no front views (review these manually!)

**What happens**:
1. GTSRB ViT classifier runs on all images → 43 classes → mapped to 7 coarse types
2. Combined back detector (saturation + GTSRB-uncertainty) labels each image as front/back
3. Per identity: **majority vote using ONLY fronts** → identity gets a type
4. All images (fronts + backs) inherit the identity's type

**Key Logic**: 
- ✅ Only fronts vote on identity type
- ✅ Backs inherit the type (no prototype pollution later)
- ✅ "unknown" types get wildcard treatment at inference

---

## 🚀 Step 2: Label Gallery/Query Set

**Purpose**: Prepare gallery/query images for retrieval

**Command**:
```bash
python signlabeler.py label_test \
    --csv test_classes.csv \
    --image_dir image_test \
    --output test_sign_types.csv
```

**Output**: 
- `test_sign_types.csv`: Contains:
  - Back views: `sign_type = "sign_back"` (wildcard)
  - Front views: `sign_type = [GTSRB→coarse mapping]` (can be "unknown")

**Key Logic**:
- ✅ Backs are marked as WILDCARD (will skip prototype filter at inference)
- ✅ Fronts are classified independently (no identity propagation)
- ✅ "sign_back" and "unknown" types will bypass prototype filtering during retrieval

---

## 🚀 Step 3: Build Prototypes

**Purpose**: Compute per-type feature prototypes using your PAT model

**Command**:
```bash
python signlabeler.py build_prototypes \
    --csv train_sign_types.csv \
    --image_dir image_train \
    --reid_checkpoint path/to/your_pat_model.pth \
    --reid_backbone vit_large \
    --output prototype_bank.pt \
    --min_samples 10
```

**Parameters**:
- `--reid_checkpoint`: Path to your trained PAT model checkpoint
- `--reid_backbone`: `vit_large` (your model) | `vit_base` | `resnet50`
- `--min_samples`: Minimum front views needed to build a prototype (default: 10)

**Output**: 
- `prototype_bank.pt`: Dictionary containing:
  ```python
  {
    "prototypes": {
      "prohibitory": tensor([1024]),   # L2-normalized mean of all front prohibitory features
      "warning": tensor([1024]),
      "mandatory": tensor([1024]),
      "stop": tensor([1024]),
      "yield": tensor([1024]),
      "priority": tensor([1024]),
      "derestriction": tensor([1024]),
    },
    "metadata": {
      "prohibitory": {"n_images": 250, "n_identities": 42},
      ...
    },
    "feature_dim": 1024,
  }
  ```

**What happens**:
1. Filters training data: **fronts only** + NOT in {sign_back, unknown}
2. For each sign type:
   - Extracts PAT embeddings from all front images
   - L2-normalizes each embedding
   - Computes mean embedding per type
   - L2-normalizes the prototype
3. Validates: prints inter-prototype cosine similarities
4. Saves the prototype bank

**Quality Check** (console output):
```
Inter-prototype cosine similarities:
  prohibitory ↔ warning:   0.523  ✓ well separated
  prohibitory ↔ stop:      0.891  ⚠️ OVERLAP (too close!)
```
- If similarities < 0.7: good separation
- If similarities > 0.85: types might be confused (consider merging or reviewing data)

---

## 🚀 Step 4: Sanity Check

**Purpose**: Visual verification of classifications

**Command**:
```bash
python signlabeler.py sanity_check \
    --csv train_sign_types.csv \
    --image_dir image_train \
    --output_dir sanity_check_output
```

**Output**: 
- Grid images: `sanity_check_output/{sign_type}_{viewpoint}.png`
- Shows 8 random samples per (type, viewpoint) combination

**What to look for**:
- ✅ "front" images should have colored, clearly visible signs
- ✅ "back" images should have gray, low-saturation backs
- ⚠️ Misclassified fronts mixed in "back" → tune back detection thresholds
- ⚠️ Fronts classified to wrong type → GTSRB mapping may need adjustment

---

## 🔄 Retrieval: Using the Prototypes

Once you have `prototype_bank.pt`, here's how to use it for Re-ID retrieval:

```python
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from signlabeler import retrieve_ranked

# Load prototype bank
pb = torch.load("prototype_bank.pt")

# Load gallery metadata
gallery_csv = pd.read_csv("test_sign_types.csv")
gallery_types = gallery_csv["sign_type"].tolist()

# Extract gallery embeddings (your PAT model on all gallery images)
# gallery_embeddings: tensor [N_gallery, 1024]

# Extract a query embedding
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def extract_query_embedding(image_path, pat_model, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = pat_model(img_tensor)
        if feat.dim() == 3:
            feat = feat[:, 0, :]  # CLS token
    return F.normalize(feat.squeeze(0), dim=0)

# Query: back view (should use WILDCARD skip)
query_emb = extract_query_embedding("some_back_image.jpg", pat_model, device)
ranked_indices, filter_mode = retrieve_ranked(
    query_emb,
    "sign_back",  # Marked as back during label_test
    gallery_embeddings,
    gallery_types,
    pb,
    top_k=100
)
print(f"Filter mode: {filter_mode}")  # Should print "wildcard"
print(f"Top 10 matches: {ranked_indices[:10]}")

# Query: front view of stop sign (should use PROTOTYPE filter)
query_emb = extract_query_embedding("some_stop_sign_front.jpg", pat_model, device)
ranked_indices, filter_mode = retrieve_ranked(
    query_emb,
    "stop",
    gallery_embeddings,
    gallery_types,
    pb,
    top_k=100
)
print(f"Filter mode: {filter_mode}")  # Should print "prototype"
```

---

## 🔧 Back-Detection Threshold Tuning

If back detection isn't working well, check the defaults in `detect_back_combined()`:

```python
def detect_back_combined(image_path, gtsrb_confidence, gtsrb_entropy,
                         sat_threshold=35,        # ← Saturation threshold (0-255)
                         min_color_frac=0.12,     # ← Min fraction of colored pixels
                         min_gtsrb_conf=0.3,      # ← Min GTSRB confidence
                         max_gtsrb_entropy=3.0):  # ← Max GTSRB uncertainty
```

**Tuning strategy**:
1. Run `label_train` on a small test set (100 images)
2. Visually inspect the review file and sanity check output
3. If backs are missed (marked as fronts):
   - Lower `sat_threshold` (e.g., 30 → 25)
   - Lower `min_color_frac` (e.g., 0.12 → 0.08)
4. If fronts are misdetected as backs:
   - Raise `sat_threshold` (e.g., 35 → 40)
   - Raise `min_color_frac` (e.g., 0.12 → 0.15)

---

## 📊 Expected Output Structure

After all 4 steps, you should have:

```
.
├── train_sign_types.csv              # Step 1 output
├── train_sign_types_review.csv       # Step 1 review items
├── test_sign_types.csv               # Step 2 output
├── prototype_bank.pt                 # Step 3 output
└── sanity_check_output/              # Step 4 output
    ├── prohibitory_front.png
    ├── prohibitory_back.png
    ├── warning_front.png
    ├── warning_back.png
    └── ...
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Too many unknowns | GTSRB uncertain | Lower `max_gtsrb_entropy` in `detect_back_combined` |
| Backs detected as fronts | Low saturation threshold too high | Lower `sat_threshold` |
| Fronts detected as backs | Too aggressive back detection | Raise `sat_threshold` or `min_color_frac` |
| Prototypes too similar | Type confusion in GTSRB mapping | Review `GTSRB_TO_COARSE` mapping or merge types |
| Out of memory | Too many images per batch | Lower `batch_size` in `build_prototypes` |

---

## 🎯 Next Steps

1. ✅ Run **Step 1**: `label_train` on a small test set (100-200 images)
2. ✅ Check the review file and sanity check output
3. ✅ Tune back-detection thresholds if needed
4. ✅ Run **Step 1** on full training set
5. ✅ Run **Step 2**: `label_test` on your gallery/query set
6. ✅ Run **Step 3**: `build_prototypes` with your PAT model
7. ✅ Check inter-prototype similarities
8. ✅ Run **Step 4**: `sanity_check` on training set
9. ✅ Implement retrieval loop using `retrieve_ranked()`

---

## 📝 Notes

- **ViT feature dim**: 1024 (set in code, don't change manually)
- **Normalization**: Both mean and std are [0.5, 0.5, 0.5]
- **Back-skip logic**: Implemented in `prototype_filter()` as Path 1
- **Gallery-back inclusion**: Implemented in `prototype_filter()` Path 5
- **Checkpoint loading**: Handles both `'model'` and `'state_dict'` keys automatically
