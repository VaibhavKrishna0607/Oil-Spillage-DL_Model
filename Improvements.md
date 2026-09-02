
## 📋 COMPREHENSIVE ANALYSIS & IMPLEMENTATION ROADMAP

I've reviewed your project, the reviews, and the Playbook. Here's the critical gap analysis and what must be built **before** training even starts:

---

## **PART 1: What's Broken in Your Current Pipeline**

### **1. Naive Split Strategy (HIGH PRIORITY FIX)**
Your `prepare_splits.py` does random shuffling, but the Playbook **explicitly warns** this leaks scenes across train/val/test.

**The Problem:**
- You have 8,070 patches from only **21 source scenes** (~380 patches/scene)
- Random split almost certainly puts crops from the same SAR scene in both train and test
- Model memorizes scene-specific noise patterns → validation IoU looks great but fails on genuinely new scenes

**Current Code Issue** (in `prepare_splits.py`):
```python
rng.shuffle(all_train)  # ❌ Random shuffle of filenames
split_train = all_train[:n_train]
split_test = all_train[n_train:]
```

**Needed Fix:**
```python
# ✓ Group-aware split using scene/cluster ID
from sklearn.model_selection import GroupShuffleSplit

# First, extract scene IDs from filenames or cluster by similarity
scene_ids = extract_scene_ids(filenames)  # or cluster_by_similarity()

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
for train_idx, test_idx in splitter.split(X, groups=scene_ids):
    split_train = [all_train[i] for i in train_idx]
    split_test = [all_train[i] for i in test_idx]
```

**Impact:** This single fix ensures your IoU numbers are real, not inflated by data leakage.

---

### **2. Missing Data Cleaning Pipeline (CRITICAL)**
The Playbook defines **8 stages** (Sections 4). You have **0** implemented.

**Stage 1 — Integrity Check:**
- No verification that every image has a matching mask
- No detection of corrupt files
- No dimension mismatch detection

**Stage 2 — Mask QC:**
- No outlier detection (masks with 0% or >70% oil are usually mislabeled)
- No morphological cleaning (1–4 pixel noise specks should be removed)
- No manual review grid generation

**Stage 3 — Radiometric Normalization (Leakage Bug):**
Your code uses:
```python
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats — WRONG for SAR
IMG_STD  = [0.229, 0.224, 0.225]
```

**This is a domain mismatch bug.** SAR data is single-channel dB backscatter, not RGB natural images. 

**Needed:**
```python
# Compute from training split only (not entire dataset!)
# 1. Load all training masks → find oil pixel distribution
# 2. Compute clip bound at 98th percentile
# 3. Apply min-max or z-score normalization
# 4. Save clip_bound, mean, std to norm_stats.json
# 5. Apply FIXED stats to val/test (no recomputation)
```

**Stage 4–7:** Scene grouping, class imbalance quantification, format standardization, geometric properties

---

### **3. No Geometric Properties Ground Truth (BLOCKS P2 + P3)**
The review requires area, perimeter, eccentricity, centroid — **extracted from the ground-truth masks BEFORE training**.

**Current gap:** Your masks are just binary arrays. You need a CSV:

```
filename,split,scene_id,area_px,perimeter_px,eccentricity,centroid_row,centroid_col,bbox_x_min,bbox_y_min,bbox_x_max,bbox_y_max
palsar_001.png,train,scene_A,2541,189,0.82,128,137,85,102,171,163
...
```

This feeds directly to:
- **P2's inference pipeline** — validate post-processing produces correct values
- **P3's drift model** — use centroid as anchor point for hindcasting
- **PPT metrics** — "mean spill area: X px² / Y km²" backed by ground truth

---

### **4. Dataset Underutilization**
Playbook Section 3 recommends **two-track strategy:**

| Track | Dataset | Size | Purpose |
|-------|---------|------|---------|
| **A** | SOS Refined (Zenodo 15298010) | 256×256 | Core training |
| **B** | Sentinel-1 SAR Oil-spill (Zenodo 8346860+) | 2,048×2,048 | Scale realism, sliding-window test |

**You're only using Track A.** Track B is *critical* for testing if your 256×256 model generalizes to real full scenes (the review flagged this as a blocker).

---

### **5. No Augmentation Validation**
Your code mentions augmentations exist but doesn't show the pipeline. **Verify you're using `albumentations` with SAR-specific transforms:**

```python
# ✓ Correct approach for SAR
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.GaussNoise(p=0.3),  # Realistic SAR speckle
    A.RandomBrightnessContrast(p=0.3),  # Radiometric jitter
], bbox_params=A.BboxParams(format='pascal_voc'))  # Preserve mask

# ❌ Wrong approach for SAR (torchvision ColorJitter, etc.)
```

---

## **PART 2: Step-by-Step Implementation Roadmap (Next 3 Days)**

### **Day 1: Data Validation & Integrity**

**Checkpoint 1a — Verify Dataset:**
```python
# Quick check: What's the actual patch size?
from PIL import Image
import os

img_path = 'Images/images/train/' + os.listdir('Images/images/train/')[0]
img = Image.open(img_path)
print(f"Patch size: {img.size}")  # Expected: (256, 256) or (416, 416)?
```

**Checkpoint 1b — Run Full Integrity Pipeline:**
1. Load all image/mask pairs
2. Log:
   - Total files found
   - Corrupt files (can't open)
   - Dimension mismatches
   - Missing masks
   - Filename patterns (extract sensor type, scene ID)
3. Output: `artifacts/integrity_report.csv` + `artifacts/integrity_report.txt`

**Code template:**
```python
# Stage 1: Integrity Check
for split in ['train', 'val']:
    img_dir = f'Images/images/{split}'
    mask_dir = f'Masks/masks/{split}'
    
    for fname in os.listdir(img_dir):
        try:
            img = Image.open(os.path.join(img_dir, fname))
            mask = Image.open(os.path.join(mask_dir, fname))
            
            if img.size != mask.size:
                log_error(f"{fname}: dimension mismatch")
            if img.mode != 'RGB' or mask.mode != 'L':
                log_error(f"{fname}: unexpected mode {img.mode}/{mask.mode}")
                
        except Exception as e:
            log_error(f"{fname}: {e}")

# Save report
pd.DataFrame(errors).to_csv('artifacts/integrity_report.csv')
```

### **Day 2: Mask QC + Normalization Stats**

**Checkpoint 2a — Mask Quality Assessment:**
1. For each valid mask:
   - Compute oil pixel fraction
   - Flag if < 0.5% or > 70% (likely mislabeled)
   - Apply morphological opening to remove < 4-pixel speckles
   - Compute actual bounding box

2. Output: `artifacts/mask_qc_report.csv` with flagged/cleaned masks

**Checkpoint 2b — Compute Normalization Stats (Training Split Only!):**
```python
# Load all training images as float32
train_images = []  # (N, 256, 256, 1)
for fname in split_train:
    img = Image.open(f'Images/images/train/{fname}').convert('L')
    train_images.append(np.array(img, dtype=np.float32))

train_images = np.stack(train_images)  # (N, 256, 256)

# Compute clip bound from oil-region backscatter
# (find 98th percentile, use as upper clip)
clip_98 = np.percentile(train_images, 98)
clip_2 = np.percentile(train_images, 2)

# Normalize
train_images_clipped = np.clip(train_images, clip_2, clip_98)
mean_val = train_images_clipped.mean()
std_val = train_images_clipped.std()

# Save (apply to val/test identically)
norm_stats = {
    'clip_2': float(clip_2),
    'clip_98': float(clip_98),
    'mean': float(mean_val),
    'std': float(std_val),
}
json.dump(norm_stats, open('artifacts/norm_stats.json', 'w'))
```

**Checkpoint 2c — Manual Overlay Eyeball Check:**
Generate a grid of 30 random image/mask overlays and inspect. This 5-minute check catches things no script will.

### **Day 3: Scene Grouping + Split Generation + Geometric Properties + Tensor Packing**

**Checkpoint 3a — Extract Scene IDs:**
```python
import re

def extract_scene_id(filename):
    # SOS dataset filenames often have pattern like "palsar_XXXXX.png" or "sentinel_XXXXX.png"
    # If filenames are anonymized, cluster by pixel similarity
    
    # Option 1: Filename pattern
    match = re.match(r'(\w+?)_(\d+)', filename)
    if match:
        return match.group(1) + '_' + match.group(2)[:3]  # "palsar_001" -> "palsar_00"
    
    # Option 2: If filenames are anonymous, use perceptual hashing
    # (cluster images with similar pixel distribution)
    # This is more complex; use scikit-learn's AgglomerativeClustering

all_files = os.listdir('Images/images/train')
scene_ids = [extract_scene_id(f) for f in all_files]
```

**Checkpoint 3b — Group-Aware Split:**
```python
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

# Create dataframe
df = pd.DataFrame({
    'filename': all_files,
    'scene_id': scene_ids,
})

# Split preserving scene groups
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(df, groups=df['scene_id']):
    split_train_files = df.iloc[train_idx]['filename'].tolist()
    split_test_files = df.iloc[test_idx]['filename'].tolist()

# Keep original val intact
split_val_files = os.listdir('Images/images/val')

# Write to split files
with open('split_train.txt', 'w') as f:
    f.write('\n'.join(split_train_files))
with open('split_test.txt', 'w') as f:
    f.write('\n'.join(split_test_files))
with open('split_val.txt', 'w') as f:
    f.write('\n'.join(split_val_files))
```

**Checkpoint 3c — Geometric Properties Ground Truth:**
```python
from skimage import measure
import csv

geometry_rows = []

for split, files in [('train', split_train_files), ('val', split_val_files), ('test', split_test_files)]:
    mask_dir = f'Masks/masks/{split if split != "test" else "train"}'
    
    for fname in files:
        mask = Image.open(os.path.join(mask_dir, fname)).convert('L')
        mask_array = np.array(mask) > 127
        
        labeled = measure.label(mask_array > 0)
        props = measure.regionprops(labeled)
        
        if props:
            largest = max(props, key=lambda p: p.area)
            geometry_rows.append({
                'filename': fname,
                'split': split,
                'area_px': largest.area,
                'perimeter_px': largest.perimeter,
                'eccentricity': largest.eccentricity,
                'centroid_row': largest.centroid[0],
                'centroid_col': largest.centroid[1],
                'bbox_x_min': largest.bbox[1],
                'bbox_y_min': largest.bbox[0],
                'bbox_x_max': largest.bbox[3],
                'bbox_y_max': largest.bbox[2],
            })

pd.DataFrame(geometry_rows).to_csv('artifacts/geometry.csv', index=False)
```

**Checkpoint 3d — Pack Tensors:**
```python
import torch

# Load and normalize all images for each split
def load_split_tensors(filenames, norm_stats, split_type='train'):
    images, masks = [], []
    mask_dir = f'Masks/masks/{split_type if split_type != "test" else "train"}'
    img_dir = f'Images/images/{split_type if split_type != "test" else "train"}'
    
    for fname in filenames:
        img = Image.open(os.path.join(img_dir, fname)).convert('L')
        mask = Image.open(os.path.join(mask_dir, fname)).convert('L')
        
        img_arr = np.array(img, dtype=np.float32)
        # Apply normalization
        img_arr = np.clip(img_arr, norm_stats['clip_2'], norm_stats['clip_98'])
        img_arr = (img_arr - norm_stats['mean']) / (norm_stats['std'] + 1e-7)
        
        mask_arr = (np.array(mask) > 127).astype(np.uint8)
        
        images.append(torch.from_numpy(img_arr[None, ...]))  # Add channel
        masks.append(torch.from_numpy(mask_arr[None, ...]))
    
    return torch.stack(images), torch.stack(masks)

# Save
for split, files in [('train', split_train_files), ('val', split_val_files), ('test', split_test_files)]:
    imgs, msks = load_split_tensors(files, norm_stats, split)
    torch.save(imgs, f'dataset_ready/{split}_images.pt')
    torch.save(msks, f'dataset_ready/{split}_masks.pt')

# Save metadata
json.dump(norm_stats, open('dataset_ready/norm_stats.json', 'w'))
pd.DataFrame(geometry_rows).to_csv('dataset_ready/geometry.csv', index=False)
```

---

## **PART 3: Updated `dataset.py` for Post-Foundation Era**

Once tensors are packed, your loader becomes much simpler:

```python
class SARSpillDataset(Dataset):
    def __init__(self, split='train', augment=False):
        self.images = torch.load(f'dataset_ready/{split}_images.pt')
        self.masks = torch.load(f'dataset_ready/{split}_masks.pt')
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc'))
    
    def __getitem__(self, idx):
        img = self.images[idx].squeeze(0).numpy()  # (256, 256)
        mask = self.masks[idx].squeeze(0).numpy()
        
        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
```

---

## **PART 4: Critical Dependencies to Install**

```bash
pip install scikit-image scikit-learn albumentations pandas rasterio
```

---

## **PART 5: What P2 Needs to Add to `predict.py` (Geometric Properties Post-Processing)**

```python
def extract_geometric_properties(mask_pred, threshold=0.5):
    """Extract measurable properties from predicted mask."""
    binary = (mask_pred > threshold).astype(np.uint8)
    labeled = measure.label(binary > 0)
    props = measure.regionprops(labeled)
    
    if not props:
        return None
    
    largest = max(props, key=lambda p: p.area)
    return {
        'area_px': float(largest.area),
        'perimeter_px': float(largest.perimeter),
        'eccentricity': float(largest.eccentricity),
        'centroid_row': float(largest.centroid[0]),
        'centroid_col': float(largest.centroid[1]),
        'bbox': [largest.bbox[1], largest.bbox[0], largest.bbox[3], largest.bbox[2]],
    }
```

This output goes directly to the drift model (P3) and Mapbox visualization (P5).

---

## **CRITICAL SUCCESS CRITERIA (Before Day 4)**

- [ ] `artifacts/integrity_report.csv` — zero corrupt files or dimension mismatches
- [ ] `artifacts/mask_qc_report.csv` — all flagged outliers reviewed & cleaned  
- [ ] `artifacts/norm_stats.json` — computed on train split only
- [ ] `split_train.txt`, `split_val.txt`, `split_test.txt` — generated via GroupShuffleSplit
- [ ] `artifacts/geometry.csv` — ground-truth area/perimeter/eccentricity for all masks
- [ ] `dataset_ready/{train,val,test}_{images,masks}.pt` — packed tensors ready to load
- [ ] Manual overlay grid reviewed and approved (screenshot for PPT)
- [ ] `qc_report.md` auto-generated with statistics

**Once these are done, P2 can train without fear of leakage or data corruption.**

---

## **Why This Matters for the Review**

✓ **Fixes Issue #1–3:** Geometric properties now available for drift model  
✓ **Fixes Issue #4 (partial):** Georeferencing foundation for Track B  
✓ **Fixes Issue #7:** Sliding-window test on Track B starts Day 4  
✓ **Eliminates data leakage:** Group-aware split + proper normalization  
✓ **Provides evidence:** QC report is your paper trail on the PPT

This is the foundation that P2/P3/P4/P5 all depend on. Get this right, and the rest cascades.
