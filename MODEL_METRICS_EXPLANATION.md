# 📊 Complete Guide: Training Metrics & Dataset Parameters

## 🚀 Training Metrics (Real-time Output)

When your model trains, you see this table updating every epoch:

```
Epoch   Train Loss   Train Dice   Val Loss   Val IoU   Val Dice   Threshold   LR
1       0.0113      0.6190       0.4419     0.3882    0.4839     0.60        2e-05
2       0.0118      0.6179       0.4389     0.3918    0.4864     0.60        2e-05
```

### 🎯 Key Metrics Breakdown:

| **Metric** | **What it means** | **Range** | **Target** | **Current Status** |
|------------|-------------------|-----------|------------|-------------------|
| **Epoch** | Training round number | 1-20 | Progress tracker | 2/20 completed |
| **Train Loss** | How wrong the model is on training data | 0.0-∞ | **< 0.01** | 0.011 (needs improvement) |
| **Train Dice** | Training accuracy (overlap measure) | 0.0-1.0 | **> 0.65** | 0.619 (close!) |
| **Val Loss** | How wrong on unseen validation data | 0.0-∞ | **< 0.40** | 0.44 (needs improvement) |
| **Val IoU** | Intersection over Union on validation | 0.0-1.0 | **> 0.40** | 0.39 (very close!) |
| **Val Dice** | **🔥 MAIN METRIC** - Real accuracy | 0.0-1.0 | **> 0.95** | 0.486 (48.6%) |
| **Threshold** | Binary classification cutoff | 0.0-1.0 | Auto-optimized | 0.60 |
| **LR** | Learning rate (adaptation speed) | 1e-6 to 1e-3 | Adaptive | 2e-05 |

### 🎯 **Val Dice is Your Success Metric**
- **Current**: 48.6% (0.486)
- **Previous Best**: 57.6% (dataset2_improved.pth)
- **Your Target**: **95%+ for deployment**
- **Minimum Acceptable**: 80% (your stated minimum)

---

## 📁 Dataset 1: SOS (Original) - `Dataset-1/`

### Dataset Overview:
- **Images**: 8,070 tiles (6,455 train + 1,615 val)
- **Source**: PALSAR satellite sensor
- **Format**: 256×256 RGB images + RGB masks
- **Status**: Achieved 83% accuracy (ready for deployment)

### Dataset Manifest Columns:

| **Column** | **Description** | **Example** | **Use Case** |
|------------|-----------------|-------------|--------------|
| `split` | train/val division | "train" | Data organization |
| `filename` | Image identifier | "palsar_0.png" | File tracking |
| `image_path` | Path to SAR image | "Images/images/train/" | Loading images |
| `mask_path` | Path to oil mask | "Masks/masks/train/" | Ground truth |
| `sensor` | Satellite sensor type | "palsar" | Data source tracking |
| `status` | Quality check result | "valid" | Data integrity |
| `width/height` | Image dimensions | 256, 256 | Preprocessing |
| `image_mode` | Color format | "RGB" | Technical spec |
| `oil_pixels` | Number of oil pixels | 2272 | Spill size |
| `oil_fraction` | % of image with oil | 0.03466797 | Spill density |
| `bbox_*` | Bounding box coords | x_min, y_min, x_max, y_max | Object location |
| `*_sha256` | File integrity hash | "3b7c21fb..." | Corruption detection |

### Geometric Features:

| **Column** | **Description** | **Range** | **Importance** |
|------------|-----------------|-----------|----------------|
| `area_px` | Oil spill area in pixels | 660-8810 | Spill size |
| `perimeter_px` | Oil boundary length | 140-1140 | Shape complexity |
| `eccentricity` | How elongated (0=circle, 1=line) | 0.83-0.99 | Shape analysis |
| `centroid_row/col` | Center point coordinates | 0-256 | Spill location |
| `bbox_*` | Minimum rectangle around oil | coordinates | Spatial extent |

---

## 📁 Dataset 2: Krestenitis (Current Training) - `Oil Spill Detection Dataset/`

### Dataset Overview:
- **Images**: 1,112 scenes (1,002 train + 110 test)
- **Source**: SAR satellite (multi-sensor)
- **Format**: JPG images + PNG RGB masks
- **Status**: Training in progress, current 48.6% → target 95%+

### Image Categories:
- **Total tiles**: 112,224 generated from 1,002 scenes
- **Informative tiles**: 49,966 (containing oil/look-alikes)
- **Current training**: 49,966 train + 1,565 validation

### RGB Mask Color Coding:

| **Color** | **RGB Value** | **Meaning** | **Model Target** |
|-----------|---------------|-------------|------------------|
| Black | (0, 0, 0) | Sea Surface | Background (ignore) |
| **Cyan** | (0, 255, 255) | **Oil Spill** | **PRIMARY TARGET** |
| Red | (255, 0, 0) | Look-alike | False positive (avoid) |
| Brown | (153, 76, 0) | Ship | Secondary object |
| Green | Various | Land | Background |

### Why Krestenitis is Harder:
1. **More complex**: 5 categories vs 2
2. **Look-alikes**: Red pixels that look like oil but aren't
3. **Multi-sensor**: Different SAR characteristics
4. **Larger scale**: 1,002 full scenes vs preprocessed tiles

---

## 🔄 Current Training Status

### What's Happening Now:
```
Option C (Aggressive): 
- Batch size: 8 (more updates)
- Learning rate: 2e-5 (aggressive)
- Epochs: 20 (with early stopping)
- Patience: 10 (stop if no improvement)
```

### Progress Tracking:
- **Completed**: 2/20 epochs
- **Current Val Dice**: 48.6%
- **Best Historical**: 57.6%
- **Target**: 95%+
- **Time remaining**: ~18 epochs × 30min = 9 hours

### Monitor Access:
- **Live HTML**: `V:\Projects\SIH\monitor.html` (auto-refresh)
- **Web Server**: `http://localhost:8502/monitor.html`

---

## 🎯 Success Criteria

### Deployment Readiness:
1. **Val Dice ≥ 95%**: Your stated requirement
2. **Val Dice ≥ 80%**: Your minimum acceptable
3. **Stable performance**: No overfitting
4. **Fast inference**: <1 second per image

### Current Gap Analysis:
- **Current**: 48.6%
- **Previous best**: 57.6%
- **Gap to minimum**: 80% - 48.6% = **31.4 points needed**
- **Gap to target**: 95% - 48.6% = **46.4 points needed**

This is why you need the aggressive Option C training to push beyond the previous plateau!