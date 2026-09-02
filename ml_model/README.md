# U-Net Oil Spill Segmentation Model (Person 2 - ML Engineer Deliverables)

This folder contains the complete, production-ready machine learning pipeline for binary oil spill detection on SAR images.

---

## Directory Structure

```
ml_model/
├── model.py            # Custom high-performance U-Net architecture
├── loss.py             # Hybrid BCE + Dice loss with class balancing
├── train.py            # Training & validation loop + TorchScript/ONNX export
├── predict.py          # Prediction CLI + Contour/Polygon generation
├── checkpoints/        # Directory where model weights are saved
│   ├── best_model.pth  # PyTorch weights (State Dict)
│   ├── best_model.pt   # TorchScript traced model
│   └── best_model.onnx # ONNX model
└── README.md           # This file
```

---

## Model Architecture

The U-Net model defined in `model.py` follows the classic contracting/expanding path:
- **Contracting Path**: 4 downsampling blocks consisting of Double Convolution blocks (`3x3 Conv => BatchNorm => ReLU => 3x3 Conv => BatchNorm => ReLU`) followed by Max Pooling (`2x2`).
- **Expanding Path**: 4 upsampling blocks consisting of 2D Bilinear Upsampling or Transposed Convolutions, concatenation of skip connections from the contracting path, and Double Convolution blocks.
- **Output Projection**: `1x1 Conv` followed by a `Sigmoid` activation function to project back to a 1-channel probability map.

---

## Loss Function Design

To address the mild class imbalance where oil pixels account for **25.4%** of the dataset and background is **74.6%**, we use a hybrid **BCEDiceLoss**:
- **BCE component**: Uses Binary Cross-Entropy with `pos_weight=2.9` to balance precision and recall.
- **Dice component**: Optimizes overlap directly, ensuring sharp, clean boundaries of detected oil spills.
- Combined loss: `0.5 * BCE + 0.5 * Dice`.

---

## How to Run

### 1. Training (Verification / Local CPU)
Since local training on CPU would be extremely slow, use the `--dry-run` flag to verify the training script and export pipeline. It runs on a small subset (8 train, 4 val) for 1 epoch:
```bash
python -X utf8 ml_model/train.py --dry-run
```

### 2. Training (Full Scale on GPU / Kaggle / Colab)
Transfer these scripts to a GPU machine and run:
```bash
python -X utf8 ml_model/train.py --epochs 15 --batch-size 16 --lr 1e-4
```
It will automatically run on GPU (`cuda`) and output the best model checkpoints to `ml_model/checkpoints`.

---

## Model Export Formats

By default, the training pipeline exports three versions of the best model:
1. `best_model.pth` - Standard PyTorch state dict.
2. `best_model.pt` - **TorchScript** serialized model (does not need model source code to load, ideal for C++/FastAPI).
3. `best_model.onnx` - **ONNX** model with dynamic batch-size support, suitable for high-performance inference using `onnxruntime` or OpenVINO.

---

## Inference & Polygon Generation (For Person 4 - Backend integration)

You can run predictions on any raw `.png` image using `predict.py`. It prints a structured **JSON output** with the polygon contours and saves a visual overlay image.

### Usage with ONNX:
```bash
python -X utf8 ml_model/predict.py \
  --image Images/images/val/palsar_0.png \
  --model ml_model/checkpoints/best_model.onnx \
  --output-image ml_model/detection_overlay.png
```

### Usage with TorchScript:
```bash
python -X utf8 ml_model/predict.py \
  --image Images/images/val/palsar_0.png \
  --model ml_model/checkpoints/best_model.pt
```

### JSON Output format:
```json
{
  "image_file": "Images/images/val/palsar_0.png",
  "spill_detected": true,
  "polygons_count": 1,
  "polygons": [
    [
      [120, 80],
      [125, 82],
      [128, 90],
      ...
    ]
  ]
}
```
Polygons are coordinates inside the `256x256` model grid representing boundaries of detected spills.
