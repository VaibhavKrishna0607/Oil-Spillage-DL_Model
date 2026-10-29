#!/usr/bin/env python3
"""Quick script to check all model performances"""
import torch
from pathlib import Path

checkpoint_dir = Path("ml_model/checkpoints")

models_to_check = [
    "best_model.pth",
    "dataset2_improved.pth", 
    "dataset2_continued.pth",
    "stage1_best.pth",
    "stage1_sos_best.pth"
]

print("=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)

for model_file in models_to_check:
    model_path = checkpoint_dir / model_file
    if model_path.exists():
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                val_dice = ckpt.get("val_dice", "N/A")
                val_iou = ckpt.get("val_iou", "N/A") 
                epoch = ckpt.get("epoch", "N/A")
                
                dice_pct = f"{val_dice*100:.1f}%" if isinstance(val_dice, float) else val_dice
                iou_pct = f"{val_iou*100:.1f}%" if isinstance(val_iou, float) else val_iou
                
                print(f"{model_file:<25} | Val Dice: {dice_pct:<8} | IoU: {iou_pct:<8} | Epoch: {epoch}")
            else:
                print(f"{model_file:<25} | Old format (no metadata)")
        except Exception as e:
            print(f"{model_file:<25} | Error: {e}")
    else:
        print(f"{model_file:<25} | Not found")

print("=" * 60)