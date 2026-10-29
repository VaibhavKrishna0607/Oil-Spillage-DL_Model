"""Sweep segmentation thresholds on full Krestenitis scenes."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.krestenitis_dataset import load_krest_stats
from ml_model.infer import krestenitis_oil_mask, load_checkpoint
from ml_model.postprocessing import (
    dihedral_tta_inference,
    gaussian_sliding_window_inference,
    morphological_clean,
)


def main():
    parser = argparse.ArgumentParser(description="Find the best full-scene segmentation threshold")
    parser.add_argument("--model", default="ml_model/checkpoints/best_model.pth")
    parser.add_argument("--max-scenes", type=int, default=30)
    parser.add_argument("--start", type=float, default=0.20)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--tta", action="store_true", help="Use 8-way flip/rotation ensemble")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(ROOT / args.model), device)
    stats = load_krest_stats(ROOT)
    test_dir = ROOT / "Oil Spill Detection Dataset" / "test"
    image_files = sorted((test_dir / "images").glob("*.jpg"))[:args.max_scenes]

    records = []
    for image_path in image_files:
        label_path = test_dir / "labels_1D" / f"{image_path.stem}.png"
        if not label_path.exists():
            continue
        gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        infer = dihedral_tta_inference if args.tta else gaussian_sliding_window_inference
        probability = infer(model, gray, stats, device, tile=256, overlap=64)
        ground_truth = krestenitis_oil_mask(label_path).astype(bool)
        records.append((probability, ground_truth))

    if not records:
        raise RuntimeError("No labelled test scenes were found.")

    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)
    results = []
    for threshold in thresholds:
        ious = []
        dices = []
        for probability, ground_truth in records:
            cleaned = morphological_clean(
                probability > threshold,
                open_kernel_size=3,
                close_kernel_size=5,
                min_area_px=15,
            ).astype(bool)
            intersection = np.logical_and(cleaned, ground_truth).sum()
            union = np.logical_or(cleaned, ground_truth).sum()
            denominator = cleaned.sum() + ground_truth.sum()
            ious.append(float((intersection + 1e-7) / (union + 1e-7)))
            dices.append(float((2 * intersection + 1e-7) / (denominator + 1e-7)))
        results.append({
            "threshold": round(float(threshold), 4),
            "mean_iou": float(np.mean(ious)),
            "mean_dice_f1": float(np.mean(dices)),
            "scenes": len(records),
        })

    best = max(results, key=lambda row: row["mean_iou"])
    output_dir = ROOT / "ml_model" / "checkpoints"
    with open(output_dir / "threshold_sweep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with open(output_dir / "threshold_meta.json", "w") as handle:
        json.dump({"best": best, "device": str(device), "model": args.model, "tta": args.tta}, handle, indent=2)

    print(f"Evaluated {len(records)} scenes on {device}")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Best full-scene IoU: {best['mean_iou']:.4f}")
    print(f"Best full-scene Dice/F1: {best['mean_dice_f1']:.4f}")
    print(f"Results: {output_dir / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
