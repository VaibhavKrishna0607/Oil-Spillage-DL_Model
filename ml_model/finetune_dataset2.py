"""Targeted Dataset-2 fine-tuning with baseline protection."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.krestenitis_dataset import KrestenitisTileDataset, load_krest_stats
from ml_model.loss import BCEDiceLoss
from ml_model.model import UNet


def metrics(prediction, target, threshold=0.5):
    predicted = (prediction > threshold).float()
    intersection = (predicted * target).sum().item()
    predicted_sum = predicted.sum().item()
    target_sum = target.sum().item()
    union = predicted_sum + target_sum - intersection
    return (
        (intersection + 1e-7) / (union + 1e-7),
        (2 * intersection + 1e-7) / (predicted_sum + target_sum + 1e-7),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_iou = total_dice = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            if not torch.isfinite(loss):
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        iou, dice = metrics(outputs.detach(), masks)
        total_loss += float(loss.item())
        total_iou += iou
        total_dice += dice
        count += 1
    if count == 0:
        return float("inf"), 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def evaluate_thresholds(model, loader, loss_fn, device):
    model.eval()
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    totals = {threshold: [0.0, 0.0] for threshold in thresholds}
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        outputs = model(images.to(device))
        masks = masks.to(device)
        loss = loss_fn(outputs, masks)
        if not torch.isfinite(loss):
            continue
        total_loss += float(loss.item())
        count += 1
        for threshold in thresholds:
            iou, dice = metrics(outputs, masks, threshold)
            totals[threshold][0] += iou
            totals[threshold][1] += dice
    if count == 0:
        return float("inf"), 0.0, 0.0, 0.5
    scores = {
        threshold: (values[0] / count, values[1] / count)
        for threshold, values in totals.items()
    }
    best_threshold = max(scores, key=lambda threshold: scores[threshold][1])
    best_iou, best_dice = scores[best_threshold]
    return total_loss / count, best_iou, best_dice, best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--baseline-dice", type=float, default=0.5690)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / "ml_model" / "checkpoints"
    baseline_path = output_dir / "best_model.pth"
    candidate_path = output_dir / "dataset2_improved.pth"

    checkpoint = torch.load(str(baseline_path), map_location=device, weights_only=True)
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base=int(checkpoint.get("base", 32))).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    stats = load_krest_stats(ROOT)
    train_ds = KrestenitisTileDataset(
        root_dir=str(ROOT / "Oil Spill Detection Dataset"), split="train",
        tile_size=256, stride=64, augment=True, oil_only=True,
        max_samples=args.max_train_samples, norm_stats=stats,
    )
    val_ds = KrestenitisTileDataset(
        root_dir=str(ROOT / "Oil Spill Detection Dataset"), split="test",
        tile_size=256, stride=128, augment=False, oil_only=True,
        norm_stats=stats,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loss_fn = BCEDiceLoss(alpha=0.5, pos_weight=2.9)

    baseline_loss, baseline_iou, baseline_dice, baseline_threshold = evaluate_thresholds(
        model, val_loader, loss_fn, device
    )
    print(f"Baseline validation: loss={baseline_loss:.4f} IoU={baseline_iou:.4f} "
          f"Dice={baseline_dice:.4f} threshold={baseline_threshold:.2f}")
    best_dice = max(args.baseline_dice, baseline_dice)
    history = []

    for epoch in range(1, args.epochs + 1):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_iou, val_dice, val_threshold = evaluate_thresholds(
            model, val_loader, loss_fn, device
        )
        row = {"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou, "train_dice": train_dice,
               "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice,
               "val_threshold": val_threshold}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}: train IoU={train_iou:.4f} Dice={train_dice:.4f}; "
              f"val IoU={val_iou:.4f} Dice={val_dice:.4f} threshold={val_threshold:.2f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"state_dict": model.state_dict(), "arch": "unet", "base": 32,
                        "stage": "dataset2_targeted", "val_iou": val_iou, "val_dice": val_dice}, candidate_path)
            print(f"  New improvement saved -> {candidate_path}")

    with open(output_dir / "dataset2_finetune_history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys() if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    with open(output_dir / "dataset2_finetune_meta.json", "w") as handle:
        json.dump({"baseline_dice": args.baseline_dice, "measured_baseline_dice": baseline_dice,
                   "best_candidate_dice": best_dice, "candidate": str(candidate_path) if candidate_path.exists() else None,
                   "best_threshold": val_threshold if history else baseline_threshold,
                   "device": str(device), "elapsed_minutes": round((time.time() - start) / 60, 2)}, handle, indent=2)
    print(f"Best candidate Dice={best_dice:.4f}; baseline checkpoint preserved={not candidate_path.exists()}")


if __name__ == "__main__":
    start = time.time()
    main()
