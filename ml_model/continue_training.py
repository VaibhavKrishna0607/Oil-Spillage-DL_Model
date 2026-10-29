"""
Continue training from the best model (53.25% Val Dice)
Fine-tunes with conservative LR to push higher without degradation
"""

import argparse
import csv
import json
import sys
import time
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.krestenitis_dataset import KrestenitisTileDataset, load_krest_stats
from ml_model.model import UNet
from ml_model.advanced_losses import EnhancedComboLoss


def metrics(prediction, target, threshold=0.5):
    """Calculate IoU and Dice"""
    predicted = (prediction > threshold).float()
    intersection = (predicted * target).sum().item()
    predicted_sum = predicted.sum().item()
    target_sum = target.sum().item()
    union = predicted_sum + target_sum - intersection
    return (
        (intersection + 1e-7) / (union + 1e-7),
        (2 * intersection + 1e-7) / (predicted_sum + target_sum + 1e-7),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    """Train or evaluate one epoch"""
    training = optimizer is not None
    model.train(training)
    total_loss = total_iou = total_dice = 0.0
    count = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        
        if torch.isfinite(loss):
            total_loss += float(loss.item())
            iou, dice = metrics(outputs.detach(), masks)
            total_iou += iou
            total_dice += dice
            count += 1
    
    if count == 0:
        return float("inf"), 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def evaluate_thresholds(model, loader, loss_fn, device):
    """Find optimal threshold on validation set"""
    model.eval()
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    totals = {t: [0.0, 0.0] for t in thresholds}
    total_loss = 0.0
    count = 0
    
    for images, masks in loader:
        outputs = model(images.to(device))
        masks = masks.to(device)
        loss = loss_fn(outputs, masks)
        
        if torch.isfinite(loss):
            total_loss += float(loss.item())
            count += 1
            for t in thresholds:
                iou, dice = metrics(outputs, masks, t)
                totals[t][0] += iou
                totals[t][1] += dice
    
    if count == 0:
        return float("inf"), 0.0, 0.0, 0.5
    
    scores = {t: (totals[t][0] / count, totals[t][1] / count) for t in thresholds}
    best_t = max(scores, key=lambda t: scores[t][1])
    best_iou, best_dice = scores[best_t]
    
    return total_loss / count, best_iou, best_dice, best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ml_model/checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = ROOT / "ml_model" / "checkpoints"
    model_path = output_dir / args.model_path.split("/")[-1]

    # Create model
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base=32, use_attention=True)
    
    # Load best model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    
    # Load datasets
    stats = load_krest_stats(ROOT)
    train_ds = KrestenitisTileDataset(
        root_dir=str(ROOT / "Oil Spill Detection Dataset"),
        split="train",
        tile_size=256,
        stride=64,
        augment=True,
        oil_only=True,
        norm_stats=stats,
    )
    val_ds = KrestenitisTileDataset(
        root_dir=str(ROOT / "Oil Spill Detection Dataset"),
        split="test",
        tile_size=256,
        stride=128,
        augment=False,
        oil_only=True,
        norm_stats=stats,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Loss and optimizer
    loss_fn = EnhancedComboLoss(
        w_bce=0.2,
        w_tversky=0.5,
        w_dice=0.3,
        pos_weight=2.9,
        focal_alpha=0.4,
        focal_beta=0.6,
        focal_gamma=1.5
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scaler = GradScaler()
    
    # Conservative learning rate schedule - very gentle
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    
    # Get baseline
    baseline_loss, baseline_iou, baseline_dice, baseline_threshold = evaluate_thresholds(
        model, val_loader, loss_fn, device
    )
    print(f"\nContinuation Baseline: Loss={baseline_loss:.4f} IoU={baseline_iou:.4f} Dice={baseline_dice:.4f}")
    
    best_dice = baseline_dice
    best_threshold = baseline_threshold
    patience_counter = 0
    history = []

    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Dice':<12} {'Val Loss':<12} {'Val IoU':<12} {'Val Dice':<12} {'Threshold':<10} {'LR':<12}")
    print("=" * 100)

    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        val_loss, val_iou, val_dice, val_threshold = evaluate_thresholds(model, val_loader, loss_fn, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "val_threshold": val_threshold,
            "learning_rate": current_lr,
        }
        history.append(row)
        
        is_best = val_dice > best_dice
        marker = " ★ NEW BEST!" if is_best else ""
        
        print(f"{epoch:<6} {train_loss:<12.4f} {train_dice:<12.4f} {val_loss:<12.4f} "
              f"{val_iou:<12.4f} {val_dice:<12.4f} {val_threshold:<10.2f} {current_lr:<12.2e}{marker}")
        
        if is_best:
            best_dice = val_dice
            best_threshold = val_threshold
            patience_counter = 0
            model_path_save = output_dir / "dataset2_continued.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "arch": "unet",
                "base": 32,
                "stage": "continued_training",
                "val_iou": val_iou,
                "val_dice": val_dice,
                "threshold": val_threshold,
                "epoch": epoch,
            }, model_path_save)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 100}")
    print(f"Continuation training complete in {elapsed / 60:.2f} minutes")
    print(f"Baseline Val Dice: {baseline_dice:.4f}")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Improvement: {(best_dice - baseline_dice):.4f} ({((best_dice - baseline_dice) / baseline_dice * 100):.1f}%)")
    
    # Save history
    with open(output_dir / "dataset2_continued_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys() if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    
    with open(output_dir / "dataset2_continued_meta.json", "w") as f:
        json.dump({
            "baseline_dice": float(baseline_dice),
            "best_dice": float(best_dice),
            "improvement": float(best_dice - baseline_dice),
            "best_threshold": float(best_threshold),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs_trained": len(history),
            "device": str(device),
            "elapsed_minutes": round(elapsed / 60, 2),
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
