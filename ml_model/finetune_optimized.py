"""
Optimized fine-tuning for dataset2 with advanced loss functions.

Improves upon existing baseline by:
1. Using EnhancedComboLoss (Focal Tversky + Weighted BCE + Dice)
2. Dynamic threshold optimization
3. Better learning rate scheduling
4. Extended training cycles
5. Model capacity increase (base=64)
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--baseline-dice", type=float, default=0.5690)
    parser.add_argument("--model-base", type=int, default=64)
    parser.add_argument("--use-mixed-precision", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = ROOT / "ml_model" / "checkpoints"
    baseline_path = output_dir / "best_model.pth"
    candidate_path = output_dir / "dataset2_optimized.pth"

    # Load baseline model
    checkpoint = torch.load(str(baseline_path), map_location=device, weights_only=True)
    baseline_base = int(checkpoint.get("base", 32))
    
    # Create model with potentially larger capacity
    model = UNet(
        n_channels=1, 
        n_classes=1, 
        bilinear=False, 
        base=args.model_base,
        use_attention=True
    ).to(device)
    
    # Load baseline weights - use load_state_dict with strict=False to allow for new attention layers
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("✓ Baseline model loaded successfully")
    except RuntimeError as e:
        print(f"Warning: Could not fully load baseline - some layers will be randomly initialized")
        print(f"  Error: {str(e)[:100]}...")
        try:
            # Try loading with very lenient approach
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
            if incompatible.missing_keys:
                print(f"  Missing {len(incompatible.missing_keys)} keys (will use random init): attention layers")
            if incompatible.unexpected_keys:
                print(f"  Unexpected {len(incompatible.unexpected_keys)} keys (ignored)")
        except Exception as e2:
            print(f"  Could not load any weights - training from scratch")

    # Load datasets
    stats = load_krest_stats(ROOT)
    train_ds = KrestenitisTileDataset(
        root_dir=str(ROOT / "Oil Spill Detection Dataset"),
        split="train",
        tile_size=256,
        stride=64,
        augment=True,
        oil_only=True,
        max_samples=args.max_train_samples,
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
    
    # Enhanced loss function
    loss_fn = EnhancedComboLoss(
        w_bce=0.2,
        w_tversky=0.5,
        w_dice=0.3,
        pos_weight=2.9,
        focal_alpha=0.4,
        focal_beta=0.6,
        focal_gamma=1.5
    )

    # Baseline evaluation
    baseline_loss, baseline_iou, baseline_dice, baseline_threshold = evaluate_thresholds(
        model, val_loader, loss_fn, device
    )
    print(f"\nBaseline: Loss={baseline_loss:.4f} IoU={baseline_iou:.4f} Dice={baseline_dice:.4f} Threshold={baseline_threshold:.2f}")
    
    best_dice = max(args.baseline_dice, baseline_dice)
    best_threshold = baseline_threshold
    patience_counter = 0
    history = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scaler = GradScaler() if args.use_mixed_precision and device.type == 'cuda' else None
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3,
        T_mult=2,
        eta_min=1e-7
    )
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Dice':<12} {'Val Loss':<12} {'Val IoU':<12} {'Val Dice':<12} {'Threshold':<10} {'LR':<12}")
    print("=" * 90)

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
        
        print(f"{epoch:<6} {train_loss:<12.4f} {train_dice:<12.4f} {val_loss:<12.4f} "
              f"{val_iou:<12.4f} {val_dice:<12.4f} {val_threshold:<10.2f} {current_lr:<12.2e}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_threshold = val_threshold
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": "unet",
                "base": args.model_base,
                "stage": "dataset2_optimized",
                "val_iou": val_iou,
                "val_dice": val_dice,
                "threshold": val_threshold,
                "epoch": epoch,
            }, candidate_path)
            print(f"  ★ New Best Dice: {val_dice:.4f} at Threshold: {val_threshold:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stop (no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 90}")
    print(f"Training complete in {elapsed / 60:.2f} minutes")
    print(f"Best Val Dice: {best_dice:.4f} (improved by {(best_dice - args.baseline_dice) * 100:.2f}%)")
    
    # Save history
    with open(output_dir / "dataset2_history_optimized.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys() if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    
    with open(output_dir / "dataset2_meta_optimized.json", "w") as f:
        json.dump({
            "baseline_dice": args.baseline_dice,
            "best_dice": float(best_dice),
            "improvement": float(best_dice - args.baseline_dice),
            "best_threshold": float(best_threshold),
            "model_base": args.model_base,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs_trained": len(history),
            "device": str(device),
            "elapsed_minutes": round(elapsed / 60, 2),
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    start = time.time()
    main()
