"""
Simplified Training Script - Fast and Effective
===============================================
A streamlined version that focuses on essential optimizations for maximum performance.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.sos_dataset import SOSDataset
from data_pipeline.krestenitis_dataset import KrestenitisTileDataset, load_krest_stats
from ml_model.model import UNet
from ml_model.loss import BCEDiceLoss
from ml_model.infer import load_norm_stats

torch.manual_seed(42)
np.random.seed(42)


def calculate_metrics(y_pred, y_true, threshold=0.5):
    """Calculate IoU and Dice metrics"""
    pred_bin = (y_pred > threshold).float()
    pred_flat = pred_bin.reshape(-1)
    true_flat = y_true.reshape(-1)
    
    tp = (pred_flat * true_flat).sum().item()
    fp = (pred_flat * (1 - true_flat)).sum().item()
    fn = ((1 - pred_flat) * true_flat).sum().item()
    
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
    
    return iou, dice


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    """Training epoch with mixed precision"""
    model.train()
    total_loss = total_iou = total_dice = 0.0
    count = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
                masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
                loss = loss_fn(outputs, masks)
            
            if not torch.isfinite(loss):
                print(f"  [NaN Guard] Skipping batch {batch_idx}")
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
            masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
            loss = loss_fn(outputs, masks)
            
            if not torch.isfinite(loss):
                print(f"  [NaN Guard] Skipping batch {batch_idx}")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
        
        iou, dice = calculate_metrics(outputs.detach(), masks)
        total_loss += float(loss.item())
        total_iou += iou
        total_dice += dice
        count += 1
        
        if batch_idx % max(1, len(loader) // 5) == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}, IoU={iou:.4f}, Dice={dice:.4f}")
    
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
    """Validation epoch"""
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    count = 0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        iou, dice = calculate_metrics(outputs, masks)
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        count += 1
    
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def find_best_threshold(model, loader, device):
    """Find optimal threshold on validation set"""
    model.eval()
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_dice = 0.0
    
    all_outputs = []
    all_masks = []
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        all_outputs.append(outputs.cpu())
        all_masks.append(masks.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_masks = torch.cat(all_masks)
    
    for threshold in thresholds:
        iou, dice = calculate_metrics(all_outputs, all_masks, threshold)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    return best_threshold, best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["dataset1", "dataset2", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)  # Smaller batch for stability
    parser.add_argument("--lr", type=float, default=5e-5)  # More conservative
    parser.add_argument("--patience", type=int, default=15)  # More patience
    parser.add_argument("--weight-decay", type=float, default=1e-5)  # Regularization
    parser.add_argument("--grad-clip", type=float, default=0.5)  # Prevent gradient explosion
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training: {args.dataset}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    
    output_dir = ROOT / "ml_model" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base=32, use_attention=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_fn = BCEDiceLoss(alpha=0.5, pos_weight=2.9)
    
    # Resume if specified
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"Resumed from epoch {start_epoch}")
    
    history = []
    
    # Dataset 1 training
    if args.dataset in ["dataset1", "both"]:
        print("\n" + "="*60)
        print("TRAINING DATASET1 (SOS)")
        print("="*60)
        
        sos_stats = load_norm_stats(ROOT)
        
        sos_train_ds = SOSDataset(
            root_dir=ROOT, split="train", augment=True,
            max_samples=None, norm_stats=sos_stats, cache_in_ram=True
        )
        sos_val_ds = SOSDataset(
            root_dir=ROOT, split="val", augment=False,
            max_samples=None, norm_stats=sos_stats, cache_in_ram=True
        )
        
        sos_train_loader = DataLoader(
            sos_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True, pin_memory=True
        )
        sos_val_loader = DataLoader(
            sos_val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        print(f"Train: {len(sos_train_ds)}, Val: {len(sos_val_ds)}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler()
        
        best_dice = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e}")
            
            train_loss, train_iou, train_dice = train_epoch(model, sos_train_loader, optimizer, loss_fn, device, scaler)
            val_loss, val_iou, val_dice = val_epoch(model, sos_val_loader, loss_fn, device)
            best_threshold, best_threshold_dice = find_best_threshold(model, sos_val_loader, device)
            
            scheduler.step()
            
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"Best threshold: {best_threshold:.2f} (Dice: {best_threshold_dice:.4f})")
            
            row = {
                "epoch": epoch,
                "dataset": "dataset1",
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "best_threshold": best_threshold,
                "best_threshold_dice": best_threshold_dice,
                "learning_rate": current_lr,
            }
            history.append(row)
            
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "arch": "unet",
                    "base": 32,
                    "stage": "dataset1_simple",
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "threshold": best_threshold,
                    "epoch": epoch,
                }, output_dir / "dataset1_simple_best.pth")
                print(f"* New best! Dice: {val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        elapsed = time.time() - start_time
        print(f"\nDataset1 done in {elapsed/60:.2f}m, Best Dice: {best_dice:.4f}")
        
        with open(output_dir / "dataset1_simple_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    
    # Dataset 2 training
    if args.dataset in ["dataset2", "both"]:
        print("\n" + "="*60)
        print("TRAINING DATASET2 (Krestenitis)")
        print("="*60)
        
        # Load best dataset1 model if training both
        if args.dataset == "both" and (output_dir / "dataset1_simple_best.pth").exists():
            ckpt = torch.load(output_dir / "dataset1_simple_best.pth", map_location=device, weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            print("Loaded best Dataset1 model")
        
        krest_stats = load_krest_stats(ROOT)
        
        krest_train_ds = KrestenitisTileDataset(
            root_dir=str(ROOT / "Oil Spill Detection Dataset"),
            split="train",
            tile_size=256,
            stride=64,
            augment=True,
            oil_only=True,
            norm_stats=krest_stats,
        )
        krest_val_ds = KrestenitisTileDataset(
            root_dir=str(ROOT / "Oil Spill Detection Dataset"),
            split="test",
            tile_size=256,
            stride=128,
            augment=False,
            oil_only=True,
            norm_stats=krest_stats,
        )
        
        krest_train_loader = DataLoader(
            krest_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True, pin_memory=True
        )
        krest_val_loader = DataLoader(
            krest_val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        print(f"Train: {len(krest_train_ds)}, Val: {len(krest_val_ds)}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        scaler = GradScaler()
        
        best_dice = 0.0
        patience_counter = 0
        history_d2 = []
        start_time = time.time()
        
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e}")
            
            train_loss, train_iou, train_dice = train_epoch(model, krest_train_loader, optimizer, loss_fn, device, scaler)
            val_loss, val_iou, val_dice = val_epoch(model, krest_val_loader, loss_fn, device)
            best_threshold, best_threshold_dice = find_best_threshold(model, krest_val_loader, device)
            
            scheduler.step()
            
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"Best threshold: {best_threshold:.2f} (Dice: {best_threshold_dice:.4f})")
            
            row = {
                "epoch": epoch,
                "dataset": "dataset2",
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "best_threshold": best_threshold,
                "best_threshold_dice": best_threshold_dice,
                "learning_rate": current_lr,
            }
            history_d2.append(row)
            
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "arch": "unet",
                    "base": 32,
                    "stage": "dataset2_simple",
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "threshold": best_threshold,
                    "epoch": epoch,
                }, output_dir / "dataset2_simple_best.pth")
                print(f"* New best! Dice: {val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        elapsed = time.time() - start_time
        print(f"\nDataset2 done in {elapsed/60:.2f}m, Best Dice: {best_dice:.4f}")
        
        with open(output_dir / "dataset2_simple_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_d2[0].keys())
            writer.writeheader()
            writer.writerows(history_d2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()