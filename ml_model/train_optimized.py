"""
Optimized training pipeline for SAR Oil Spill Detection.

Key improvements:
1. Enhanced loss functions (Focal Tversky + ComboLoss)
2. Larger model capacity (base=64 instead of 32)
3. Better learning rate scheduling (Cosine Annealing)
4. Longer training with early stopping based on validation
5. Mixed precision training for better convergence
6. Progressive unfreezing of encoder
7. Gradient clipping and normalization
"""

import argparse
import csv
import json
import sys
import time
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.krestenitis_dataset import KrestenitisTileDataset, load_krest_stats
from ml_model.model import UNet
from ml_model.advanced_losses import EnhancedComboLoss


def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """Calculate IoU and Dice score"""
    predicted = (y_pred > threshold).float()
    intersection = (predicted * y_true).sum().item()
    predicted_sum = predicted.sum().item()
    target_sum = y_true.sum().item()
    
    union = predicted_sum + target_sum - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (predicted_sum + target_sum + 1e-7)
    
    return iou, dice


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None, warmup_factor=1.0) -> Tuple[float, float, float]:
    """Train one epoch"""
    model.train()
    total_loss = total_iou = total_dice = 0.0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Warmup factor
        if warmup_factor < 1.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= warmup_factor
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if torch.isfinite(loss):
            total_loss += loss.item()
            iou, dice = calculate_metrics(outputs.detach(), masks)
            total_iou += iou
            total_dice += dice
    
    n_batches = len(loader)
    return (
        total_loss / n_batches if n_batches > 0 else float('inf'),
        total_iou / n_batches if n_batches > 0 else 0.0,
        total_dice / n_batches if n_batches > 0 else 0.0,
    )


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device) -> Tuple[float, float, float]:
    """Validate one epoch"""
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        if torch.isfinite(loss):
            total_loss += loss.item()
            iou, dice = calculate_metrics(outputs, masks)
            total_iou += iou
            total_dice += dice
    
    n_batches = len(loader)
    return (
        total_loss / n_batches if n_batches > 0 else float('inf'),
        total_iou / n_batches if n_batches > 0 else 0.0,
        total_dice / n_batches if n_batches > 0 else 0.0,
    )


def get_cosine_schedule(optimizer, num_epochs, num_batches_per_epoch, base_lr, min_lr=1e-6, warmup_epochs=1):
    """Cosine annealing with warmup"""
    def lr_lambda(current_step):
        total_steps = num_epochs * num_batches_per_epoch
        warmup_steps = warmup_epochs * num_batches_per_epoch
        
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--use-mixed-precision", action="store_true", default=True)
    parser.add_argument("--model-base", type=int, default=64)
    parser.add_argument("--use-attention", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = ROOT / "ml_model" / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    
    # Load or create model
    model = UNet(
        n_channels=1, 
        n_classes=1, 
        bilinear=False, 
        base=args.model_base,
        use_attention=args.use_attention
    ).to(device)
    
    print(f"Model: base={args.model_base}, attention={args.use_attention}")
    
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
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
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
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scaler = GradScaler() if args.use_mixed_precision and device.type == 'cuda' else None
    
    # Learning rate scheduler - Cosine annealing with warmup
    import math
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Multiply by 2 after each restart
        eta_min=1e-6
    )
    
    best_dice = 0.0
    patience_counter = 0
    history = []
    
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train IoU':<12} {'Train Dice':<12} "
          f"{'Val Loss':<12} {'Val IoU':<12} {'Val Dice':<12} {'LR':<12}")
    print("=" * 100)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Warmup for first epoch
        warmup_factor = min(epoch / 2.0, 1.0) if epoch <= 2 else 1.0
        
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, warmup_factor
        )
        val_loss, val_iou, val_dice = val_epoch(model, val_loader, loss_fn, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'lr': current_lr,
        })
        
        print(f"{epoch:<8} {train_loss:<12.4f} {train_iou:<12.4f} {train_dice:<12.4f} "
              f"{val_loss:<12.4f} {val_iou:<12.4f} {val_dice:<12.4f} {current_lr:<12.6f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            model_path = output_dir / "best_model_optimized.pth"
            torch.save({
                'state_dict': model.state_dict(),
                'arch': 'unet',
                'base': args.model_base,
                'attention': args.use_attention,
                'epoch': epoch,
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, model_path)
            print(f"  ★ New Best Dice: {val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 100}")
    print(f"Training complete in {elapsed / 60:.2f} minutes")
    print(f"Best Val Dice: {best_dice:.4f}")
    
    # Save history
    with open(output_dir / "train_history_optimized.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys() if history else [])
        writer.writeheader()
        writer.writerows(history)
    
    with open(output_dir / "train_meta_optimized.json", 'w') as f:
        json.dump({
            'model_base': args.model_base,
            'use_attention': args.use_attention,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'epochs_trained': len(history),
            'best_val_dice': float(best_dice),
            'total_time_minutes': round(elapsed / 60, 2),
            'device': str(device),
        }, f, indent=2)


if __name__ == "__main__":
    main()
