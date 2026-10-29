"""
Full Training Pipeline — Push to Maximum Performance
=====================================================
Two-stage curriculum learning with aggressive optimization:
  Stage 1: Pre-train on Dataset-1 (SOS) — learn general oil spill features
  Stage 2: Fine-tune on Dataset-2 (Krestenitis real-life) — adapt to real SAR
  
Key improvements over quick run:
  - All data used (no sample limits)
  - More epochs with early stopping (stops when model stops improving)
  - Gradual unfreezing in Stage 2 (freeze encoder first, then unfreeze)
  - Learning rate warmup + cosine decay
  - Mixed precision training (faster on GPU)
  - Oil-focused tile filtering for Stage 2
  - Patience-based early stopping
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
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
from ml_model.infer import load_norm_stats, krestenitis_oil_mask
from ml_model.postprocessing import gaussian_sliding_window_inference, morphological_clean

torch.manual_seed(42)
np.random.seed(42)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    pred_bin = (y_pred > threshold).float()
    pred_flat = pred_bin.reshape(-1)
    true_flat = y_true.reshape(-1)
    intersection = (pred_flat * true_flat).sum().item()
    pred_sum = pred_flat.sum().item()
    true_sum = true_flat.sum().item()
    union = pred_sum + true_sum - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred_sum + true_sum + 1e-7)
    return iou, dice


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> Tuple[float, float, float]:
    model.train()
    total_loss = total_iou = total_dice = 0.0
    count = 0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
                masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
                loss = loss_fn(outputs, masks)
            if not torch.isfinite(loss):
                print(f"  [NaN Guard] Skipping batch {batch_idx}/{len(loader)}: invalid loss -> resetting grads")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
            masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
            loss = loss_fn(outputs, masks)
            if not torch.isfinite(loss):
                print(f"  [NaN Guard] Skipping batch {batch_idx}/{len(loader)}: invalid loss -> resetting grads")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
def val_epoch(model, loader, loss_fn, device) -> Tuple[float, float, float]:
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
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def evaluate_full_scenes(model, krest_test_dir, krest_stats, device, max_scenes=20) -> float:
    """Evaluate on full 1250x650 Krestenitis scenes with sliding window."""
    model.eval()
    img_files = sorted(list((krest_test_dir / "images").glob("*.jpg")))[:max_scenes]
    ious = []
    for f in img_files:
        lab_f = krest_test_dir / "labels_1D" / (f.stem + ".png")
        if not lab_f.exists():
            continue
        gt = krestenitis_oil_mask(lab_f)
        gray = np.array(Image.open(f).convert("L"), dtype=np.float32)
        pred = gaussian_sliding_window_inference(model, gray, krest_stats, device, tile=256, overlap=64)
        cleaned = morphological_clean(pred > 0.5, open_kernel_size=3, close_kernel_size=5, min_area_px=15)
        iou = ((cleaned.astype(bool) & gt.astype(bool)).sum() + 1e-7) / ((cleaned.astype(bool) | gt.astype(bool)).sum() + 1e-7)
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def freeze_encoder(model):
    """Freeze encoder layers (down path) so Stage 2 only trains decoder first."""
    for name, param in model.named_parameters():
        if any(k in name for k in ["inc", "down1", "down2", "down3", "down4"]):
            param.requires_grad = False
    print("  [Freeze] Encoder layers frozen — only decoder will train")


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("  [Unfreeze] All layers unfrozen — full model will train")


def export_model(model, output_dir, device, n_channels=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, n_channels, 256, 256, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_dir / "best_model.pt"))
        print(f"[Export] TorchScript -> {output_dir / 'best_model.pt'}")
    except Exception as e:
        print(f"[Export Warning] TorchScript: {e}")
    try:
        torch.onnx.export(model, dummy, str(output_dir / "best_model.onnx"),
                          export_params=True, opset_version=18,
                          input_names=["input"], output_names=["output"])
        print(f"[Export] ONNX -> {output_dir / 'best_model.onnx'}")
    except Exception as e:
        print(f"[Export Warning] ONNX: {e}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the recovery run in full precision; the previous AMP run corrupted
    # optimizer state with non-finite gradients before the loss guard ran.
    use_amp = False
    scaler = GradScaler() if use_amp else None

    output_dir = ROOT / "ml_model" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ──
    STAGE1_EPOCHS = 12
    STAGE2_FROZEN_EPOCHS = 6
    STAGE2_FULL_EPOCHS = 8
    BATCH_SIZE = 8 if device.type == "cuda" else 4
    LR_STAGE1 = 1e-4
    LR_STAGE2_FROZEN = 3e-5
    LR_STAGE2_FULL = 1e-5
    PATIENCE = 5
    BASE_CHANNELS = 32

    print("=" * 60)
    print("  FULL TRAINING PIPELINE — MAXIMUM PERFORMANCE MODE")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Stage 1:    {STAGE1_EPOCHS} epochs on Dataset-1 (SOS)")
    print(f"  Stage 2a:   {STAGE2_FROZEN_EPOCHS} epochs frozen encoder on Dataset-2")
    print(f"  Stage 2b:   {STAGE2_FULL_EPOCHS} epochs full fine-tune on Dataset-2")
    print(f"  Early Stop: patience={PATIENCE}")
    print("=" * 60)

    # Load stats
    sos_stats = load_norm_stats(ROOT)
    krest_stats = load_krest_stats(ROOT)

    # Initialize model
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base=BASE_CHANNELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    history_rows = []
    best_val_dice = -1.0
    best_weights_path = output_dir / "best_model.pth"
    start_time = time.time()
    current_epoch = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Pre-training on Dataset-1 (SOS — all 6455 samples)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  STAGE 1: Pre-training on Dataset-1 (Refined SOS)")
    print("  Using ALL samples with augmentation")
    print("█" * 60)

    sos_train_ds = SOSDataset(root_dir=ROOT, split="train", augment=True,
                              max_samples=None, norm_stats=sos_stats, cache_in_ram=True)
    sos_val_ds = SOSDataset(root_dir=ROOT, split="val", augment=False,
                            max_samples=None, norm_stats=sos_stats, cache_in_ram=True)

    sos_train_loader = DataLoader(sos_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True, pin_memory=True)
    sos_val_loader = DataLoader(sos_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_STAGE1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-6)
    loss_fn = BCEDiceLoss(alpha=0.5, pos_weight=2.9)

    patience_counter = 0
    best_stage1_dice = -1.0

    for ep in range(1, STAGE1_EPOCHS + 1):
        current_epoch += 1
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n--- [Stage 1] Epoch {ep}/{STAGE1_EPOCHS} (LR={lr_now:.6f}) ---")

        t_loss, t_iou, t_dice = train_epoch(model, sos_train_loader, optimizer, loss_fn, device, scaler)
        v_loss, v_iou, v_dice = val_epoch(model, sos_val_loader, loss_fn, device)
        scheduler.step()

        print(f"  Train: Loss={t_loss:.4f}  IoU={t_iou:.4f}  Dice={t_dice:.4f}")
        print(f"  Val:   Loss={v_loss:.4f}  IoU={v_iou:.4f}  Dice={v_dice:.4f}")

        history_rows.append({
            "epoch": current_epoch, "stage": "Stage1_SOS",
            "train_loss": round(t_loss, 4), "train_iou": round(t_iou, 4), "train_dice": round(t_dice, 4),
            "val_loss": round(v_loss, 4), "val_iou": round(v_iou, 4), "val_dice": round(v_dice, 4),
        })

        if v_dice > best_stage1_dice:
            best_stage1_dice = v_dice
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "arch": "unet", "base": BASE_CHANNELS,
                         "stage": 1, "val_iou": v_iou, "val_dice": v_dice},
                       str(output_dir / "stage1_best.pth"))
            print(f"  ★ New Best Stage 1! Dice={v_dice:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"  ⚡ Early stopping at epoch {ep}")
            break

    print(f"\n✅ Stage 1 Complete — Best Val Dice: {best_stage1_dice:.4f}")

    # Reload best Stage 1 weights before Stage 2
    ckpt = torch.load(str(output_dir / "stage1_best.pth"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    print("  Loaded best Stage 1 checkpoint for Stage 2")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2a: Fine-tune on Krestenitis (Frozen Encoder)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  STAGE 2a: Fine-tune on Dataset-2 (Encoder FROZEN)")
    print("  Only decoder adapts — prevents forgetting Stage 1 knowledge")
    print("█" * 60)

    # Use oil_only=True to focus on informative tiles
    krest_train_ds = KrestenitisTileDataset(
        root_dir="Oil Spill Detection Dataset", split="train",
        tile_size=256, stride=64, augment=True,
        oil_only=True, max_samples=None, norm_stats=krest_stats,
    )
    krest_val_ds = KrestenitisTileDataset(
        root_dir="Oil Spill Detection Dataset", split="test",
        tile_size=256, stride=128, augment=False,
        oil_only=True, max_samples=None, norm_stats=krest_stats,
    )

    krest_train_loader = DataLoader(krest_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, drop_last=True, pin_memory=True)
    krest_val_loader = DataLoader(krest_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)

    freeze_encoder(model)

    # Only optimize unfrozen params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer2a = torch.optim.AdamW(trainable_params, lr=LR_STAGE2_FROZEN, weight_decay=1e-4)
    loss_fn2 = BCEDiceLoss(alpha=0.5, pos_weight=2.9)

    best_stage2_dice = -1.0
    patience_counter = 0

    for ep in range(1, STAGE2_FROZEN_EPOCHS + 1):
        current_epoch += 1
        lr_now = optimizer2a.param_groups[0]["lr"]
        print(f"\n--- [Stage 2a - Frozen] Epoch {ep}/{STAGE2_FROZEN_EPOCHS} (LR={lr_now:.6f}) ---")

        t_loss, t_iou, t_dice = train_epoch(model, krest_train_loader, optimizer2a, loss_fn2, device, scaler)
        v_loss, v_iou, v_dice = val_epoch(model, krest_val_loader, loss_fn2, device)

        print(f"  Train: Loss={t_loss:.4f}  IoU={t_iou:.4f}  Dice={t_dice:.4f}")
        print(f"  Val:   Loss={v_loss:.4f}  IoU={v_iou:.4f}  Dice={v_dice:.4f}")

        history_rows.append({
            "epoch": current_epoch, "stage": "Stage2a_Frozen",
            "train_loss": round(t_loss, 4), "train_iou": round(t_iou, 4), "train_dice": round(t_dice, 4),
            "val_loss": round(v_loss, 4), "val_iou": round(v_iou, 4), "val_dice": round(v_dice, 4),
        })

        if v_dice > best_stage2_dice:
            best_stage2_dice = v_dice
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "arch": "unet", "base": BASE_CHANNELS,
                         "stage": "2a", "val_iou": v_iou, "val_dice": v_dice},
                       str(best_weights_path))
            print(f"  ★ New Best! Dice={v_dice:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"  ⚡ Early stopping at epoch {ep}")
            break

    print(f"\n✅ Stage 2a Complete — Best Val Dice: {best_stage2_dice:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2b: Full Fine-tune (All layers unfrozen, very low LR)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  STAGE 2b: Full Fine-tune (ALL layers, very low LR)")
    print("  Gentle full-model adaptation to real SAR data")
    print("█" * 60)

    # Reload best Stage 2a weights
    ckpt2a = torch.load(str(best_weights_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt2a["state_dict"])
    unfreeze_all(model)

    optimizer2b = torch.optim.AdamW(model.parameters(), lr=LR_STAGE2_FULL, weight_decay=1e-5)
    scheduler2b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2b, T_max=STAGE2_FULL_EPOCHS, eta_min=1e-7)

    patience_counter = 0

    for ep in range(1, STAGE2_FULL_EPOCHS + 1):
        current_epoch += 1
        lr_now = optimizer2b.param_groups[0]["lr"]
        print(f"\n--- [Stage 2b - Full] Epoch {ep}/{STAGE2_FULL_EPOCHS} (LR={lr_now:.6f}) ---")

        t_loss, t_iou, t_dice = train_epoch(model, krest_train_loader, optimizer2b, loss_fn2, device, scaler)
        v_loss, v_iou, v_dice = val_epoch(model, krest_val_loader, loss_fn2, device)
        scheduler2b.step()

        print(f"  Train: Loss={t_loss:.4f}  IoU={t_iou:.4f}  Dice={t_dice:.4f}")
        print(f"  Val:   Loss={v_loss:.4f}  IoU={v_iou:.4f}  Dice={v_dice:.4f}")

        history_rows.append({
            "epoch": current_epoch, "stage": "Stage2b_Full",
            "train_loss": round(t_loss, 4), "train_iou": round(t_iou, 4), "train_dice": round(t_dice, 4),
            "val_loss": round(v_loss, 4), "val_iou": round(v_iou, 4), "val_dice": round(v_dice, 4),
        })

        if v_dice > best_stage2_dice:
            best_stage2_dice = v_dice
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "arch": "unet", "base": BASE_CHANNELS,
                         "stage": "2b", "val_iou": v_iou, "val_dice": v_dice},
                       str(best_weights_path))
            print(f"  ★ New Best! Dice={v_dice:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"  ⚡ Early stopping at epoch {ep}")
            break

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  FINAL EVALUATION ON FULL SCENES")
    print("█" * 60)

    # Reload absolute best weights
    best_ckpt = torch.load(str(best_weights_path), map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["state_dict"])

    scene_iou = evaluate_full_scenes(
        model, ROOT / "Oil Spill Detection Dataset" / "test",
        krest_stats, device, max_scenes=30,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Best Val Dice: {best_stage2_dice:.4f}")
    print(f"  Full-Scene IoU: {scene_iou:.4f}")
    print(f"{'=' * 60}")

    # Save history
    history_csv = output_dir / "train_history.csv"
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"History -> {history_csv}")

    # Save metadata
    meta = {
        "arch": "unet", "base": BASE_CHANNELS,
        "total_epochs": current_epoch,
        "best_val_dice": round(best_stage2_dice, 4),
        "full_scene_iou": round(scene_iou, 4),
        "training_time_minutes": round(elapsed / 60, 1),
        "stages": {
            "stage1_best_dice": round(best_stage1_dice, 4),
            "stage2_best_dice": round(best_stage2_dice, 4),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Export
    export_model(model, output_dir, device)

    print("\n🏁 Pipeline fully completed!")


if __name__ == "__main__":
    main()
