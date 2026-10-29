"""
Dual-Dataset Two-Stage Training Pipeline
=========================================
Implements the creative two-stage curriculum strategy:
  Stage 1: Foundational Pre-training on Dataset-1 (Refined SOS 256x256 patches)
           Objective: Master SAR oil slick boundaries and texture features using ComboLoss.
  Stage 2: Domain Adaptation & Look-Alike Suppression on Dataset-2 (Krestenitis Real-Life 1250x650 scenes)
           Objective: Fine-tune using Focal Tversky Loss to sharply penalize false alarms on look-alikes.

Exports:
  - best_model.pth (PyTorch state dict)
  - best_model.pt  (TorchScript serialized)
  - best_model.onnx (ONNX dynamic batch model)
  - train_history.csv (Live metrics across all epochs)
"""

import argparse
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.sos_dataset import SOSDataset
from data_pipeline.krestenitis_dataset import KrestenitisTileDataset, load_krest_stats
from ml_model.model import UNet
from ml_model.model_enhanced import AttentionUNet
from ml_model.advanced_losses import ComboLoss, FocalTverskyLoss
from ml_model.infer import load_norm_stats, predict_array, mask_iou, krestenitis_oil_mask
from ml_model.postprocessing import gaussian_sliding_window_inference, morphological_clean

torch.manual_seed(42)
np.random.seed(42)


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


def train_epoch(model, loader, optimizer, loss_fn, device, scheduler=None) -> Tuple[float, float, float]:
    model.train()
    total_loss = total_iou = total_dice = 0.0
    count = 0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        iou, dice = calculate_metrics(outputs, masks)
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        count += 1

        if batch_idx % max(1, len(loader) // 4) == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: Loss = {loss.item():.4f}, IoU = {iou:.4f}, Dice = {dice:.4f}")

    if scheduler is not None:
        scheduler.step()

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
def evaluate_krestenitis_full_scenes(
    model: torch.nn.Module,
    krest_test_dir: Path,
    krest_stats: dict,
    device: torch.device,
    max_scenes: int = 15,
) -> float:
    """Evaluate sliding-window performance on real-life 1250x650 Krestenitis scenes."""
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
        iou = mask_iou(cleaned.astype(bool), gt.astype(bool))
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def export_model(model: torch.nn.Module, output_dir: Path, device: torch.device, n_channels: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, n_channels, 256, 256, device=device)

    # 1. TorchScript
    try:
        traced_path = output_dir / "best_model.pt"
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(traced_path))
        print(f"[Export] TorchScript -> {traced_path}")
    except Exception as e:
        print(f"[Export Warning] TorchScript failed: {e}")

    # 2. ONNX
    try:
        onnx_path = output_dir / "best_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"[Export] ONNX -> {onnx_path}")
    except Exception as e:
        safe_msg = str(e).encode("ascii", "replace").decode()
        print(f"[Export Warning] ONNX export failed: {safe_msg}")


def main():
    parser = argparse.ArgumentParser(description="Dual-Dataset Two-Stage Training for SAR Oil Spill Detection")
    parser.add_argument("--arch", type=str, default="attention", choices=["standard", "attention"], help="Model architecture")
    parser.add_argument("--base", type=int, default=32, help="Channel width (32 for CPU/lab, 64 for GPU)")
    parser.add_argument("--epochs-stage1", type=int, default=5, help="Epochs for SOS Stage 1")
    parser.add_argument("--epochs-stage2", type=int, default=4, help="Epochs for Krestenitis Stage 2")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr-stage1", type=float, default=1e-4, help="Learning rate for Stage 1")
    parser.add_argument("--lr-stage2", type=float, default=2e-5, help="Learning rate for Stage 2")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset of samples for fast iteration")
    parser.add_argument("--dry-run", action="store_true", help="Dry run test (1 batch each)")
    parser.add_argument("--output-dir", type=str, default="ml_model/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Dual-Dataset Training Pipeline ===")
    print(f"Device:       {device}")
    print(f"Architecture: {args.arch} (base={args.base})")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_csv = output_dir / "train_history.csv"
    meta_json = output_dir / "train_meta.json"
    best_weights_path = output_dir / "best_model.pth"

    # Quick/Dry-run parameter caps
    sos_max_train = 500 if args.quick else (16 if args.dry_run else None)
    sos_max_val = 100 if args.quick else (8 if args.dry_run else None)
    krest_max_train = 200 if args.quick else (16 if args.dry_run else None)
    krest_max_val = 50 if args.quick else (8 if args.dry_run else None)

    if args.quick:
        args.epochs_stage1 = min(args.epochs_stage1, 3)
        args.epochs_stage2 = min(args.epochs_stage2, 2)
        args.batch_size = min(args.batch_size, 8)
        print(f"[Quick Mode] Stage1 epochs={args.epochs_stage1} (max_train={sos_max_train}), Stage2 epochs={args.epochs_stage2} (max_train={krest_max_train})")
    elif args.dry_run:
        args.epochs_stage1 = 1
        args.epochs_stage2 = 1
        args.batch_size = 4
        print(f"[Dry-Run Mode] 1 epoch each, minimal samples")

    # Load Normalization Stats
    sos_stats = load_norm_stats(ROOT)
    krest_stats = load_krest_stats(ROOT)

    # Initialize Architecture
    if args.arch == "attention":
        model = AttentionUNet(n_channels=1, n_classes=1, base=args.base).to(device)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=False, base=args.base).to(device)

    history_rows = []
    best_combined_score = -1.0
    start_time = time.time()
    current_epoch = 0

    # -------------------------------------------------------------------------
    # STAGE 1: Pre-training on Dataset-1 (SOS Refined 256x256 patches)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STAGE 1: Pre-training on Dataset-1 (Refined SOS)")
    print("="*50)

    sos_train_ds = SOSDataset(root_dir=ROOT, split="train", augment=True, max_samples=sos_max_train, norm_stats=sos_stats, cache_in_ram=True)
    sos_val_ds = SOSDataset(root_dir=ROOT, split="val", augment=False, max_samples=sos_max_val, norm_stats=sos_stats, cache_in_ram=True)

    num_workers = 0
    sos_train_loader = DataLoader(sos_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    sos_val_loader = DataLoader(sos_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    optimizer_stage1 = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=1e-4)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=args.epochs_stage1, eta_min=1e-6)
    loss_fn_stage1 = ComboLoss(w_bce=0.3, w_dice=0.4, w_focal=0.3, pos_weight=2.9)

    for ep in range(1, args.epochs_stage1 + 1):
        current_epoch += 1
        print(f"\n--- [Stage 1 - SOS] Epoch {ep}/{args.epochs_stage1} (Total Epoch {current_epoch}) ---")
        t_loss, t_iou, t_dice = train_epoch(model, sos_train_loader, optimizer_stage1, loss_fn_stage1, device, scheduler_stage1)
        v_loss, v_iou, v_dice = val_epoch(model, sos_val_loader, loss_fn_stage1, device)
        print(f"Train: Loss={t_loss:.4f}, IoU={t_iou:.4f}, Dice={t_dice:.4f}")
        print(f"Val  : Loss={v_loss:.4f}, IoU={v_iou:.4f}, Dice={v_dice:.4f}")

        history_rows.append({
            "epoch": current_epoch,
            "stage": "Stage 1 (SOS)",
            "train_loss": round(t_loss, 4),
            "train_iou": round(t_iou, 4),
            "train_dice": round(t_dice, 4),
            "val_loss": round(v_loss, 4),
            "val_iou": round(v_iou, 4),
            "val_dice": round(v_dice, 4),
        })

    # Save Stage 1 checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "base": args.base,
        "stage": 1,
        "val_iou": v_iou,
        "val_dice": v_dice,
    }, str(output_dir / "stage1_sos_best.pth"))
    print(f"\n[Stage 1 Completed] Best SOS Val IoU: {v_iou:.4f}, Dice: {v_dice:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 2: Domain Adaptation & Look-Alike Suppression (Krestenitis Dataset)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STAGE 2: Domain Adaptation on Dataset-2 (Krestenitis Real-Life)")
    print("="*50)

    krest_train_ds = KrestenitisTileDataset(
        root_dir="Oil Spill Detection Dataset",
        split="train",
        tile_size=256,
        stride=128,
        augment=True,
        max_samples=krest_max_train,
        norm_stats=krest_stats,
    )
    krest_val_ds = KrestenitisTileDataset(
        root_dir="Oil Spill Detection Dataset",
        split="test",
        tile_size=256,
        stride=256,
        augment=False,
        max_samples=krest_max_val,
        norm_stats=krest_stats,
    )

    krest_train_loader = DataLoader(krest_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    krest_val_loader = DataLoader(krest_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Use Focal Tversky Loss with beta=0.7 to heavily penalize look-alike false alarms
    loss_fn_stage2 = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    optimizer_stage2 = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=1e-4)

    for ep in range(1, args.epochs_stage2 + 1):
        current_epoch += 1
        print(f"\n--- [Stage 2 - Krestenitis] Epoch {ep}/{args.epochs_stage2} (Total Epoch {current_epoch}) ---")
        t_loss, t_iou, t_dice = train_epoch(model, krest_train_loader, optimizer_stage2, loss_fn_stage2, device)
        v_loss, v_iou, v_dice = val_epoch(model, krest_val_loader, loss_fn_stage2, device)
        print(f"Train: Loss={t_loss:.4f}, IoU={t_iou:.4f}, Dice={t_dice:.4f}")
        print(f"Val  : Loss={v_loss:.4f}, IoU={v_iou:.4f}, Dice={v_dice:.4f}")

        history_rows.append({
            "epoch": current_epoch,
            "stage": "Stage 2 (Krestenitis)",
            "train_loss": round(t_loss, 4),
            "train_iou": round(t_iou, 4),
            "train_dice": round(t_dice, 4),
            "val_loss": round(v_loss, 4),
            "val_iou": round(v_iou, 4),
            "val_dice": round(v_dice, 4),
        })

        # Checkpoint if improving
        if v_dice > best_combined_score:
            best_combined_score = v_dice
            payload = {
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "n_channels": 1,
                "base": args.base,
                "val_dice": best_combined_score,
                "val_iou": v_iou,
                "stage": 2,
            }
            torch.save(payload, str(best_weights_path))
            print(f"  [Checkpoint] New Best Model Saved -> {best_weights_path} (Val Dice: {best_combined_score:.4f})")

    # Final Benchmark on full scenes
    print("\n" + "="*50)
    print("RUNNING FINAL FULL-SCENE BENCHMARK...")
    print("="*50)
    max_eval_scenes = 2 if args.dry_run else (4 if args.quick else 10)
    krest_scene_iou = evaluate_krestenitis_full_scenes(
        model=model,
        krest_test_dir=ROOT / "Oil Spill Detection Dataset" / "test",
        krest_stats=krest_stats,
        device=device,
        max_scenes=max_eval_scenes,
    )
    print(f"Krestenitis Full-Scene Sliding-Window Mean IoU: {krest_scene_iou:.4f}")

    # Write training history CSV
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"Training history saved -> {history_csv}")

    # Write metadata JSON
    meta = {
        "arch": args.arch,
        "base": args.base,
        "total_epochs": current_epoch,
        "best_val_dice": round(best_combined_score, 4),
        "krestenitis_full_scene_iou": round(krest_scene_iou, 4),
        "training_time_seconds": round(time.time() - start_time, 2),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved -> {meta_json}")

    # Export Model Formats
    export_model(model, output_dir, device, n_channels=1)
    print("\nPipeline execution fully completed!")


if __name__ == "__main__":
    main()
