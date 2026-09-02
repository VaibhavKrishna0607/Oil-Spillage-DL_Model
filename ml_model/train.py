import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.packed_dataset import PackedSARDataset
from ml_model.model import UNet
from ml_model.loss import BCEDiceLoss

torch.manual_seed(42)


def calculate_metrics(y_pred, y_true, threshold=0.5):
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


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_iou = total_dice = 0.0
    count = 0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        iou, dice = calculate_metrics(outputs, masks)
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        count += 1
        if batch_idx % max(1, len(loader) // 5) == 0:
            print("  Batch {}/{}: Loss = {:.4f}, IoU = {:.4f}, Dice = {:.4f}".format(
                batch_idx, len(loader), loss.item(), iou, dice
            ))
    return total_loss / count, total_iou / count, total_dice / count


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
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


def export_model(model, output_dir, device, n_channels):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, n_channels, 256, 256, device=device)
    try:
        traced_path = output_dir / "best_model.pt"
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(traced_path))
        print("[Export] TorchScript -> {}".format(traced_path))
    except Exception as e:
        print("[Export Warning] TorchScript failed: {}".format(e))
    try:
        onnx_path = output_dir / "best_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print("[Export] ONNX -> {}".format(onnx_path))
    except Exception as e:
        print("[Export Warning] ONNX failed: {}".format(e))


def main():
    parser = argparse.ArgumentParser(description="Train SAR oil-spill U-Net on packed SOS tensors")
    parser.add_argument("--data-dir", type=str, default="dataset_ready")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=64, help="U-Net width (32 for CPU lab, 64 for GPU)")
    parser.add_argument("--quick", action="store_true", help="CPU lab run: smaller net, subset, few epochs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="ml_model/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.quick:
        args.base = 32
        args.epochs = min(args.epochs, 8)
        args.batch_size = min(args.batch_size, 4)
        train_max, val_max = 400, 80
        print("[Quick] base=32, train=400, val=80, epochs={}".format(args.epochs))
    else:
        train_max = val_max = None

    if args.dry_run:
        train_max, val_max = 8, 4
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        print("[Dry-Run] 8 train / 4 val, 1 epoch")

    train_ds = PackedSARDataset(args.data_dir, "train", augment=True, max_samples=train_max)
    val_ds = PackedSARDataset(args.data_dir, "val", augment=False, max_samples=val_max)

    num_workers = 0 if (device.type == "cpu" or os.name == "nt") else 4
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    print("Train samples:", len(train_ds))
    print("Val samples  :", len(val_ds))

    n_channels = int(train_ds.images.shape[1])
    model = UNet(n_channels=n_channels, n_classes=1, bilinear=False, base=args.base).to(device)
    loss_fn = BCEDiceLoss(alpha=0.5, pos_weight=2.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    history_path = output_dir / "train_history.csv"

    best_val_dice = -1.0
    history_rows = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print("\n--- Epoch {}/{} ---".format(epoch, args.epochs))
        train_loss, train_iou, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print("Train: Loss = {:.4f}, IoU = {:.4f}, Dice = {:.4f}".format(train_loss, train_iou, train_dice))
        val_loss, val_iou, val_dice = val_epoch(model, val_loader, loss_fn, device)
        print("Val  : Loss = {:.4f}, IoU = {:.4f}, Dice = {:.4f}".format(val_loss, val_iou, val_dice))
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        })
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            payload = {
                "state_dict": model.state_dict(),
                "n_channels": n_channels,
                "base": args.base,
                "val_dice": best_val_dice,
                "val_iou": val_iou,
            }
            torch.save(payload, str(best_path))
            print("  [Checkpoint] val Dice {:.4f} -> {}".format(best_val_dice, best_path))

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)

    meta = {
        "best_val_dice": best_val_dice,
        "n_channels": n_channels,
        "base": args.base,
        "epochs": args.epochs,
        "quick": args.quick,
        "seconds": round(time.time() - start_time, 2),
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining completed in {:.2f}s. Best val Dice = {:.4f}".format(time.time() - start_time, best_val_dice))
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        export_model(model, args.output_dir, device, n_channels)


if __name__ == "__main__":
    main()
