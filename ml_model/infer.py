"""Shared inference: SAR preprocess, U-Net forward, tiling, geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from ml_model.model import UNet
try:
    from ml_model.model_enhanced import AttentionUNet
except ImportError:
    AttentionUNet = None

DEFAULT_STATS = {
    "clip_2": 14.0,
    "clip_98": 234.0,
    "mean": 108.27242279052734,
    "std": 53.41604232788086,
}


def load_norm_stats(root: Path) -> dict:
    for candidate in (
        root / "dataset_ready" / "norm_stats.json",
        root / "artifacts" / "norm_stats.json",
    ):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return DEFAULT_STATS


def load_checkpoint(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    path = Path(model_path)
    ext = path.suffix.lower()
    meta = {"n_channels": 1, "base": 64}

    if ext == ".onnx":
        return None, {"backend": "onnx", "path": str(path)}  # type: ignore

    if ext == ".pt":
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model, {"backend": "torchscript", "n_channels": 1, "base": None}

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        n_channels = int(ckpt.get("n_channels", 1))
        base = int(ckpt.get("base", 64))
        arch = ckpt.get("arch", "standard")
        if arch == "attention" and AttentionUNet is not None:
            model = AttentionUNet(n_channels=n_channels, n_classes=1, base=base)
        else:
            model = UNet(n_channels=n_channels, n_classes=1, bilinear=False, base=base)
        
        # Load with strict=False to handle attention layer mismatches
        try:
            model.load_state_dict(ckpt["state_dict"])
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e) and "attn" in str(e):
                print(f"⚠️ Loading with strict=False due to attention layer mismatch")
                model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                raise e
                
        meta = {"backend": "pytorch", "n_channels": n_channels, "base": base, "arch": arch, "val_dice": ckpt.get("val_dice"), "val_iou": ckpt.get("val_iou")}
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=False, base=64)
        model.load_state_dict(ckpt)
        meta = {"backend": "pytorch", "n_channels": 1, "base": 64, "arch": "standard"}
    model.to(device)
    model.eval()
    return model, meta


def image_to_gray_float(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"), dtype=np.float32)


def normalize_sar(gray: np.ndarray, stats: dict) -> np.ndarray:
    clipped = np.clip(gray, stats["clip_2"], stats["clip_98"])
    return (clipped - stats["mean"]) / (stats["std"] + 1e-7)


def tensor_from_gray(normed: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_array(
    model,
    gray: np.ndarray,
    stats: dict,
    device: torch.device,
    tile: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """Return probability map same HxW as gray. Uses sliding window if larger than tile."""
    h, w = gray.shape
    if h == tile and w == tile:
        x = tensor_from_gray(normalize_sar(gray, stats)).to(device)
        return model(x).squeeze().cpu().numpy()

    stride = max(1, tile - overlap)

    def needed(length: int) -> int:
        if length <= tile:
            return tile
        n = math.ceil((length - tile) / stride)
        return tile + n * stride

    pad_h = needed(h) - h
    pad_w = needed(w) - w
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")
    acc = np.zeros_like(padded, dtype=np.float32)
    weight = np.zeros_like(padded, dtype=np.float32)

    ph, pw = padded.shape
    for y in range(0, ph - tile + 1, stride):
        for x0 in range(0, pw - tile + 1, stride):
            crop = padded[y : y + tile, x0 : x0 + tile]
            pred = model(tensor_from_gray(normalize_sar(crop, stats)).to(device)).squeeze().cpu().numpy()
            acc[y : y + tile, x0 : x0 + tile] += pred
            weight[y : y + tile, x0 : x0 + tile] += 1.0

    weight = np.maximum(weight, 1e-6)
    stitched = acc / weight
    return stitched[:h, :w]


def extract_geometric_properties(mask: np.ndarray, pixel_resolution_m: float = 10.0, min_area_px: int = 10) -> List[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slicks = []
    for idx, contour in enumerate(contours):
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px:
            continue
        perimeter_px = cv2.arcLength(contour, closed=True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = float(M["m10"] / M["m00"])
            cY = float(M["m01"] / M["m00"])
        else:
            cX, cY = 0.0, 0.0
        x, y, w, h = cv2.boundingRect(contour)
        eccentricity = 0.0
        orientation_deg = 0.0
        major_axis_px = float(max(w, h))
        minor_axis_px = float(min(w, h))
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_cx, _cy), (axis_a, axis_b), angle = ellipse
                major_axis_px = max(axis_a, axis_b)
                minor_axis_px = min(axis_a, axis_b)
                orientation_deg = float(angle)
                if major_axis_px > 0:
                    ratio = minor_axis_px / major_axis_px
                    eccentricity = math.sqrt(max(0.0, 1.0 - ratio ** 2))
            except Exception:
                pass
        pixel_area_m2 = pixel_resolution_m * pixel_resolution_m
        area_m2 = float(area_px * pixel_area_m2)
        pts = contour.squeeze(1).tolist()
        if len(pts) < 3:
            continue
        slicks.append({
            "slick_id": idx + 1,
            "area_px": int(area_px),
            "area_m2": round(area_m2, 2),
            "area_km2": round(area_m2 / 1e6, 4),
            "perimeter_px": round(float(perimeter_px), 2),
            "perimeter_m": round(float(perimeter_px * pixel_resolution_m), 2),
            "centroid": {"x": round(cX, 2), "y": round(cY, 2)},
            "bbox": {"min_x": int(x), "min_y": int(y), "max_x": int(x + w), "max_y": int(y + h)},
            "eccentricity": round(float(eccentricity), 4),
            "orientation_deg": round(float(orientation_deg), 2),
            "polygon": pts,
        })
    return slicks


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-7) / (union + 1e-7))


def overlay_rgb(gray: np.ndarray, binary: np.ndarray, slicks: List[dict]) -> np.ndarray:
    vis = np.stack([gray, gray, gray], axis=-1)
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    color = vis.copy()
    color[binary > 0] = (0, 90, 200)
    blended = cv2.addWeighted(vis, 0.65, color, 0.35, 0)
    for s in slicks:
        poly = np.array(s["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(blended, [poly], -1, (220, 70, 40), 2)
        cx, cy = int(s["centroid"]["x"]), int(s["centroid"]["y"])
        cv2.circle(blended, (cx, cy), 4, (255, 220, 0), -1)
    return blended


def krestenitis_oil_mask(label_path: Path) -> np.ndarray:
    """Class 1 = oil spill. All other classes (sea, look-alike, ship, land) = 0."""
    arr = np.array(Image.open(label_path))
    if arr.ndim == 3:
        # RGB palette: cyan (0,255,255) is oil
        oil = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255)
        return oil.astype(np.uint8)
    return (arr == 1).astype(np.uint8)
