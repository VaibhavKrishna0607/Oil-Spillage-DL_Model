"""
Post-Processing & Refinement for SAR Oil Spill Detection
========================================================
Implements Strategy 4 (Prompt 4.2) from prompts.md:
  1. Morphological opening (removes small false-positive specks)
  2. Morphological closing (fills internal radar sensor holes in real slicks)
  3. Area-based filtering (removes tiny stray detections)
  4. Gaussian-weighted sliding-window stitching for 1250x650 full SAR scenes
"""

import math
from typing import Optional, Tuple
import cv2
import numpy as np
import torch


def morphological_clean(
    binary_mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
    min_area_px: int = 15,
) -> np.ndarray:
    """
    Clean raw binary predictions using morphological operations.
    - Opening: Removes isolated 1-3 px noise specks.
    - Closing: Bridges small internal pinholes in real slicks.
    - Connected Components: Discards components smaller than min_area_px.
    """
    mask = (binary_mask > 0).astype(np.uint8) * 255

    # Opening: erode then dilate
    if open_kernel_size > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Closing: dilate then erode
    if close_kernel_size > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Area filtering via connected components
    if min_area_px > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                cleaned[labels == i] = 255
        mask = cleaned

    return (mask > 0).astype(np.uint8)


def get_gaussian_weight_window(tile_size: int, sigma: float = 0.25) -> np.ndarray:
    """
    Generate a 2D Gaussian blending weight window.
    Pixels near the center receive higher weight than boundary pixels,
    preventing edge seam artifacts during sliding-window stitching.
    """
    ax = np.linspace(-(tile_size - 1) / 2.0, (tile_size - 1) / 2.0, tile_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma * tile_size))
    weight_2d = np.outer(gauss, gauss)
    return weight_2d / np.max(weight_2d)


@torch.no_grad()
def gaussian_sliding_window_inference(
    model: torch.nn.Module,
    gray: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    tile: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Run sliding-window inference with Gaussian window blending.
    Produces smooth, artifact-free probability maps on full-scene images (e.g. 1250x650).
    """
    model.eval()
    h, w = gray.shape

    # Fast path if exact tile size
    if h == tile and w == tile:
        clipped = np.clip(gray, norm_stats["clip_2"], norm_stats["clip_98"])
        normed = (clipped - norm_stats["mean"]) / (norm_stats["std"] + 1e-7)
        tensor = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0).to(device)
        pred = model(tensor).squeeze().cpu().numpy()
        return pred

    stride = max(1, tile - overlap)
    gaussian_weight = get_gaussian_weight_window(tile)

    def needed(length: int) -> int:
        if length <= tile:
            return tile
        n = math.ceil((length - tile) / stride)
        return tile + n * stride

    pad_h = needed(h) - h
    pad_w = needed(w) - w
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")

    acc = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)

    ph, pw = padded.shape
    crops = []
    coords = []

    for y in range(0, ph - tile + 1, stride):
        for x0 in range(0, pw - tile + 1, stride):
            crop = padded[y : y + tile, x0 : x0 + tile]
            clipped = np.clip(crop, norm_stats["clip_2"], norm_stats["clip_98"])
            normed = (clipped - norm_stats["mean"]) / (norm_stats["std"] + 1e-7)
            crops.append(torch.from_numpy(normed).float().unsqueeze(0))
            coords.append((y, x0))

    # Batch crops for speed
    batch_size = 16
    for i in range(0, len(crops), batch_size):
        batch_crops = torch.stack(crops[i : i + batch_size]).to(device)  # (B, 1, H, W)
        preds = model(batch_crops).squeeze(1).cpu().numpy()  # (B, H, W)
        if preds.ndim == 2:
            preds = preds[np.newaxis, ...]

        for b in range(preds.shape[0]):
            y, x0 = coords[i + b]
            acc[y : y + tile, x0 : x0 + tile] += preds[b] * gaussian_weight
            weights[y : y + tile, x0 : x0 + tile] += gaussian_weight

    weights = np.maximum(weights, 1e-6)
    stitched = acc / weights
    return stitched[:h, :w]


@torch.no_grad()
def dihedral_tta_inference(
    model: torch.nn.Module,
    gray: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    tile: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """Average predictions over the eight flip/rotation symmetries."""
    predictions = []
    for rotate in range(4):
        for mirrored in (False, True):
            transformed = np.rot90(gray, rotate)
            if mirrored:
                transformed = np.fliplr(transformed)
            prediction = gaussian_sliding_window_inference(
                model, transformed.copy(), norm_stats, device, tile=tile, overlap=overlap
            )
            if mirrored:
                prediction = np.fliplr(prediction)
            prediction = np.rot90(prediction, -rotate)
            predictions.append(prediction)
    return np.mean(predictions, axis=0, dtype=np.float32)


if __name__ == '__main__':
    # Unit test
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 1 # Real slick
    dummy_mask[50, 50] = 0 # Single hole
    dummy_mask[10, 10] = 1 # Stray noise pixel
    dummy_mask[10, 11] = 1

    cleaned = morphological_clean(dummy_mask, open_kernel_size=3, close_kernel_size=5, min_area_px=10)
    print("Morphological test:")
    print(f"Original positive pixels: {dummy_mask.sum()}")
    print(f"Cleaned positive pixels:  {cleaned.sum()}")
    assert cleaned[10, 10] == 0, "Noise speck should be removed!"
    assert cleaned[50, 50] == 1, "Internal hole should be closed!"
    print("All post-processing unit tests passed!")
