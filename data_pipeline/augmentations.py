"""
augmentations.py
================
All data augmentation transforms for the SAR oil-spill pipeline.

These are *joint* transforms — they apply the same random operation to both
the image and its mask so spatial alignment is never broken.

Usage
-----
    from data_pipeline.augmentations import get_train_transform, get_val_transform

    train_ds = SARSpillDataset(root, split='train', transform=get_train_transform())
    val_ds   = SARSpillDataset(root, split='val',   transform=get_val_transform())
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch

# -------------------------------------------------------------------------
# ImageNet normalisation constants (same as in dataset.py)
# -------------------------------------------------------------------------
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================================
# Primitive joint transforms
# =========================================================================

def _to_tensors(img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert PIL Image (RGB) + PIL mask (L) to normalised tensors."""
    img_arr  = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
    img_arr  = (img_arr - IMG_MEAN) / IMG_STD
    img_t    = torch.from_numpy(img_arr).permute(2, 0, 1)           # (3,H,W)

    mask_arr  = np.array(mask.convert('L'), dtype=np.float32)
    mask_bin  = (mask_arr > 127).astype(np.float32)
    mask_t    = torch.from_numpy(mask_bin).unsqueeze(0)              # (1,H,W)

    return img_t, mask_t


def _random_hflip(img: Image.Image, mask: Image.Image, p: float = 0.5):
    if random.random() < p:
        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def _random_vflip(img: Image.Image, mask: Image.Image, p: float = 0.5):
    if random.random() < p:
        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


def _random_rotate90(img: Image.Image, mask: Image.Image, p: float = 0.5):
    """Rotate by a random multiple of 90 degrees."""
    if random.random() < p:
        k = random.choice([1, 2, 3])
        angle = k * 90
        img  = img.rotate(angle)
        mask = mask.rotate(angle)
    return img, mask


def _random_brightness(img: Image.Image, mask: Image.Image,
                       factor_range: Tuple[float, float] = (0.7, 1.3)):
    """Randomly adjust image brightness (mask untouched)."""
    factor = random.uniform(*factor_range)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return img, mask


def _random_contrast(img: Image.Image, mask: Image.Image,
                     factor_range: Tuple[float, float] = (0.8, 1.2)):
    """Randomly adjust image contrast (mask untouched)."""
    factor = random.uniform(*factor_range)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img, mask


def _gaussian_noise(img: Image.Image, mask: Image.Image,
                    std_range: Tuple[float, float] = (0.0, 0.05),
                    p: float = 0.3):
    """
    Add Gaussian noise to simulate SAR speckle variation.
    Only applied to the image; mask is unchanged.
    """
    if random.random() < p:
        arr   = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, random.uniform(*std_range), arr.shape).astype(np.float32)
        arr   = np.clip(arr + noise, 0.0, 1.0)
        img   = Image.fromarray((arr * 255).astype(np.uint8))
    return img, mask


# =========================================================================
# Public transform factories
# =========================================================================

class TrainTransform:
    """
    Full augmentation pipeline applied during training.

    Augmentations chosen for SAR imagery:
      - Horizontal / vertical flip  (oceans look the same either way)
      - 90-degree rotation          (no preferred orientation from orbit)
      - Brightness / contrast jitter (sensor calibration variation)
      - Gaussian noise              (SAR speckle simulation)
    """

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = _random_hflip(img, mask, p=0.5)
        img, mask = _random_vflip(img, mask, p=0.5)
        img, mask = _random_rotate90(img, mask, p=0.5)
        img, mask = _random_brightness(img, mask, factor_range=(0.75, 1.25))
        img, mask = _random_contrast(img, mask, factor_range=(0.8, 1.2))
        img, mask = _gaussian_noise(img, mask, std_range=(0.0, 0.04), p=0.3)
        return _to_tensors(img, mask)


class ValTransform:
    """
    No augmentation — only tensor conversion + normalisation.
    Used for validation and test sets.
    """

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _to_tensors(img, mask)


def get_train_transform() -> TrainTransform:
    return TrainTransform()


def get_val_transform() -> ValTransform:
    return ValTransform()
