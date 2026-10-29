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
from PIL import Image, ImageEnhance, ImageFilter
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


def _random_rotation(img: Image.Image, mask: Image.Image,
                     angle_range: Tuple[float, float] = (-15, 15),
                     p: float = 0.5):
    """Random rotation by arbitrary angle (not just 90° multiples)."""
    if random.random() < p:
        angle = random.uniform(*angle_range)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    return img, mask


def _random_affine(img: Image.Image, mask: Image.Image,
                   translate_range: Tuple[float, float] = (0.1, 0.1),
                   scale_range: Tuple[float, float] = (0.9, 1.1),
                   p: float = 0.5):
    """Random affine transformation (translation + scaling)."""
    if random.random() < p:
        tx = random.uniform(-translate_range[0], translate_range[0])
        ty = random.uniform(-translate_range[1], translate_range[1])
        scale = random.uniform(*scale_range)
        
        width, height = img.size
        # Apply scaling
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.BILINEAR)
        mask = mask.resize(new_size, Image.NEAREST)
        
        # Apply translation
        img = img.transform((width, height), Image.AFFINE,
                           (1, 0, tx * width, 0, 1, ty * height),
                           Image.BILINEAR, fillcolor=0)
        mask = mask.transform((width, height), Image.AFFINE,
                             (1, 0, tx * width, 0, 1, ty * height),
                             Image.NEAREST, fillcolor=0)
    return img, mask


def _random_elastic_transform(img: Image.Image, mask: Image.Image,
                             alpha_range: Tuple[float, float] = (0.5, 1.5),
                             sigma_range: Tuple[float, float] = (0.1, 0.3),
                             p: float = 0.3):
    """Elastic deformation for more robust spatial invariance (simplified version)."""
    # Simplified version without scipy - skip for now to avoid dependency issues
    return img, mask


def _random_gaussian_blur(img: Image.Image, mask: Image.Image,
                         kernel_range: Tuple[int, int] = (3, 7),
                         sigma_range: Tuple[float, float] = (0.1, 2.0),
                         p: float = 0.3):
    """Random Gaussian blur for smoothing and noise reduction."""
    if random.random() < p:
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
        sigma = random.uniform(*sigma_range)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img, mask


def _random_sharpness(img: Image.Image, mask: Image.Image,
                     factor_range: Tuple[float, float] = (0.5, 1.5),
                     p: float = 0.3):
    """Random sharpness adjustment."""
    if random.random() < p:
        factor = random.uniform(*factor_range)
        img = ImageEnhance.Sharpness(img).enhance(factor)
    return img, mask


def _cutout(img: Image.Image, mask: Image.Image,
           n_holes: int = 1,
           length_range: Tuple[int, int] = (16, 64),
           p: float = 0.3):
    """Cutout augmentation - randomly mask out rectangular regions."""
    if random.random() < p:
        width, height = img.size
        img_arr = np.array(img)
        mask_arr = np.array(mask)
        
        # Handle both grayscale and color images
        if len(img_arr.shape) == 2:
            img_arr = img_arr[:, :, np.newaxis]
        
        for _ in range(n_holes):
            length = random.randint(*length_range)
            x = random.randint(0, max(1, width - length))
            y = random.randint(0, max(1, height - length))
            
            # Ensure we don't go out of bounds
            x_end = min(x + length, width)
            y_end = min(y + length, height)
            
            img_arr[y:y_end, x:x_end] = 0
            mask_arr[y:y_end, x:x_end] = 0
        
        # Remove extra dimension if it was added
        if img_arr.shape[2] == 1:
            img_arr = img_arr[:, :, 0]
        
        img = Image.fromarray(img_arr.astype(np.uint8))
        mask = Image.fromarray(mask_arr.astype(np.uint8))
    
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
      - Arbitrary rotation          (additional angle variability)
      - Brightness / contrast jitter (sensor calibration variation)
      - Gaussian noise              (SAR speckle simulation)
      - Affine transforms           (translation + scaling)
      - Gaussian blur               (smoothing variations)
      - Sharpness adjustment        (edge enhancement)
      - Cutout                      (occlusion robustness)
    """

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Basic geometric transforms
        img, mask = _random_hflip(img, mask, p=0.5)
        img, mask = _random_vflip(img, mask, p=0.5)
        img, mask = _random_rotate90(img, mask, p=0.5)
        img, mask = _random_rotation(img, mask, angle_range=(-15, 15), p=0.4)
        img, mask = _random_affine(img, mask, translate_range=(0.1, 0.1), 
                                   scale_range=(0.9, 1.1), p=0.4)
        
        # Photometric transforms
        img, mask = _random_brightness(img, mask, factor_range=(0.7, 1.3))
        img, mask = _random_contrast(img, mask, factor_range=(0.8, 1.2))
        img, mask = _random_sharpness(img, mask, factor_range=(0.7, 1.3), p=0.3)
        img, mask = _random_gaussian_blur(img, mask, kernel_range=(3, 7), 
                                        sigma_range=(0.1, 2.0), p=0.3)
        img, mask = _gaussian_noise(img, mask, std_range=(0.0, 0.05), p=0.4)
        
        # Occlusion
        img, mask = _cutout(img, mask, n_holes=1, length_range=(16, 48), p=0.2)
        
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
