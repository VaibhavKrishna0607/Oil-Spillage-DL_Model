"""
Refined SOS Dataset (Dataset-1) Loader
======================================
Loads 256x256 SAR patches from Dataset-1/ using group-aware split files:
  - split_train.txt (5,255 samples)
  - split_val.txt   (1,615 samples)
  - split_test.txt  (1,200 samples)
Applies domain-calibrated radiometric normalization (clip_2, clip_98, mean, std)
and joint SAR augmentations.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml_model.infer import load_norm_stats


class SOSDataset(Dataset):
    """
    Direct-load Dataset for Refined SOS (Dataset-1).
    Avoids need for pre-generating multi-gigabyte monolithic tensor files.
    Supports in-memory caching of tensors for ultra-fast training epochs.
    """
    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        augment: bool = False,
        max_samples: Optional[int] = None,
        norm_stats: Optional[dict] = None,
        cache_in_ram: bool = True,
    ):
        root = Path(root_dir)
        self.split = split
        self.augment = augment and split == "train"
        self.norm_stats = norm_stats or load_norm_stats(root)

        # Read split file
        split_file = root / f"split_{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found!")

        with open(split_file, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            filenames = filenames[:max_samples]

        self.filenames = filenames
        disk_split = "train" if split == "test" else split
        self.img_dir = root / "Dataset-1" / "Images" / "images" / disk_split
        self.mask_dir = root / "Dataset-1" / "Masks" / "masks" / disk_split

        self.cached_images = []
        self.cached_masks = []
        self.cache_in_ram = cache_in_ram

        if cache_in_ram:
            print(f"[SOSDataset] Caching {len(filenames)} samples in RAM for split '{split}'...")
            for fname in self.filenames:
                img_path = self.img_dir / fname
                mask_path = self.mask_dir / fname

                img_arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
                mask_arr = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127).astype(np.float32)

                # Apply normalization
                clipped = np.clip(img_arr, self.norm_stats["clip_2"], self.norm_stats["clip_98"])
                normed = (clipped - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-7)

                self.cached_images.append(torch.from_numpy(normed).unsqueeze(0).float()) # (1, 256, 256)
                self.cached_masks.append(torch.from_numpy(mask_arr).unsqueeze(0).float())

            print(f"[SOSDataset] Cached {len(self.cached_images)} tensors successfully.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_ram:
            img = self.cached_images[idx]
            mask = self.cached_masks[idx]
        else:
            fname = self.filenames[idx]
            img_arr = np.array(Image.open(self.img_dir / fname).convert("L"), dtype=np.float32)
            mask_arr = (np.array(Image.open(self.mask_dir / fname).convert("L"), dtype=np.uint8) > 127).astype(np.float32)

            clipped = np.clip(img_arr, self.norm_stats["clip_2"], self.norm_stats["clip_98"])
            normed = (clipped - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-7)

            img = torch.from_numpy(normed).unsqueeze(0).float()
            mask = torch.from_numpy(mask_arr).unsqueeze(0).float()

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = torch.rot90(img, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
            # Multiplicative speckle noise (Gamma-like simulation)
            if torch.rand(1).item() < 0.3:
                speckle = torch.randn_like(img) * 0.1
                img = img + speckle

        return img.contiguous(), mask.contiguous()


if __name__ == "__main__":
    ds = SOSDataset(split="train", max_samples=20)
    img, mask = ds[0]
    print("SOS sample image shape:", img.shape, "Mask shape:", mask.shape)
    print("Min:", img.min().item(), "Max:", img.max().item())
    print("SOSDataset test passed!")
