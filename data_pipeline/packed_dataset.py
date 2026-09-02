"""
PackedSARDataset
================
Loads the Stage-3 packed tensors from dataset_ready/.

Tensors are already:
  - grayscale (1, 256, 256)
  - clipped + z-scored with train-only SAR stats
  - group-aware split
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class PackedSARDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset_ready",
        split: str = "train",
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        data_dir = Path(data_dir)
        images_path = data_dir / f"{split}_images.pt"
        masks_path = data_dir / f"{split}_masks.pt"
        if not images_path.exists() or not masks_path.exists():
            raise FileNotFoundError(
                "Packed tensors not found in {}. Expected {}_images.pt and {}_masks.pt".format(
                    data_dir, split, split
                )
            )

        self.images = torch.load(images_path, map_location="cpu", weights_only=True)
        self.masks = torch.load(masks_path, map_location="cpu", weights_only=True)
        if self.images.ndim != 4 or self.masks.ndim != 4:
            raise ValueError("Expected packed tensors of shape (N, C, H, W)")

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]

        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        mask = self.masks[idx]
        if self.augment:
            img, mask = _augment_pair(img, mask)
        return img.float(), mask.float()


def _augment_pair(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return img.contiguous(), mask.contiguous()
