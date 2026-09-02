"""
SARSpillDataset
==============
Loads matched image-mask pairs from the SOS/PALSAR+Sentinel dataset.

Directory layout expected (same as the original download):
  <root>/Images/images/{train,val}/  -- SAR images  (256x256 RGB PNG)
  <root>/Masks/masks/{train,val}/    -- Binary masks (256x256 RGB PNG, values 0 or 255)

Usage
-----
    from data_pipeline.dataset import SARSpillDataset
    ds = SARSpillDataset(root='path/to/SIH', split='train')
    img_tensor, mask_tensor = ds[0]
    # img_tensor  : (3, 256, 256) float32  [0, 1]
    # mask_tensor : (1, 256, 256) float32  {0.0, 1.0}
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SARSpillDataset(Dataset):
    """
    PyTorch Dataset for SAR oil-spill segmentation.

    Parameters
    ----------
    root : str | Path
        Root directory of the project (contains 'Images/' and 'Masks/').
    split : str
        One of {'train', 'val', 'test'}.
        'test' is synthesised from a held-out portion of the original 'train'
        directory -- see prepare_splits.py for how those lists are made.
    file_list : list[str], optional
        Explicit list of filenames (e.g., from a pre-computed split file).
        If None, every PNG in the split directory is used.
    transform : callable, optional
        A function/transform applied to (image_pil, mask_pil) that returns
        (image_tensor, mask_tensor).  If None, the default pipeline is used
        (convert to tensor, normalise image, binarise mask).
    sensor_filter : str | None
        'palsar', 'sentinel', or None (both).  Filters files by filename prefix.
    """

    IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats -- reasonable starting point
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        sensor_filter: Optional[str] = None,
    ):
        super().__init__()
        self.root   = Path(root)
        self.split  = split
        self.transform = transform

        # Resolve the actual directory names on disk
        # Original dataset has train/val; 'test' comes from a pre-split file list
        disk_split = 'train' if split == 'test' else split

        self.img_dir  = self.root / 'Images' / 'images' / disk_split
        self.mask_dir = self.root / 'Masks'  / 'masks'  / disk_split

        if not self.img_dir.exists():
            raise FileNotFoundError("Image directory not found: {}".format(self.img_dir))
        if not self.mask_dir.exists():
            raise FileNotFoundError("Mask directory not found: {}".format(self.mask_dir))

        # Build file list
        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted(
                f for f in os.listdir(self.img_dir) if f.endswith('.png')
            )

        # Optional sensor filter
        if sensor_filter is not None:
            prefix = sensor_filter.lower()
            self.files = [f for f in self.files if f.startswith(prefix)]

        if len(self.files) == 0:
            raise RuntimeError(
                "No files found in {} (sensor_filter={!r})".format(
                    self.img_dir, sensor_filter
                )
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]

        img_path  = self.img_dir  / fname
        mask_path = self.mask_dir / fname

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')   # grayscale -> 0..255

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img, mask)
        else:
            img_tensor, mask_tensor = self._default_transform(img, mask)

        return img_tensor, mask_tensor

    def _default_transform(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts raw PIL images to tensors without any augmentation.
        Used for validation and test sets.
        """
        # Image -> (3, H, W) float32 in [0,1], then normalise
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = (img_arr - self.IMG_MEAN) / self.IMG_STD
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # CHW

        # Mask -> (1, H, W) binary float32  {0.0, 1.0}
        mask_arr    = np.array(mask, dtype=np.float32)
        mask_binary = (mask_arr > 127).astype(np.float32)        # handles 0/255
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # 1HW

        return img_tensor, mask_tensor

    def get_filename(self, idx: int) -> str:
        """Return the filename for sample at index idx (useful for debugging)."""
        return self.files[idx]

    def get_sensor(self, idx: int) -> str:
        """Return 'palsar' or 'sentinel' for sample at index idx."""
        fname = self.files[idx]
        return 'palsar' if fname.startswith('palsar') else 'sentinel'
