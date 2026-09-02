"""
dataloaders.py
==============
Builds and returns PyTorch DataLoader objects for all three splits.

Usage
-----
    from data_pipeline.dataloaders import build_dataloaders

    loaders = build_dataloaders(root='.', batch_size=16, num_workers=4)
    train_loader = loaders['train']
    val_loader   = loaders['val']
    test_loader  = loaders['test']

    for images, masks in train_loader:
        # images : (B, 3, 256, 256) float32
        # masks  : (B, 1, 256, 256) float32  binary {0,1}
        ...

Notes
-----
- ``prepare_splits.py`` must be run once before calling this, to generate
  split_train.txt / split_val.txt / split_test.txt.
- If those files don't exist, all files in the directories are used
  (train dir -> train, val dir -> val, and no test set).
- ``pin_memory=True`` is set when a GPU is available.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from data_pipeline.dataset import SARSpillDataset
from data_pipeline.augmentations import get_train_transform, get_val_transform


def _load_split_file(path: Path):
    """Return list of filenames from a split text file, or None if file missing."""
    if not path.exists():
        return None
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def build_dataloaders(
    root: str = '.',
    batch_size: int = 16,
    num_workers: int = 4,
    sensor_filter: Optional[str] = None,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for train, val, and test splits.

    Parameters
    ----------
    root : str
        Project root (parent of Images/ and Masks/).
    batch_size : int
        Batch size for all loaders (train shuffles, val/test do not).
    num_workers : int
        Number of worker processes for data loading.
    sensor_filter : str | None
        Optionally restrict to 'palsar' or 'sentinel'.
    pin_memory : bool | None
        If None, auto-detected based on CUDA availability.

    Returns
    -------
    dict with keys 'train', 'val', 'test'  (test may be absent if no split file).
    """
    root_path = Path(root)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_files = _load_split_file(root_path / 'split_train.txt')
    val_files   = _load_split_file(root_path / 'split_val.txt')
    test_files  = _load_split_file(root_path / 'split_test.txt')

    # ---- datasets ----
    train_ds = SARSpillDataset(
        root=root,
        split='train',
        file_list=train_files,
        transform=get_train_transform(),
        sensor_filter=sensor_filter,
    )

    val_ds = SARSpillDataset(
        root=root,
        split='val',
        file_list=val_files,
        transform=get_val_transform(),
        sensor_filter=sensor_filter,
    )

    # ---- loaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,          # keeps batch size stable
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loaders = {'train': train_loader, 'val': val_loader}

    if test_files is not None:
        test_ds = SARSpillDataset(
            root=root,
            split='test',
            file_list=test_files,
            transform=get_val_transform(),
            sensor_filter=sensor_filter,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders['test'] = test_loader

    return loaders
