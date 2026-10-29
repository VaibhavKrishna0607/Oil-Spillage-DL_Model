"""
Krestenitis Real-Life SAR Dataset Pipeline
===========================================
Tiles large SAR scenes (1250x650) into 256x256 patches.
Converts 5-class annotations:
  - Class 1 (Oil Spill) -> Positive Target (1.0)
  - Class 2 (Look-Alike) -> Hard Negative Target (0.0)
  - Class 0 (Sea), Class 3 (Ship), Class 4 (Land) -> Negative Target (0.0)
Applies domain-specific radiometric normalization.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_KREST_STATS = {
    "clip_2": 15.0,
    "clip_98": 254.0,
    "mean": 136.02,
    "std": 59.69,
}


def load_krest_stats(root: Path) -> dict:
    for candidate in [
        root / "artifacts" / "norm_stats_krestenitis.json",
        root / "dataset_ready" / "norm_stats_krestenitis.json",
    ]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    return DEFAULT_KREST_STATS


class KrestenitisTileDataset(Dataset):
    """
    Tiled dataset from Krestenitis real-life 1250x650 scenes.
    Extracts 256x256 windows with stride.
    Allows balanced sampling of oil-containing and look-alike tiles.
    """
    def __init__(
        self,
        root_dir: str = "Oil Spill Detection Dataset",
        split: str = "train",
        tile_size: int = 256,
        stride: int = 128,
        augment: bool = False,
        oil_only: bool = False,
        max_samples: Optional[int] = None,
        norm_stats: Optional[dict] = None,
        sample_seed: int = 42,
    ):
        self.root = Path(root_dir) / split
        self.tile_size = tile_size
        self.stride = stride
        self.augment = augment and split == "train"
        self.norm_stats = norm_stats or DEFAULT_KREST_STATS

        img_dir = self.root / "images"
        label_dir = self.root / "labels_1D"

        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")

        image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        self.samples = []

        print(f"[KrestenitisTileDataset] Scanning {len(image_files)} scenes for split '{split}'...")
        
        # Pre-index tiles
        for img_idx, img_path in enumerate(image_files):
            label_path = label_dir / (img_path.stem + ".png")
            if not label_path.exists():
                continue

            # We know Krestenitis scenes are 1250x650
            # Width=1250, Height=650
            h, w = 650, 1250
            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    self.samples.append({
                        "img_path": str(img_path),
                        "label_path": str(label_path),
                        "y": y,
                        "x": x,
                    })

        if oil_only:
            # Quick filter for tiles that actually contain oil or look-alikes
            print(f"[KrestenitisTileDataset] Initial tiles: {len(self.samples)}. Filtering for oil/look-alike...")
            filtered = []
            for s in self.samples:
                # Open small crop of label
                lab = Image.open(s["label_path"])
                crop = lab.crop((s["x"], s["y"], s["x"] + tile_size, s["y"] + tile_size))
                arr = np.array(crop)
                has_oil = (arr == 1).any()
                has_lookalike = (arr == 2).any()
                if has_oil or has_lookalike:
                    filtered.append(s)
            self.samples = filtered
            print(f"[KrestenitisTileDataset] Filtered to {len(self.samples)} informative tiles.")

        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(sample_seed)
            self.samples = rng.sample(self.samples, max_samples)

        print(f"[KrestenitisTileDataset] Ready with {len(self.samples)} tiles.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("L")
        label = Image.open(sample["label_path"])

        x, y = sample["x"], sample["y"]
        img_crop = img.crop((x, y, x + self.tile_size, y + self.tile_size))
        label_crop = label.crop((x, y, x + self.tile_size, y + self.tile_size))

        img_arr = np.array(img_crop, dtype=np.float32)
        label_arr = np.array(label_crop)

        # Multi-class to binary:
        # Class 1 = Oil spill (1.0)
        # Class 0 (Sea), Class 2 (Look-alike), Class 3 (Ship), Class 4 (Land) = 0.0
        mask_arr = (label_arr == 1).astype(np.float32)

        # Radiometric normalization
        clipped = np.clip(img_arr, self.norm_stats["clip_2"], self.norm_stats["clip_98"])
        normed = (clipped - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-7)

        img_t = torch.from_numpy(normed).unsqueeze(0).float()   # (1, H, W)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float() # (1, H, W)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img_t = torch.flip(img_t, dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1).item() < 0.5:
                img_t = torch.flip(img_t, dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img_t = torch.rot90(img_t, k, dims=[-2, -1])
                mask_t = torch.rot90(mask_t, k, dims=[-2, -1])
            if torch.rand(1).item() < 0.5:
                gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.20
                bias = (torch.rand(1).item() - 0.5) * 0.20
                img_t = img_t * gain + bias
            if torch.rand(1).item() < 0.25:
                img_t = img_t + torch.randn_like(img_t) * 0.05

        return img_t.contiguous(), mask_t.contiguous()


if __name__ == "__main__":
    # Test dataset
    ds = KrestenitisTileDataset(
        root_dir="Oil Spill Detection Dataset",
        split="test",
        tile_size=256,
        stride=256,
        max_samples=10,
    )
    img_t, mask_t = ds[0]
    print("Tile image shape:", img_t.shape, "Mask shape:", mask_t.shape)
    print("Oil pixels in tile:", (mask_t > 0).sum().item())
    print("KrestenitisTileDataset test passed!")
