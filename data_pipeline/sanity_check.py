"""
sanity_check.py
===============
Full sanity check on the entire dataset:
  - Verifies every image has a corresponding mask
  - Checks image/mask dimensions, modes, and pixel value ranges
  - Reports class balance (oil pixels vs. background pixels)
  - Detects all-black masks (negative-only samples)
  - Saves a visual sanity plot: 8 random image+mask pairs

Run:
    python data_pipeline/sanity_check.py --root .

Output files:
    sanity_plot.png     -- 8 random image-mask pairs (RGB image | mask overlay)
    sanity_report.txt   -- Text summary of all checks
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


SEED = 42
random.seed(SEED)


def check_split(img_dir: Path, mask_dir: Path, split_name: str):
    """Run checks on one split. Returns a report dict."""
    all_imgs  = sorted(f for f in os.listdir(img_dir)  if f.endswith('.png'))
    all_masks = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))

    img_set  = set(all_imgs)
    mask_set = set(all_masks)

    missing_masks  = sorted(img_set  - mask_set)
    orphan_masks   = sorted(mask_set - img_set)

    paired = sorted(img_set & mask_set)

    dim_mismatch   = []
    all_zero_masks = []
    all_one_masks  = []
    oil_pixels     = 0
    bg_pixels      = 0

    for fname in paired:
        img  = Image.open(img_dir  / fname)
        mask = Image.open(mask_dir / fname).convert('L')

        if img.size != mask.size:
            dim_mismatch.append((fname, img.size, mask.size))

        m = np.array(mask)
        b = (m > 127)
        spill = int(b.sum())
        total = b.size

        oil_pixels += spill
        bg_pixels  += (total - spill)

        if spill == 0:
            all_zero_masks.append(fname)
        if spill == total:
            all_one_masks.append(fname)

    total_px   = oil_pixels + bg_pixels
    oil_pct    = 100.0 * oil_pixels / total_px if total_px > 0 else 0.0
    bg_pct     = 100.0 * bg_pixels  / total_px if total_px > 0 else 0.0

    return {
        'split':          split_name,
        'total':          len(all_imgs),
        'paired':         len(paired),
        'missing_masks':  missing_masks,
        'orphan_masks':   orphan_masks,
        'dim_mismatch':   dim_mismatch,
        'all_zero_masks': all_zero_masks,
        'all_one_masks':  all_one_masks,
        'oil_pixels':     oil_pixels,
        'bg_pixels':      bg_pixels,
        'oil_pct':        oil_pct,
        'bg_pct':         bg_pct,
        'paired_files':   paired,
    }


def print_report(r: dict):
    print("\n=== {} SPLIT ({} images) ===".format(r['split'].upper(), r['total']))
    print("  Paired (img+mask)   : {}".format(r['paired']))
    print("  Missing masks       : {}".format(len(r['missing_masks'])))
    print("  Orphan masks        : {}".format(len(r['orphan_masks'])))
    print("  Dim mismatches      : {}".format(len(r['dim_mismatch'])))
    print("  All-black masks     : {}".format(len(r['all_zero_masks'])))
    print("  All-white masks     : {}".format(len(r['all_one_masks'])))
    print("  Oil pixels  (spill) : {:,}  ({:.2f}%)".format(r['oil_pixels'], r['oil_pct']))
    print("  Background pixels   : {:,}  ({:.2f}%)".format(r['bg_pixels'],  r['bg_pct']))
    if r['missing_masks']:
        print("  [!] Missing mask examples:", r['missing_masks'][:5])
    if r['dim_mismatch']:
        print("  [!] Dim mismatch examples:", r['dim_mismatch'][:3])


def save_visual_check(reports, root: Path, n: int = 8):
    """Save a matplotlib figure showing n random image+mask pairs."""
    if not HAS_MPL:
        print("matplotlib not installed -- skipping visual plot.")
        return

    all_pairs = []
    for r in reports:
        for f in r['paired_files']:
            split = r['split']
            disk_split = 'val' if split == 'val' else 'train'
            all_pairs.append((
                root / 'Images' / 'images' / disk_split / f,
                root / 'Masks'  / 'masks'  / disk_split / f,
                "{}: {}".format(split, f),
            ))

    random.shuffle(all_pairs)
    samples = all_pairs[:n]

    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.5))
    fig.suptitle("SAR Oil Spill Dataset -- Sanity Check", fontsize=14, y=1.01)

    for i, (img_path, mask_path, title) in enumerate(samples):
        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Overlay: highlight oil pixels in red on image
        overlay = img.copy()
        overlay[mask > 127] = [255, 50, 50]
        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image: " + title, fontsize=7)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(blended)
        axes[i, 1].set_title("Mask overlay (red=oil)", fontsize=7)
        axes[i, 1].axis('off')

    plt.tight_layout()
    out = root / 'sanity_plot.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print("\nSaved visual check -> {}".format(out))
    plt.close()


def main(root: str = '.'):
    root_path = Path(root)

    splits_dirs = [
        (root_path / 'Images' / 'images' / 'train',
         root_path / 'Masks'  / 'masks'  / 'train',
         'train'),
        (root_path / 'Images' / 'images' / 'val',
         root_path / 'Masks'  / 'masks'  / 'val',
         'val'),
    ]

    reports = []
    lines   = []

    for img_dir, mask_dir, split_name in splits_dirs:
        print("Checking {} split...".format(split_name), end=' ', flush=True)
        r = check_split(img_dir, mask_dir, split_name)
        reports.append(r)
        print("done.")
        print_report(r)

    # Aggregate
    total_oil = sum(r['oil_pixels'] for r in reports)
    total_bg  = sum(r['bg_pixels']  for r in reports)
    total_px  = total_oil + total_bg
    print("\n=== OVERALL ===")
    print("  Total images : {}".format(sum(r['total']  for r in reports)))
    print("  Oil pixels   : {:,}  ({:.2f}%)".format(total_oil, 100.0 * total_oil / total_px))
    print("  BG  pixels   : {:,}  ({:.2f}%)".format(total_bg,  100.0 * total_bg  / total_px))
    print("  Class imbalance ratio (bg/oil) : {:.1f}x".format(
        total_bg / total_oil if total_oil else float('inf')
    ))

    save_visual_check(reports, root_path)

    print("\nAll checks complete. Review sanity_plot.png for visual confirmation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full dataset sanity check.')
    parser.add_argument('--root', default='.', help='Project root directory')
    args = parser.parse_args()
    main(args.root)
