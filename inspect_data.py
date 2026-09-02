"""
Quick sanity-check script for the SIH SAR dataset.
Run: python inspect_data.py
"""
import os
from PIL import Image
import numpy as np
from collections import Counter
from glob import glob

base_img  = 'Images/images'
base_mask = 'Masks/masks'
splits    = ['train', 'val']

results = {}

for split in splits:
    img_dir  = os.path.join(base_img,  split)
    mask_dir = os.path.join(base_mask, split)

    files    = sorted(f for f in glob(os.path.join(img_dir, '*.png')))

    img_sizes    = Counter()
    mask_sizes   = Counter()
    mask_uniq    = Counter()
    img_modes    = Counter()
    mask_modes   = Counter()
    corrupt      = []
    dim_mismatch = []
    tiny_mask    = []

    sample_size = min(200, len(files))
    sampled     = files[:sample_size]

    for fname in sampled:
        try:
            img  = Image.open(os.path.join(img_dir,  fname))
            mask = Image.open(os.path.join(mask_dir, fname))

            img_sizes[img.size]   += 1
            mask_sizes[mask.size] += 1
            img_modes[img.mode]   += 1
            mask_modes[mask.mode] += 1

            if img.size != mask.size:
                dim_mismatch.append(fname)

            m_arr = np.array(mask)
            uniq  = tuple(sorted(np.unique(m_arr).tolist()))
            mask_uniq[uniq] += 1

            if mask.mode != 'L':
                mask_gray = mask.convert('L')
            else:
                mask_gray = mask
            m_gray     = np.array(mask_gray)
            spill_ratio = (m_gray > 0).mean()
            if spill_ratio < 0.01:
                tiny_mask.append((fname, round(float(spill_ratio), 4)))

        except Exception as e:
            corrupt.append((fname, str(e)))

    results[split] = {
        'total':         len(files),
        'sampled':       sample_size,
        'img_sizes':     dict(img_sizes.most_common(5)),
        'mask_sizes':    dict(mask_sizes.most_common(5)),
        'img_modes':     dict(img_modes),
        'mask_modes':    dict(mask_modes),
        'dim_mismatch':  dim_mismatch,
        'corrupt':       corrupt,
        'mask_uniq':     dict(mask_uniq.most_common(10)),
        'tiny_count':    len(tiny_mask),
        'tiny_examples': tiny_mask[:5],
    }

for split, r in results.items():
    print("=== {} (sampled {}/{}) ===".format(split.upper(), r['sampled'], r['total']))
    print("  Image sizes (top5) :", r['img_sizes'])
    print("  Mask  sizes (top5) :", r['mask_sizes'])
    print("  Image modes        :", r['img_modes'])
    print("  Mask  modes        :", r['mask_modes'])
    print("  Dim mismatches     :", len(r['dim_mismatch']), r['dim_mismatch'][:3])
    print("  Corrupt files      :", len(r['corrupt']),      r['corrupt'][:3])
    print("  Mask unique combos :", r['mask_uniq'])
    print("  Empty masks (<1%)  :", r['tiny_count'], "examples:", r['tiny_examples'])
    print()

