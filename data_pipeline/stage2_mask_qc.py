"""
Stage 2: Mask QC + Compute Normalization Stats (Training Split Only!)
Run: python data_pipeline/stage2_mask_qc.py --root .
Output: artifacts/mask_qc_report.csv + artifacts/norm_stats.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

def qc_and_normalize(root_dir):
    root = Path(root_dir)
    
    # Read valid filenames from integrity report
    integrity_csv = root / 'artifacts' / 'integrity_report.csv'
    if not integrity_csv.exists():
        print("✗ Run stage1_integrity.py first!")
        return
    
    valid_files = {'train': [], 'val': []}
    
    with open(integrity_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'valid':
                valid_files[row['split']].append(row['filename'])
    
    print(f"Processing {sum(len(v) for v in valid_files.values())} valid files...")
    
    qc_rows = []
    
    # --- QC Pass: Check mask quality ---
    for split in ['train', 'val']:
        mask_dir = root / 'Masks' / 'masks' / split
        
        for fname in valid_files[split]:
            mask_path = mask_dir / fname
            mask_pil = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask_pil, dtype=np.uint8)
            
            binary = mask_arr > 127
            oil_pixels = int(binary.sum())
            total_pixels = mask_arr.size
            oil_fraction = oil_pixels / total_pixels
            
            # Flag suspicious masks
            flag = ''
            if oil_fraction < 0.001:
                flag = 'nearly_empty'
            elif oil_fraction > 0.70:
                flag = 'mostly_oil_suspect'
            
            qc_rows.append({
                'split': split,
                'filename': fname,
                'oil_pixels': oil_pixels,
                'oil_fraction': f'{oil_fraction:.6f}',
                'flag': flag,
            })
    
    # Write QC report
    with open(root / 'artifacts' / 'mask_qc_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'filename', 'oil_pixels', 'oil_fraction', 'flag'])
        writer.writeheader()
        writer.writerows(qc_rows)
    
    flagged = sum(1 for r in qc_rows if r['flag'])
    print(f"  QC complete: {len(qc_rows)} masks, {flagged} flagged")
    
    # --- Compute normalization stats from TRAINING SPLIT ONLY ---
    print("Computing normalization stats from training split only...")
    img_dir_train = root / 'Images' / 'images' / 'train'
    
    img_values = []
    for fname in valid_files['train']:
        img_path = img_dir_train / fname
        img_pil = Image.open(img_path).convert('L')
        img_arr = np.array(img_pil, dtype=np.float32)
        img_values.append(img_arr.flatten())
    
    img_values = np.concatenate(img_values)
    
    # Clip at 2nd and 98th percentile (SAR outlier handling)
    clip_2 = float(np.percentile(img_values, 2))
    clip_98 = float(np.percentile(img_values, 98))
    
    img_clipped = np.clip(img_values, clip_2, clip_98)
    mean_val = float(img_clipped.mean())
    std_val = float(img_clipped.std())
    
    norm_stats = {
        'clip_2': clip_2,
        'clip_98': clip_98,
        'mean': mean_val,
        'std': std_val,
    }
    
    with open(root / 'artifacts' / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"  Normalization stats: mean={mean_val:.2f}, std={std_val:.2f}")
    print(f"  Clipping range: [{clip_2:.1f}, {clip_98:.1f}]")
    print(f"  ✓ Saved to artifacts/norm_stats.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Project root')
    args = parser.parse_args()
    qc_and_normalize(args.root)