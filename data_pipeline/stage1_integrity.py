"""
Stage 1: Integrity & Pairing Check
Run once at the start: python data_pipeline/stage1_integrity.py --root .
Output: artifacts/integrity_report.csv + artifacts/integrity_report.txt
"""

import argparse
import csv
import os
from pathlib import Path
from PIL import Image
import numpy as np

def check_integrity(root_dir):
    root = Path(root_dir)
    img_dir_train = root / 'Images' / 'images' / 'train'
    mask_dir_train = root / 'Masks' / 'masks' / 'train'
    img_dir_val = root / 'Images' / 'images' / 'val'
    mask_dir_val = root / 'Masks' / 'masks' / 'val'
    
    errors = []
    pairs = []
    
    for split, img_dir, mask_dir in [
        ('train', img_dir_train, mask_dir_train),
        ('val', img_dir_val, mask_dir_val),
    ]:
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = img_dir / fname
            mask_path = mask_dir / fname
            
            status = 'valid'
            reason = ''
            
            # Check mask exists
            if not mask_path.exists():
                status, reason = 'invalid', 'missing_mask'
                errors.append((fname, reason))
                continue
            
            # Try to load
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        status, reason = 'invalid', f'size_mismatch: {img.size} vs {mask.size}'
                    if img.mode not in ['L', 'RGB']:
                        status, reason = 'invalid', f'img_mode: {img.mode}'
                    if mask.mode not in ['L', 'RGB', 'P']:
                        status, reason = 'invalid', f'mask_mode: {mask.mode}'
            except Exception as e:
                status, reason = 'invalid', str(e)[:50]
                errors.append((fname, reason))
                continue
            
            pairs.append({
                'split': split,
                'filename': fname,
                'status': status,
                'reason': reason,
            })
    
    # Write reports
    artifacts_dir = root / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    with open(artifacts_dir / 'integrity_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'filename', 'status', 'reason'])
        writer.writeheader()
        writer.writerows(pairs)
    
    with open(artifacts_dir / 'integrity_report.txt', 'w') as f:
        valid_count = sum(1 for p in pairs if p['status'] == 'valid')
        invalid_count = len(pairs) - valid_count
        f.write(f"Integrity Check Report\n")
        f.write(f"======================\n")
        f.write(f"Total pairs: {len(pairs)}\n")
        f.write(f"Valid: {valid_count}\n")
        f.write(f"Invalid: {invalid_count}\n\n")
        if errors:
            f.write("Errors:\n")
            for fname, reason in errors:
                f.write(f"  {fname}: {reason}\n")
    
    print(f"✓ Integrity check complete: {valid_count} valid, {invalid_count} invalid")
    print(f"  Report saved to artifacts/integrity_report.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Project root directory')
    args = parser.parse_args()
    check_integrity(args.root)