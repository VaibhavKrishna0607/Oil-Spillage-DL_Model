"""
Stage 3: Group-Aware Split + Geometric Properties + Tensor Packing
Run: python data_pipeline/stage3_split_geometry.py --root .
Output: split_*.txt + artifacts/geometry.csv + dataset_ready/
"""

import argparse
import csv
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from skimage import measure

def extract_scene_id(filename):
    """Extract scene ID from filename (sensor + first 2 digits)."""
    try:
        # Format: palsar_00123.png or sentinel_45678.png
        parts = filename.replace('.png', '').replace('.jpg', '').split('_')
        if len(parts) >= 2:
            sensor = parts[0][:3]  # 'pal' or 'sen'
            idx = parts[1][:2]  # First 2 digits
            return f"{sensor}_{idx}"
    except:
        pass
    return filename[:10]  # Fallback

def run_pipeline(root_dir):
    root = Path(root_dir)
    
    # Load integrity report
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
    
    # Load normalization stats
    norm_stats_path = root / 'artifacts' / 'norm_stats.json'
    if not norm_stats_path.exists():
        print("✗ Run stage2_mask_qc.py first!")
        return
    
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    # --- STEP 1: Group-Aware Split ---
    print("Creating group-aware split...")
    train_filenames = valid_files['train']
    scene_ids = np.array([extract_scene_id(f) for f in train_filenames])
    
    gss = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    train_idx, test_idx = next(gss.split(train_filenames, groups=scene_ids))
    
    split_train = [train_filenames[i] for i in train_idx]
    split_test = [train_filenames[i] for i in test_idx]
    split_val = valid_files['val']
    
    # Write split files
    for split_name, split_list in [('train', split_train), ('test', split_test), ('val', split_val)]:
        with open(root / f'split_{split_name}.txt', 'w') as f:
            f.write('\n'.join(split_list))
    
    print(f"  Train: {len(split_train)}, Val: {len(split_val)}, Test: {len(split_test)}")
    
    # --- STEP 2: Extract Geometric Properties ---
    print("Extracting geometric properties...")
    geometry_rows = []
    
    for split_name, split_list in [('train', split_train), ('val', split_val), ('test', split_test)]:
        disk_split = 'train' if split_name == 'test' else split_name
        mask_dir = root / 'Masks' / 'masks' / disk_split
        
        for fname in split_list:
            mask_path = mask_dir / fname
            mask_pil = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask_pil, dtype=np.uint8) > 127
            
            labeled = measure.label(mask_arr > 0)
            props = measure.regionprops(labeled)
            
            if props:
                largest = max(props, key=lambda p: p.area)
                geometry_rows.append({
                    'filename': fname,
                    'split': split_name,
                    'area_px': int(largest.area),
                    'perimeter_px': float(largest.perimeter),
                    'eccentricity': float(largest.eccentricity),
                    'centroid_row': float(largest.centroid[0]),
                    'centroid_col': float(largest.centroid[1]),
                    'bbox_x_min': int(largest.bbox[1]),
                    'bbox_y_min': int(largest.bbox[0]),
                    'bbox_x_max': int(largest.bbox[3]),
                    'bbox_y_max': int(largest.bbox[2]),
                })
    
    with open(root / 'artifacts' / 'geometry.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=geometry_rows[0].keys())
        writer.writeheader()
        writer.writerows(geometry_rows)
    
    print(f"  Extracted properties for {len(geometry_rows)} masks")
    
    # --- STEP 3: Pack Tensors ---
    print("Packing tensors...")
    dataset_dir = root / 'dataset_ready'
    dataset_dir.mkdir(exist_ok=True)
    
    for split_name, split_list in [('train', split_train), ('val', split_val), ('test', split_test)]:
        disk_split = 'train' if split_name == 'test' else split_name
        img_dir = root / 'Images' / 'images' / disk_split
        mask_dir = root / 'Masks' / 'masks' / disk_split
        
        images, masks = [], []
        
        for fname in split_list:
            img_pil = Image.open(img_dir / fname).convert('L')
            mask_pil = Image.open(mask_dir / fname).convert('L')
            
            img_arr = np.array(img_pil, dtype=np.float32)
            mask_arr = (np.array(mask_pil, dtype=np.uint8) > 127).astype(np.float32)
            
            # Normalize image (using stats from training only)
            img_arr = np.clip(img_arr, norm_stats['clip_2'], norm_stats['clip_98'])
            img_arr = (img_arr - norm_stats['mean']) / (norm_stats['std'] + 1e-7)
            
            images.append(torch.from_numpy(img_arr[None, ...]))  # (1, H, W)
            masks.append(torch.from_numpy(mask_arr[None, ...]))
        
        images_tensor = torch.stack(images)
        masks_tensor = torch.stack(masks)
        
        torch.save(images_tensor, dataset_dir / f'{split_name}_images.pt')
        torch.save(masks_tensor, dataset_dir / f'{split_name}_masks.pt')
        print(f"  {split_name}: {images_tensor.shape}")
    
    # Save metadata
    json.dump(norm_stats, open(dataset_dir / 'norm_stats.json', 'w'), indent=2)
    
    print(f"\n✓ Pipeline complete! Dataset ready in dataset_ready/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Project root')
    args = parser.parse_args()
    run_pipeline(args.root)