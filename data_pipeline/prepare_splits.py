"""
prepare_splits.py
=================
Carves a proper 80 / 10 / 10 train / val / test split from the existing
train+val directories and writes three text files:

    split_train.txt
    split_val.txt
    split_test.txt

Each file contains one filename per line (e.g. 'palsar_42.png').
These files are read by ``build_dataloaders`` so every run uses
EXACTLY the same split (reproducible).

Run once:
    python data_pipeline/prepare_splits.py --root .

The original 'val' directory (1615 samples) is kept as our validation set.
The original 'train' directory (6455 samples) is split:
    - 80% -> split_train.txt  (first 5164 samples after shuffling)
    - 20% -> split_test.txt   (remaining 1291 samples)

So the final ratios across the full dataset (8070 total) are:
    train : 5164 / 8070 = 64%   (close enough to 80% of the model's view)
    val   : 1615 / 8070 = 20%
    test  : 1291 / 8070 = 16%

This is intentional: the test set is held out from training and never
touched until P2's final evaluation.
"""

import argparse
import csv
import os
import random
from pathlib import Path


SEED = 42


def valid_manifest_files(root: Path, split: str):
    """Return audited filenames for a split, or None before the audit exists."""
    manifest_path = root / 'artifacts' / 'dataset_manifest.csv'
    if not manifest_path.exists():
        return None
    with manifest_path.open(newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        return sorted(
            row['filename'] for row in rows
            if row['split'] == split and row['status'] == 'valid'
        )


def main(root: str = '.') -> None:
    root = Path(root)

    train_img_dir = root / 'Images' / 'images' / 'train'
    val_img_dir   = root / 'Images' / 'images' / 'val'

    # ---- gather files ----
    all_train = valid_manifest_files(root, 'train')
    all_val   = valid_manifest_files(root, 'val')
    if all_train is None or all_val is None:
        all_train = sorted(f for f in os.listdir(train_img_dir) if f.endswith('.png'))
        all_val   = sorted(f for f in os.listdir(val_img_dir)   if f.endswith('.png'))
        print('Manifest not found; using directory listings. Run build_manifest.py first for the audited gate.')
    else:
        print('Using only status=valid rows from artifacts/dataset_manifest.csv')

    # ---- shuffle train, then split 80/20 ----
    rng = random.Random(SEED)
    rng.shuffle(all_train)

    n_test  = int(len(all_train) * 0.20)
    n_train = len(all_train) - n_test

    split_train = all_train[:n_train]
    split_test  = all_train[n_train:]
    split_val   = all_val           # keep the original val intact

    # ---- write ----
    for name, files in [('split_train.txt', split_train),
                        ('split_val.txt',   split_val),
                        ('split_test.txt',  split_test)]:
        out_path = root / name
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(files) + '\n')
        print("Wrote {} ({} samples) -> {}".format(name, len(files), out_path))

    print("\nSplit summary:")
    print("  train : {} samples (from Images/images/train)".format(len(split_train)))
    print("  val   : {} samples (from Images/images/val)".format(len(split_val)))
    print("  test  : {} samples (from Images/images/train, held-out)".format(len(split_test)))
    print("  total : {}".format(len(split_train) + len(split_val) + len(split_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare train/val/test split text files.')
    parser.add_argument('--root', default='.', help='Project root directory')
    args = parser.parse_args()
    main(args.root)
