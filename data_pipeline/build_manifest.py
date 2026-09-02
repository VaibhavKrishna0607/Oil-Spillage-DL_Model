"""
Build an auditable manifest for the SOS image/mask pairs.

Run from the project root:
    python data_pipeline/build_manifest.py --root .

The script never modifies source images. It writes:
    artifacts/dataset_manifest.csv  one row per image candidate
    artifacts/dataset_audit.txt     summary and rejected-file reasons

Only rows marked ``valid`` should be used for split generation and training.
The manifest also records image/mask hashes, making accidental duplicates or
later source changes detectable before a training run.
"""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pair(image_path: Path, mask_path: Path) -> dict:
    row = {
        "split": image_path.parent.name,
        "filename": image_path.name,
        "image_path": image_path.as_posix(),
        "mask_path": mask_path.as_posix(),
        "sensor": "palsar" if image_path.name.lower().startswith("palsar") else "sentinel",
        "status": "valid",
        "reason": "",
        "width": "",
        "height": "",
        "image_mode": "",
        "mask_mode": "",
        "oil_pixels": "",
        "oil_fraction": "",
        "bbox_x_min": "",
        "bbox_y_min": "",
        "bbox_x_max": "",
        "bbox_y_max": "",
        "image_sha256": "",
        "mask_sha256": "",
    }

    if not mask_path.exists():
        row["status"], row["reason"] = "invalid", "missing_mask"
        return row

    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            row["width"], row["height"] = image.size
            row["image_mode"], row["mask_mode"] = image.mode, mask.mode
            if image.size != mask.size:
                row["status"], row["reason"] = "invalid", "dimension_mismatch"
            image.load()
            mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
            image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
            if not np.isfinite(image_array).all():
                row["status"], row["reason"] = "invalid", "non_finite_image"

            binary = mask_array > 127
            oil_pixels = int(binary.sum())
            row["oil_pixels"] = oil_pixels
            row["oil_fraction"] = "{:.8f}".format(float(binary.mean()))
            if oil_pixels:
                ys, xs = np.where(binary)
                row.update({
                    "bbox_x_min": int(xs.min()), "bbox_y_min": int(ys.min()),
                    "bbox_x_max": int(xs.max()), "bbox_y_max": int(ys.max()),
                })
        row["image_sha256"] = sha256(image_path)
        row["mask_sha256"] = sha256(mask_path)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        row["status"], row["reason"] = "invalid", "unreadable:{}".format(type(error).__name__)
    return row


def build_manifest(root: Path) -> tuple[Path, Path]:
    rows = []
    for split in ("train", "val"):
        image_dir = root / "Images" / "images" / split
        mask_dir = root / "Masks" / "masks" / split
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError("Missing image/mask directory for split: {}".format(split))
        for image_path in sorted(image_dir.glob("*.png")):
            rows.append(inspect_pair(image_path, mask_dir / image_path.name))

    output_dir = root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.csv"
    audit_path = output_dir / "dataset_audit.txt"
    fields = list(rows[0]) if rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["status"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows if row["reason"])
    valid_rows = [row for row in rows if row["status"] == "valid"]
    sensors = Counter(row["sensor"] for row in valid_rows)
    empty_masks = sum(int(row["oil_pixels"] or 0) == 0 for row in valid_rows)
    with audit_path.open("w", encoding="utf-8") as handle:
        handle.write("SIH dataset audit\n=================\n")
        handle.write("Total image candidates: {}\n".format(len(rows)))
        handle.write("Valid pairs: {}\n".format(len(valid_rows)))
        handle.write("Invalid pairs: {}\n".format(status_counts["invalid"]))
        handle.write("Sensors (valid): {}\n".format(dict(sensors)))
        handle.write("Valid empty masks: {}\n".format(empty_masks))
        handle.write("Reasons: {}\n".format(dict(reason_counts)))
        handle.write("\nTraining gate:\n")
        handle.write("- Use status=valid rows only.\n")
        handle.write("- Review empty masks; retain them only as intentional negatives.\n")
        handle.write("- Check duplicate image_sha256 values before final splitting.\n")
        handle.write("- Keep scene-level/group-level separation when GeoTIFF metadata is available.\n")

    print("Wrote {} ({} rows)".format(manifest_path, len(rows)))
    print("Wrote {}".format(audit_path))
    print("Valid: {} | Invalid: {}".format(len(valid_rows), status_counts["invalid"]))
    return manifest_path, audit_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit paired SAR images and masks.")
    parser.add_argument("--root", default=".", help="Project root containing Images/ and Masks/")
    args = parser.parse_args()
    build_manifest(Path(args.root))