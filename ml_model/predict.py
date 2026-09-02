import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_model.infer import (
    extract_geometric_properties,
    load_checkpoint,
    load_norm_stats,
    overlay_rgb,
    predict_array,
)


def main():
    parser = argparse.ArgumentParser(description="SAR oil-spill inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="ml_model/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-res", type=float, default=10.0)
    parser.add_argument("--output-image", type=str, default="ml_model/detection_overlay.png")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.root)
    stats = load_norm_stats(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.model, device)
    if model is None:
        print("This CLI currently expects a .pth or TorchScript .pt checkpoint, not ONNX.")
        sys.exit(1)

    img = Image.open(args.image)
    gray = np.array(img.convert("L"), dtype=np.float32)
    pred = predict_array(model, gray, stats, device)
    mask = (pred > args.threshold).astype(np.uint8)
    slicks = extract_geometric_properties(mask * 255, pixel_resolution_m=args.pixel_res)

    output_data = {
        "image_file": str(args.image),
        "model_meta": {k: v for k, v in meta.items() if k != "state_dict"},
        "spill_detected": len(slicks) > 0,
        "slicks_count": len(slicks),
        "total_area_km2": round(sum(s["area_km2"] for s in slicks), 4),
        "slicks_characterization": [
            {k: v for k, v in s.items() if k != "polygon"} for s in slicks
        ],
    }
    print(json.dumps(output_data, indent=2))

    overlay = overlay_rgb(gray, mask, slicks)
    out_path = Path(args.output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(out_path))
    print("Saved overlay:", out_path)


if __name__ == "__main__":
    main()
