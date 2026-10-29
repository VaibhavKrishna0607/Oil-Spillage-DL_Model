"""
Oil Spill Detection API - Ready for Frontend Integration
Connects trained models to frontend via REST API
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml_model.model import UNet
from data_pipeline.krestenitis_dataset import load_krest_stats


class OilSpillDetector:
    """
    Main inference class for oil spill detection
    Loads trained model and runs predictions on SAR images
    """
    
    def __init__(self, model_path: str, device: str = "cuda", base: int = 32):
        """
        Initialize detector with trained model
        
        Args:
            model_path: Path to trained .pth model file
            device: "cuda" or "cpu"
            base: Model base channels (32 or 64)
        """
        self.device = torch.device(device)
        self.model = UNet(n_channels=1, n_classes=1, bilinear=False, base=base, use_attention=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.norm_stats = load_krest_stats(ROOT)
        print(f"✓ Model loaded from {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Norm stats: mean={self.norm_stats['mean']}, std={self.norm_stats['std']}")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize SAR image using radiometric statistics"""
        # Clip to 2% and 98% percentiles
        image = np.clip(
            image,
            self.norm_stats["clip_2"],
            self.norm_stats["clip_98"]
        )
        
        # Standardize
        image = (image - self.norm_stats["mean"]) / self.norm_stats["std"]
        return image.astype(np.float32)
    
    def extract_polygons(self, mask: np.ndarray, threshold: float = 0.5) -> List[Dict]:
        """
        Extract oil spill polygons from prediction mask
        
        Returns list of polygons with area, perimeter, centroid
        """
        import cv2
        
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Skip tiny contours
                continue
            
            # Get polygon points
            points = contour.squeeze(1).tolist()
            if len(points) < 3:
                continue
            
            # Calculate properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            moment = cv2.moments(contour)
            
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
            else:
                cx, cy = 0, 0
            
            # Eccentricity
            if len(points) > 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    eccentricity = 0
            else:
                eccentricity = 0
            
            polygons.append({
                "points": points,
                "area_pixels": float(area),
                "perimeter_pixels": float(perimeter),
                "centroid": [int(cx), int(cy)],
                "eccentricity": float(eccentricity),
                "confidence": float(np.mean(mask[binary_mask > 0])) if np.any(binary_mask > 0) else 0.0
            })
        
        return polygons
    
    @torch.no_grad()
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Predict oil spills in a SAR image
        
        Args:
            image_path: Path to SAR image (PNG/JPG)
            threshold: Detection threshold (0.0-1.0)
        
        Returns:
            Dictionary with spill detection results
        """
        # Load image
        image = Image.open(image_path).convert("L")
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize
        image_norm = self.normalize_image(image_array)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)  # [1, 1, H, W]
        
        # Convert to numpy
        prediction = output[0, 0].cpu().numpy()
        
        # Extract polygons
        polygons = self.extract_polygons(prediction, threshold)
        
        # Calculate overall statistics
        detected_area = np.sum(prediction > threshold)
        total_area = prediction.size
        detection_percentage = (detected_area / total_area) * 100
        
        return {
            "spill_detected": detected_area > 0,
            "detection_percentage": float(detection_percentage),
            "confidence": float(np.mean(prediction)),
            "max_confidence": float(np.max(prediction)),
            "threshold_used": threshold,
            "polygons_count": len(polygons),
            "polygons": polygons,
            "mask_shape": prediction.shape,
            "total_area_pixels": int(total_area),
            "detected_area_pixels": int(detected_area)
        }
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str], threshold: float = 0.5) -> List[Dict]:
        """Predict multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, threshold)
            result["image_path"] = image_path
            results.append(result)
        return results


class DualTrackDetector:
    """
    Dual-track detection: SOS validation + Krestenitis confirmation
    Reduces false positives by using two models
    """
    
    def __init__(
        self,
        krestenitis_model_path: str,
        sos_model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize dual-track detector
        
        Args:
            krestenitis_model_path: Path to Krestenitis model (main detector)
            sos_model_path: Optional path to SOS model (validator)
            device: "cuda" or "cpu"
        """
        self.krestenitis = OilSpillDetector(krestenitis_model_path, device=device, base=32)
        self.sos = None
        if sos_model_path:
            self.sos = OilSpillDetector(sos_model_path, device=device, base=32)
            print("✓ Dual-track mode: SOS validator enabled")
        else:
            print("⚠ Single-track mode: Using Krestenitis only")
    
    def predict(self, image_path: str, sos_threshold: float = 0.75, 
                krestenitis_threshold: float = 0.5) -> Dict:
        """
        Dual-track prediction
        
        1. If SOS confidence < threshold → likely not oil, skip expensive check
        2. If SOS confidence > threshold → run Krestenitis confirmation
        3. Return averaged confidence
        """
        # Track B: Krestenitis (main detector)
        krestenitis_result = self.krestenitis.predict(image_path, krestenitis_threshold)
        
        if self.sos is None:
            # Single track
            return {
                **krestenitis_result,
                "track": "single_krestenitis",
                "final_confidence": krestenitis_result["confidence"]
            }
        
        # Track A: SOS (validator) - only run if Krestenitis looks positive
        if krestenitis_result["confidence"] > 0.4:
            sos_result = self.sos.predict(image_path, sos_threshold)
            
            # Combine scores
            final_confidence = (krestenitis_result["confidence"] + sos_result["confidence"]) / 2
            
            return {
                "spill_detected": final_confidence > 0.5,
                "track": "dual_krestenitis_sos",
                "krestenitis_confidence": krestenitis_result["confidence"],
                "sos_confidence": sos_result["confidence"],
                "final_confidence": final_confidence,
                "polygons": krestenitis_result["polygons"],  # Use Krestenitis polygons
                "polygons_count": krestenitis_result["polygons_count"],
                "krestenitis_result": krestenitis_result,
                "sos_result": sos_result
            }
        else:
            # Krestenitis not confident, don't bother with SOS
            return {
                **krestenitis_result,
                "track": "krestenitis_only",
                "final_confidence": krestenitis_result["confidence"]
            }


def create_flask_app():
    """Create Flask app for API deployment"""
    from flask import Flask, request, jsonify, send_file
    from werkzeug.utils import secure_filename
    import tempfile
    
    app = Flask(__name__)
    
    # Initialize detector
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    krestenitis_model = str(checkpoint_dir / "dataset2_optimized.pth")
    
    if not Path(krestenitis_model).exists():
        print(f"⚠ Model not found: {krestenitis_model}")
        print(f"  Using best_model.pth instead")
        krestenitis_model = str(checkpoint_dir / "best_model.pth")
    
    detector = DualTrackDetector(krestenitis_model)
    
    @app.route("/api/detect", methods=["POST"])
    def detect_oil_spill():
        """
        Endpoint: POST /api/detect
        Body: multipart/form-data with 'image' file
        Returns: JSON with detection results
        """
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        try:
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                file.save(tmp.name)
                result = detector.predict(tmp.name)
            
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "ok",
            "model": "OilSpillDetector",
            "version": "1.0"
        }), 200
    
    return app


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python api_inference.py <image_path> [threshold]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    model_path = str(checkpoint_dir / "dataset2_optimized.pth")
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        model_path = str(checkpoint_dir / "best_model.pth")
    
    print(f"Loading model: {model_path}")
    detector = OilSpillDetector(model_path)
    
    print(f"Predicting: {image_path}")
    result = detector.predict(image_path, threshold)
    
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)
