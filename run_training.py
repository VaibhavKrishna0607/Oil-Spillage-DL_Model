#!/usr/bin/env python
"""
Standalone training launcher - run this directly without needing terminal integration
Usage: python run_training.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    script = Path(__file__).parent / "ml_model" / "finetune_optimized.py"
    
    print("=" * 80)
    print("🚀 Starting Optimized SAR Oil Spill Model Training")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Epochs: 20")
    print("  - Batch Size: 16")
    print("  - Learning Rate: 5e-5")
    print("  - Model Base: 64 (increased capacity)")
    print("  - Loss Function: EnhancedComboLoss (Focal Tversky + BCE + Dice)")
    print("  - Mixed Precision: Enabled (if GPU available)")
    print()
    print("Expected Results:")
    print("  - Epoch 5: ~0.65-0.70 Dice")
    print("  - Epoch 10: ~0.75-0.78 Dice")
    print("  - Epoch 15: ~0.80-0.85 Dice")
    print("  - Epoch 20: ~0.82-0.88 Dice")
    print()
    print("=" * 80)
    print()
    
    cmd = [
        sys.executable,
        str(script),
        "--epochs", "20",
        "--batch-size", "16",
        "--lr", "5e-5",
        "--model-base", "64",
        "--use-mixed-precision"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
