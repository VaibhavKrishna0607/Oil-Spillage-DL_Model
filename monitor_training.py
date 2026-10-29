#!/usr/bin/env python
"""
Monitor training progress in real-time
Watches the CSV output and shows progress
"""

import csv
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def monitor_training():
    """Watch training progress and show real-time stats"""
    
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    history_file = checkpoint_dir / "dataset2_history_optimized.csv"
    
    if not history_file.exists():
        print(f"⏳ Waiting for training to start...")
        print(f"   Watching: {history_file}")
        
        # Wait for file to appear
        while not history_file.exists():
            time.sleep(5)
        print(f"✅ Training started! Monitoring progress...\n")
        time.sleep(2)
    
    last_row_count = 0
    best_dice = 0.0
    best_epoch = 0
    
    print("=" * 100)
    print("TRAINING PROGRESS MONITOR - Krestenitis Oil Spill Detection")
    print("=" * 100)
    print()
    
    while True:
        try:
            with open(history_file, 'r') as f:
                reader = list(csv.DictReader(f))
            
            if len(reader) > last_row_count:
                # New epoch completed
                last_row = reader[-1]
                epoch = int(last_row['epoch'])
                train_loss = float(last_row['train_loss'])
                train_dice = float(last_row['train_dice'])
                val_loss = float(last_row['val_loss'])
                val_dice = float(last_row['val_dice'])
                threshold = float(last_row['val_threshold'])
                lr = float(last_row['learning_rate'])
                
                # Check if new best
                is_best = val_dice > best_dice
                if is_best:
                    best_dice = val_dice
                    best_epoch = epoch
                
                # Print progress
                star = " ★ NEW BEST!" if is_best else ""
                print(f"Epoch {epoch:2d}/20  │  "
                      f"Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}  │  "
                      f"Val: Loss={val_loss:.4f}, Dice={val_dice:.4f}  │  "
                      f"LR={lr:.2e}{star}")
                
                # Progress bar
                progress = (epoch / 20) * 100
                bar_length = 50
                filled = int(bar_length * epoch / 20)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"          │ [{bar}] {progress:.0f}%")
                print()
                
                last_row_count = len(reader)
                
                # Check if training might be done
                if epoch >= 20:
                    print("=" * 100)
                    print("✅ TRAINING COMPLETED!")
                    print("=" * 100)
                    print(f"Best Epoch: {best_epoch}")
                    print(f"Best Dice: {best_dice:.4f}")
                    print(f"Improvement from baseline (0.1573): {(best_dice - 0.1573):.4f} ({((best_dice - 0.1573) / 0.1573 * 100):.1f}%)")
                    print()
                    print("Model saved to:")
                    print(f"  {checkpoint_dir / 'dataset2_optimized.pth'}")
                    print()
                    print("Next steps:")
                    print("  1. python api_inference.py --model dataset2_optimized.pth")
                    print("  2. python -m flask --app api_inference:create_flask_app run")
                    print("  3. Connect to frontend!")
                    break
            
            time.sleep(10)  # Check every 10 seconds
        
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


def show_summary():
    """Show final training summary"""
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    meta_file = checkpoint_dir / "dataset2_meta_optimized.json"
    
    if meta_file.exists():
        import json
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        print("\n" + "=" * 100)
        print("FINAL TRAINING SUMMARY")
        print("=" * 100)
        print(f"Baseline Dice:        {meta.get('baseline_dice', 'N/A')}")
        print(f"Best Dice:            {meta.get('best_dice', 'N/A'):.4f}")
        print(f"Improvement:          {meta.get('improvement', 'N/A'):.4f}")
        print(f"Best Threshold:       {meta.get('best_threshold', 'N/A')}")
        print(f"Epochs Trained:       {meta.get('epochs_trained', 'N/A')}")
        print(f"Training Time:        {meta.get('elapsed_minutes', 'N/A')} minutes")
        print(f"Device:               {meta.get('device', 'N/A')}")
        print("=" * 100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        show_summary()
    else:
        try:
            monitor_training()
            show_summary()
        except KeyboardInterrupt:
            print("\n\n⏸ Monitoring stopped by user")
            show_summary()
