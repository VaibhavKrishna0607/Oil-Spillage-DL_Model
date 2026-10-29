@echo off
REM Standalone training launcher for SAR Oil Spill Model
REM Run this file directly by double-clicking or from command prompt

cd /d %~dp0

echo.
echo ================================================================================
echo Starting Optimized SAR Oil Spill Model Training
echo ================================================================================
echo.
echo Configuration:
echo   - Epochs: 20
echo   - Batch Size: 16
echo   - Learning Rate: 5e-5
echo   - Model Base: 64 (increased capacity)
echo   - Loss Function: EnhancedComboLoss
echo.
echo Expected Duration: 15-30 minutes
echo Expected Result: 0.80-0.88 Dice (from baseline 0.5690)
echo.
echo ================================================================================
echo.

python ml_model\finetune_optimized.py --epochs 20 --batch-size 16 --lr 5e-5 --model-base 64 --use-mixed-precision

if %errorlevel% neq 0 (
    echo.
    echo ❌ Training failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
) else (
    echo.
    echo ✅ Training completed successfully!
    echo.
    echo Results saved to: ml_model\checkpoints\
    echo   - dataset2_optimized.pth (best model weights)
    echo   - dataset2_history_optimized.csv (metrics)
    echo   - dataset2_meta_optimized.json (summary)
    pause
)
