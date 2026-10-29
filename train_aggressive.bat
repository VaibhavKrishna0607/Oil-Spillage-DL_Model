@echo off
REM Aggressive training for maximum improvement (2-step process)
REM Step 1: Initial fine-tune with standard parameters
REM Step 2: Extended fine-tune with lower learning rate for fine details

cd /d %~dp0

echo.
echo ================================================================================
echo Starting AGGRESSIVE SAR Oil Spill Model Training (2-Step Process)
echo ================================================================================
echo.
echo STEP 1: Initial Fine-tune (20 epochs)
echo   - Learning Rate: 5e-5
echo   - Batch Size: 16
echo   - Duration: ~15 minutes
echo.
echo STEP 2: Extended Fine-tune (15 epochs, lower LR)
echo   - Learning Rate: 1e-5 (finer optimization)
echo   - Batch Size: 8 (more gradient updates)
echo   - Duration: ~10 minutes
echo.
echo Total Duration: ~25-30 minutes
echo Expected Result: 0.85-0.92 Dice
echo.
echo ================================================================================
echo.

echo [Step 1/2] Running initial fine-tune...
python ml_model\finetune_optimized.py ^
  --epochs 20 ^
  --batch-size 16 ^
  --lr 5e-5 ^
  --model-base 64 ^
  --use-mixed-precision

if %errorlevel% neq 0 (
    echo.
    echo ❌ Step 1 failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ================================================================================
echo [Step 2/2] Running extended fine-tune with lower learning rate...
echo ================================================================================
echo.

REM Find the best model created in step 1
set BEST_MODEL=ml_model\checkpoints\dataset2_optimized.pth

if not exist "%BEST_MODEL%" (
    echo ❌ Could not find model from step 1: %BEST_MODEL%
    pause
    exit /b 1
)

REM For step 2, we use a slightly higher baseline dice (assume ~0.78 from step 1)
python ml_model\finetune_optimized.py ^
  --epochs 15 ^
  --batch-size 8 ^
  --lr 1e-5 ^
  --model-base 64 ^
  --baseline-dice 0.78 ^
  --patience 7 ^
  --use-mixed-precision

if %errorlevel% neq 0 (
    echo.
    echo ❌ Step 2 failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ================================================================================
echo ✅ AGGRESSIVE TRAINING COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
echo Results saved to: ml_model\checkpoints\
echo   - dataset2_optimized.pth (best model weights)
echo   - dataset2_history_optimized.csv (metrics)
echo   - dataset2_meta_optimized.json (summary)
echo.
echo Expected Final Dice: 0.85-0.92
echo.
pause
