# AI Training Prompts — Progressive Strategy for Better Model Performance

This file contains multiple AI prompts designed to progressively improve your oil spill detection model. Use these prompts with AI coding assistants (Claude, GitHub Copilot, etc.) in sequence or as fallbacks.

---
these prompts(1-6) is for ml model training
## 🎯 STRATEGY 1: Enhanced Training Configuration

### Prompt 1.1: Optimize Training Parameters
```
I have a U-Net model for oil spill detection on SAR images. Current training:
- Dataset: 5,255 train, 1,615 val, 1,200 test samples (256×256)
- Current config: 15 epochs, batch_size=8, base=32, lr=1e-4
- Current results: Val IoU ~0.3-0.4 (needs improvement)

Please suggest optimized training parameters to improve IoU to 0.70+:
1. Better learning rate schedule (cosine annealing, warmup, etc.)
2. Optimal batch size for CPU training
3. Better loss function configuration
4. Data augmentation improvements
5. Training duration recommendations

Provide specific code changes for ml_model/train.py
```

### Prompt 1.2: Advanced Loss Functions
```
My current loss function is BCE + Dice (0.5 + 0.5) with pos_weight=2.9.
The model struggles with class imbalance (25.4% oil pixels, 74.6% background).

Please implement and compare these advanced loss functions:
1. Focal Loss (with appropriate gamma and alpha)
2. Tversky Loss (with beta parameter)
3. Combo Loss (BCE + Dice + Focal)
4. Class-balanced focal loss

Create a new file ml_model/advanced_losses.py with implementations
and modify train.py to use the best one.
```

### Prompt 1.3: Learning Rate Scheduling
```
Current training uses constant learning rate (1e-4). Please implement:
1. Cosine annealing with warmup
2. ReduceLROnPlateau (reduce LR when validation loss plateaus)
3. OneCycleLR (fast and effective for image segmentation)
4. AdamW optimizer instead of Adam

Modify ml_model/train.py to include these schedulers and
add command-line arguments to choose between them.
```

---

## 🔧 STRATEGY 2: Model Architecture Improvements

### Prompt 2.1: Enhanced U-Net Architecture
```
Current U-Net has base=32 (smaller for CPU). Please implement:
1. Attention U-Net (attention gates in skip connections)
2. Residual U-Net (residual connections in blocks)
3. Deep Supervision (auxiliary loss at different scales)
4. Larger base size (64) with efficient CPU optimizations

Create ml_model/model_enhanced.py with these architectures
and keep the original as fallback.
```

### Prompt 2.2: Pre-trained Backbones
```
Instead of training from scratch, please implement transfer learning:
1. Use ResNet-34 or EfficientNet as encoder (pre-trained on ImageNet)
2. Decoder with skip connections (like U-Net)
3. Freeze encoder initially, then fine-tune
4. Compare performance vs. training from scratch

Create ml_model/model_transfer.py with transfer learning implementation.
```

### Prompt 2.3: Multi-scale Processing
```
Please implement a multi-scale U-Net that:
1. Processes input at multiple scales (256×256, 128×128, 64×64)
2. Combines predictions from different scales
3. Helps detect both large and small oil spills
4. Improves boundary detection

Create ml_model/model_multiscale.py with multi-scale architecture.
```

---

## 📊 STRATEGY 3: Data Augmentation Strategy

### Prompt 3.1: Advanced Augmentations
```
Current augmentations: horizontal/vertical flip, 90° rotation, brightness/contrast jitter.

Please implement SAR-specific augmentations using albumentations:
1. Speckle noise injection (simulates SAR noise)
2. Gaussian blur (simulates atmospheric effects)
3. Elastic deformations (simulates water movement)
4. Cutout and MixUp (regularization)
5. Random gamma correction (radiometric variation)

Modify data_pipeline/augmentations.py with these advanced transforms.
```

### Prompt 3.2: Test-Time Augmentation (TTA)
```
Please implement test-time augmentation to improve inference:
1. Apply multiple augmentations to test images
2. Average predictions across augmentations
3. Common TTA transforms: flips, rotations, small crops
4. Expected IoU improvement: 2-5%

Create ml_model/tta.py with TTA implementation and evaluation.
```

### Prompt 3.3: Oversampling Minority Class
```
Oil pixels are only 25.4% of the dataset. Please implement:
1. Oversample images with high oil content
2. Create weighted sampler for DataLoader
3. Focal loss to focus on hard examples
4. Compare balanced vs. imbalanced training

Modify data_pipeline/dataloaders.py with oversampling strategies.
```

---

## 🎨 STRATEGY 4: Post-Processing & Refinement

### Prompt 4.1: CRF Post-Processing
```
Model predictions can be noisy. Please implement:
1. Conditional Random Field (CRF) post-processing
2. Using pydensecrf or similar library
3. Refines boundaries and removes small false positives
4. Expected IoU improvement: 3-5%

Create ml_model/postprocessing.py with CRF implementation.
```

### Prompt 4.2: Morphological Operations
```
Please implement morphological post-processing:
1. Opening to remove small false positives
2. Closing to fill small holes
3. Area-based filtering (remove very small detections)
4. Boundary smoothing with morphological operations

Create ml_model/morphological.py with these operations.
```

### Prompt 4.3: Ensemble Methods
```
Please implement model ensembling:
1. Train multiple models with different seeds
2. Average predictions from multiple models
3. Weighted averaging based on validation performance
4. Expected IoU improvement: 2-4%

Create ml_model/ensemble.py with ensemble training and inference.
```

---

## 🚀 STRATEGY 5: Training Optimization

### Prompt 5.1: Mixed Precision Training
```
CPU training is slow. Please implement:
1. Automatic Mixed Precision (AMP) if GPU available
2. Gradient accumulation to simulate larger batch sizes
3. Efficient data loading with prefetching
4. Memory optimization techniques

Modify ml_model/train.py with these optimizations.
```

### Prompt 5.2: Progressive Training
```
Please implement progressive training strategy:
1. Start with smaller images (128×128) for fast initial learning
2. Gradually increase to full size (256×256)
3. Freeze early layers, fine-tune later layers
4. Expected training time reduction: 30-40%

Create ml_model/progressive_training.py with this strategy.
```

### Prompt 5.3: Curriculum Learning
```
Please implement curriculum learning:
1. Start with easy examples (clear oil spills)
2. Gradually introduce harder examples (look-alikes, small spills)
3. Dynamic difficulty adjustment based on model performance
4. Expected faster convergence and better final performance

Create ml_model/curriculum_learning.py with curriculum implementation.
```

---

## 🔍 STRATEGY 6: Analysis & Debugging

### Prompt 6.1: Error Analysis
```
Please create a comprehensive error analysis tool:
1. Identify which images the model performs worst on
2. Analyze failure modes (false positives vs. false negatives)
3. Visualize difficult examples
4. Suggest targeted improvements based on analysis

Create ml_model/error_analysis.py with analysis and visualization.
```

### Prompt 6.2: Per-Sensor Analysis
```
Dataset has two sensors: PALSAR (L-band) and Sentinel-1 (C-band).
Please implement:
1. Separate training and evaluation per sensor
2. Sensor-specific data augmentation
3. Sensor-specific model fine-tuning
4. Report per-sensor metrics and identify which is harder

Modify ml_model/train.py to support sensor-specific training.
```

### Prompt 6.3: Uncertainty Quantification
```
Please implement uncertainty estimation:
1. Monte Carlo dropout during inference
2. Ensemble uncertainty estimation
3. Identify where model is uncertain
4. Use uncertainty to guide data collection

Create ml_model/uncertainty.py with uncertainty estimation methods.
```

Vaibhav's prompts ends

---
this prompts(7-8) is shared responsibility
## 🌐 STRATEGY 7: Cloud GPU Training

### Prompt 7.1: Google Colab Setup
```
Please create a complete Google Colab notebook for GPU training:
1. Setup cell (install dependencies, mount Google Drive)
2. Data loading cell (upload or download dataset)
3. Training cell with GPU optimization
4. Evaluation and visualization
5. Model export and download

Create notebooks/train_gpu_colab.ipynb with complete setup.
```

### Prompt 7.2: Kaggle Kernel Setup
```
Please create a Kaggle kernel script:
1. Dataset preparation for Kaggle environment
2. GPU-optimized training configuration
3. Save model to Kaggle output
4. Submit to competition format if applicable

Create kaggle/train_gpu_kernel.py with Kaggle-specific setup.
```

### Prompt 7.3: AWS/GCP GPU Setup
```
Please create cloud GPU training scripts:
1. AWS EC2 setup with GPU instance
2. Docker container with all dependencies
3. Training script optimized for multi-GPU
4. Cost optimization (spot instances, etc.)

Create cloud/aws_train.py and cloud/gcp_train.py with cloud setup.
```

---

## 📈 STRATEGY 8: Evaluation & Metrics

### Prompt 8.1: Comprehensive Metrics
```
Current metrics: IoU, Dice, F1, Precision, Recall.

Please implement additional metrics:
1. Boundary IoU (focuses on edge accuracy)
2. Hausdorff distance (boundary quality)
3. Average precision (detection quality)
4. Per-class metrics (oil vs. background)
5. Confidence calibration metrics

Create ml_model/metrics.py with comprehensive evaluation.
```

### Prompt 8.2: Visualization Tools
```
Please create visualization tools for model analysis:
1. Prediction vs. ground truth overlays
2. Confidence heatmaps
3. Error maps (where model fails)
4. Activation maps (what model focuses on)
5. Training curves visualization

Create ml_model/visualization.py with these visualization tools.
```

### Prompt 8.3: Ablation Studies
```
Please create ablation study framework:
1. Test importance of each component (augmentation, loss function, etc.)
2. Compare different architectures
3. Test impact of different training strategies
4. Provide recommendations based on results

Create ml_model/ablation_study.py with systematic testing.
```
shared responsibility prompts end here

---

this prompt(9) is for backend developer-Anany/Uday
## 🎯 STRATEGY 9: Transfer to Production

### Prompt 9.1: Model Optimization
```
Please optimize the model for deployment:
1. Model quantization (FP16, INT8)
2. Pruning (remove unimportant weights)
3. Knowledge distillation (teacher-student)
4. Expected speedup: 2-4x, size reduction: 2-4x

Create ml_model/optimization.py with optimization techniques.
```

### Prompt 9.2: ONNX Export & Optimization
```
Please create robust ONNX export pipeline:
1. Export to ONNX with dynamic batch size
2. ONNX optimization (constant folding, etc.)
3. ONNX Runtime setup for fast inference
4. Benchmark ONNX vs. PyTorch performance

Create ml_model/onnx_export.py with complete ONNX pipeline.
```

### Prompt 9.3: API Integration
```
Please create FastAPI endpoints for model inference:
1. POST /predict - single image inference
2. POST /batch_predict - batch inference
3. GET /model_info - model metadata
4. POST /evaluate - evaluation on test set

Create api/inference_api.py with complete API implementation.
```

---

this prompt(10) is for backend developer-Anany/Uday
## 🔄 STRATEGY 10: Iterative Improvement Loop

### Prompt 10.1: Automated Hyperparameter Tuning
```
Please implement automated hyperparameter tuning:
1. Use Optuna or Hyperopt for optimization
2. Search space: learning rate, batch size, loss weights, etc.
3. Objective: maximize validation IoU
4. Early stopping to save time

Create ml_model/hyperparameter_tuning.py with automated tuning.
```

### Prompt 10.2: Active Learning
```
Please implement active learning loop:
1. Identify uncertain predictions
2. Select most informative samples for labeling
3. Retrain on augmented dataset
4. Expected better performance with less data

Create ml_model/active_learning.py with active learning pipeline.
```

### Prompt 10.3: Continuous Training Pipeline
```
Please create a continuous training pipeline:
1. Automated data pipeline updates
2. Scheduled retraining
3. Model versioning and rollback
4. Performance monitoring and alerts

Create scripts/continuous_training.py with automation.
```

---

## 🚨 FALLBACK STRATEGIES

### Fallback 1: If GPU Training Fails
```
GPU training is not available or too expensive. Please:
1. Optimize CPU training with gradient accumulation
2. Use smaller model architecture
3. Train on subset of data, then fine-tune
4. Implement early stopping to save time

Create ml_model/cpu_optimized_training.py with CPU-specific optimizations.
```

### Fallback 2: If Model Performance Plateaus
```
Model IoU stuck at 0.5-0.6. Please try:
1. Increase model capacity (more layers, filters)
2. Collect more diverse training data
3. Implement ensemble methods
4. Try different loss functions and schedulers

Create ml_model/plateau_breakthrough.py with breakthrough strategies.
```

### Fallback 3: If Training is Unstable
```
Training loss is NaN or oscillating. Please:
1. Reduce learning rate
2. Add gradient clipping
3. Implement learning rate warmup
4. Check for data issues (corrupted images, etc.)

Create ml_model/stable_training.py with stability improvements.
```

### Fallback 4: If Overfitting Occurs
```
Training IoU >> Validation IoU. Please:
1. Increase data augmentation
2. Add dropout layers
3. Implement early stopping
4. Use L2 regularization

Create ml_model/overfitting_prevention.py with regularization techniques.
```

---

## 📋 EXECUTION ORDER

### Phase 1: Quick Wins (1-2 days)
1. **Prompt 1.1**: Optimize training parameters
2. **Prompt 1.3**: Learning rate scheduling
3. **Prompt 3.1**: Advanced augmentations

### Phase 2: Architecture Improvements (2-3 days)
4. **Prompt 2.1**: Enhanced U-Net
5. **Prompt 4.2**: Morphological post-processing
6. **Prompt 6.1**: Error analysis

### Phase 3: Advanced Techniques (3-4 days)
7. **Prompt 1.2**: Advanced loss functions
8. **Prompt 4.1**: CRF post-processing
9. **Prompt 5.1**: Mixed precision training

### Phase 4: Production Optimization (2-3 days)
10. **Prompt 7.1**: Google Colab GPU training
11. **Prompt 9.1**: Model optimization
12. **Prompt 9.2**: ONNX export

### Phase 5: Continuous Improvement (ongoing)
13. **Prompt 10.1**: Hyperparameter tuning
14. **Prompt 10.2**: Active learning
15. **Prompt 8.1**: Comprehensive metrics

---

## 🎯 SUCCESS CRITERIA

By the end of this process, you should achieve:
- **Validation IoU**: 0.70-0.75 (from current 0.3-0.4)
- **Inference time**: <100ms per image (256×256)
- **Model size**: <50MB (after optimization)
- **Robustness**: Works on both PALSAR and Sentinel-1 data
- **Production-ready**: ONNX export, API integration

---

## 💡 HOW TO USE THESE PROMPTS

1. **Start with Phase 1** (quick wins)
2. **Test each improvement** before moving to next
3. **Keep backup of working versions**
4. **Use fallback prompts** if something fails
5. **Document what works** for your specific dataset

Each prompt is designed to be independent, so you can skip or reorder based on your needs and time constraints.

---

## 📊 EXPECTED IMPROVEMENTS

| Strategy | Expected IoU Gain | Time Investment |
|----------|------------------|------------------|
| Phase 1 (Quick Wins) | +0.10-0.15 | 1-2 days |
| Phase 2 (Architecture) | +0.08-0.12 | 2-3 days |
| Phase 3 (Advanced) | +0.05-0.10 | 3-4 days |
| Phase 4 (GPU Training) | +0.10-0.15 | 2-3 days |
| Phase 5 (Continuous) | +0.05-0.08 | Ongoing |

**Total Expected**: IoU 0.70-0.75 (from current 0.3-0.4)

---

## 🚀 START NOW

Begin with **Prompt 1.1** to optimize your current training configuration. This should give you immediate improvements with minimal code changes.

Good luck with your model training! 🎯
