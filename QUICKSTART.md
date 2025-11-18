# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- 1×A100 GPU (40GB or 80GB)
- Python 3.8+
- OCID-Grasp dataset downloaded
- Qwen3-VL-8B-Instruct (auto-downloaded from HuggingFace)

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python setup_check.py

# 3. Update data path
# Edit configs/config.py and set 'ocid_grasp_path' to your dataset location
```

## Fast Track (3 Commands)

```bash
# 1. Test data loading (30 seconds)
python data/test_loader.py

# 2. Sanity check (5 minutes)
python model/test_overfit.py --n_samples 10 --n_steps 200

# 3. Start training (8-12 hours)
python train/train_grasp_lora.py
```

## Using Makefile (Optional)

```bash
make check          # Verify setup
make test-loader    # Test data
make test-overfit   # Sanity check
make train          # Start training
make eval           # Evaluate model
```

## Step-by-Step Walkthrough

### Step 1: Verify Setup (30 seconds)

```bash
python setup_check.py
```

**Expected output:**
```
✓ Python 3.10.x
✓ torch
✓ transformers
✓ CUDA available: NVIDIA A100
✓ Config loaded
✓ Data directory exists
```

If any checks fail, install missing packages or update config.

### Step 2: Test Data Loading (30 seconds)

```bash
python data/test_loader.py
```

**Expected output:**
```
Loading dataset...
Dataset size: 45000
Train: 45000 grasp samples from 1200 images

Sample 0:
  Grasp string: 512 304 27 118 246
  Bins: {'x': 512, 'y': 304, ...}

Mean quantization errors:
  Center X: 0.32 px
  Center Y: 0.31 px
  Angle: 0.50 deg

Data loader test passed! ✓
```

**Visualization**: `data_loader_test.png` shows 10 samples with original (blue) and quantized (green) grasps overlaid.

### Step 3: Overfitting Test (5 minutes)

```bash
python model/test_overfit.py --n_samples 10 --n_steps 200
```

**Expected output:**
```
OVERFITTING TEST: 10 samples, 200 steps

Loading model from Qwen/Qwen3-VL-8B-Instruct...
Trainable params: 4,456,448 (0.51%)

Training...
Step 10: loss=5.234
Step 50: loss=2.103
Step 100: loss=0.892
Step 200: loss=0.234

Sample 0:
  Ground truth: 512 304 27 118 246
  Generated:    512 303 28 117 245
  ✓ MATCH

Overfitting accuracy: 8/10 = 80.0%
✓ Overfitting test PASSED (>60% accuracy)
```

**If this fails**: Something is wrong with the training pipeline. Check loss masking and tokenizer.

### Step 4: Full Training (8-12 hours)

```bash
python train/train_grasp_lora.py \
  --data_dir ./ocid-grasp \
  --checkpoint_dir ./checkpoints/run1 \
  --run_name grasp-vlm-run1
```

**Training progress:**
```
Epoch 1/10:   loss=3.456  val_loss=3.234  [2h]
Epoch 5/10:   loss=1.234  val_loss=1.456  [10h]
Epoch 10/10:  loss=0.892  val_loss=1.123  [20h]

Training complete!
Checkpoints saved to: ./checkpoints/run1
```

**Monitor training:**
- Live logs: `tail -f checkpoints/run1/logs/*.log`
- W&B dashboard (if enabled)
- Checkpoints saved every 500 steps

**Resume training:**
```bash
python train/train_grasp_lora.py \
  --resume_from_checkpoint ./checkpoints/run1/checkpoint-5000
```

### Step 5: Test Inference (10 seconds)

```bash
python inference/predict_grasp.py \
  --checkpoint ./checkpoints/run1/final \
  --image ./test_image.png \
  --ensemble 4 \
  --output prediction.png
```

**Output:**
```
Loading model from ./checkpoints/run1/final...
Predicting grasp (ensemble=4)...

Predicted grasp:
  Center: (320.5, 240.3)
  Angle: -12.7°
  Size: 45.2 x 89.1

Visualization saved to prediction.png
```

### Step 6: Full Evaluation (10 minutes)

```bash
python eval/evaluate_all.py \
  --checkpoint ./checkpoints/run1/final \
  --baselines
```

**Output:**
```
Loading test dataset...
Test dataset: 163 images

Evaluating model on 7500 samples...
100%|██████████| 7500/7500 [08:23<00:00, 14.9it/s]

Evaluating baselines...

EVALUATION RESULTS
================================================================
MODEL_H1
  mean_iou:                     0.451 (45.1%)
  success_rate@0.25:            0.723 (72.3%)
  mean_angle_error_deg:         12.34
  mean_center_error_px:         15.67

MODEL_H4 (Ensemble)
  mean_iou:                     0.487 (48.7%)
  success_rate@0.25:            0.765 (76.5%)
  mean_angle_error_deg:         10.12
  mean_center_error_px:         12.43

BASELINE_PCA
  mean_iou:                     0.152 (15.2%)
  mean_angle_error_deg:         24.56

BASELINE_MINAREA
  mean_iou:                     0.198 (19.8%)
  mean_angle_error_deg:         19.87

Results saved to results/evaluation_results.json
Visualizations saved to results/predictions_vis.png
```

---

## Troubleshooting

### Data not loading

**Error:** `FileNotFoundError: Could not find dataset`

**Fix:**
```python
# Edit configs/config.py
CONFIG = {
    "ocid_grasp_path": "/path/to/your/ocid-grasp",  # Update this!
    ...
}
```

### OOM errors

**Error:** `CUDA out of memory`

**Fix:** Reduce batch size in `configs/config.py`:
```python
"train_batch_size": 1,  # Reduce from 2
"gradient_accumulation_steps": 16,  # Increase to maintain effective batch
```

### Model not learning

**Check 1:** Run overfitting test
```bash
python model/test_overfit.py
```

Should reach >60% accuracy. If not, issue with pipeline.

**Check 2:** Visualize data loading
```bash
python data/test_loader.py
```

Verify quantization errors are small (<1 pixel, <1 degree).

**Check 3:** Check loss masking
Loss should only be computed on assistant tokens. Verify in logs.

### Invalid outputs

**Problem:** Model outputs "The grasp is at..." instead of "512 304 27 118 246"

**Fix:** Digits-only logits processor should be enabled by default. Check:
```python
# In inference/predict_grasp.py
use_constrained=True  # Should be True
```

---

## Next Steps

### Ablations

Try different LoRA ranks:
```bash
# Edit configs/config.py
"lora_r": 8   # vs 16 vs 32
```

Try different ensemble sizes:
```bash
python inference/predict_grasp.py --ensemble 1  # vs 4 vs 8
```

### Language Conditioning (OCID-VLG)

Download OCID-VLG and use instructions:
```bash
python inference/predict_grasp.py \
  --instruction "grasp the red mug"
```

### Hyperparameter Tuning

```python
# configs/config.py
"learning_rate": 1e-4,  # Try 5e-5, 2e-4, 5e-4
"lora_r": 32,           # Try 8, 16, 32
"aug_rotation_deg": 30, # Try 0, 15, 30
```

---

## Tips for Fast Iteration

1. **Start small**: Use `--max_samples 1000` for quick experiments
2. **Use checkpoints**: Resume from best checkpoint instead of retraining
3. **Ensemble last**: H=1 for development, H=4 for final results
4. **Monitor early**: Check first 100 steps; if loss doesn't drop, stop and debug
5. **Visualize often**: Use `inference/visualize.py` to inspect predictions

---

**You're ready to go! 🚀**

For detailed docs, see `README.md`.

For issues, check troubleshooting or open a GitHub issue.
