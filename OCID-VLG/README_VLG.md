# VLM Language-Guided Grasp Detection (OCID-VLG)

This codebase implements a VLA-0 style approach for **language-guided grasp detection** using Vision-Language Models.

## Key Idea

Instead of adding special tokens or action heads, we simply ask the VLM to output grasp poses as text:

```
Input:  Image + "Grasp the red apple on the left"
Output: "512 304 89 118 246"  (x, y, theta, w, h as integers)
```

## Setup

### 1. Download OCID-VLG Dataset

Download from: https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view

Extract to your desired location:
```bash
# Example
mkdir -p /data/OCID-VLG
cd /data/OCID-VLG
unzip ocid_vlg.zip
```

The dataset structure should be:
```
OCID-VLG/
├── multiple/
│   ├── train.pkl (or train.json)
│   ├── val.pkl
│   └── test.pkl
├── unique/
├── novel-instances/
└── novel-classes/
```

### 2. Install Dependencies

```bash
pip install torch transformers peft accelerate bitsandbytes
pip install pillow numpy opencv-python matplotlib
pip install shapely  # For IoU computation
pip install wandb    # Optional, for logging
```

### 3. Update Config

Edit `config_vlg.py`:
```python
CONFIG = {
    "ocid_vlg_path": "/data/OCID-VLG",  # UPDATE THIS
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    ...
}
```

### 4. Verify Dataset Loading

```bash
python ocid_vlg_loader.py --root /data/OCID-VLG --version multiple
```

Expected output:
```
OCID-VLG [multiple/train]: Loaded XXXXX samples

  Sample 0:
    Sentence: 'the red apple on the left side of the table'
    Target: apple
    Num grasps: 5
    First grasp: cx=320.5, cy=240.3, θ=-15.0°
```

## Training

```bash
python train_grasp_vlg.py \
    --data_dir /data/OCID-VLG \
    --version multiple \
    --checkpoint_dir checkpoints/grasp-vlg
```

### Training Options

```bash
# Different dataset versions
--version multiple          # Multiple expressions per target (default)
--version unique            # One expression per target
--version novel-instances   # Test on unseen object instances
--version novel-classes     # Test on unseen object classes

# Resume training
--resume_from_checkpoint checkpoints/grasp-vlg/checkpoint-1000

# Disable W&B
--no_wandb
```

## Evaluation

```bash
python evaluate_vlg.py \
    --checkpoint checkpoints/grasp-vlg/final \
    --data_dir /data/OCID-VLG \
    --split test \
    --version multiple \
    --visualize
```

### Expected Results

Based on prior work, target metrics for OCID-VLG:

| Method | Grasp Success Rate |
|--------|-------------------|
| CROG (baseline) | ~75-80% |
| MapleGrasp | ~82-85% |
| **VLM (this work)** | TBD |

## Dataset Versions

OCID-VLG provides four evaluation protocols:

1. **multiple** (default): Multiple referring expressions per target object
   - Tests general language understanding
   - Easiest version

2. **unique**: One referring expression per target object
   - Tests precise language grounding
   - Harder than multiple

3. **novel-instances**: Test on unseen object instances
   - All classes seen during training
   - Tests instance-level generalization

4. **novel-classes**: Test on unseen object classes
   - Some classes never seen during training
   - Tests zero-shot generalization

## Key Differences from Original OCID-Grasp

| OCID-Grasp | OCID-VLG |
|------------|----------|
| No language | Referring expressions |
| All grasps in scene | Grasps for specific target |
| Ambiguous training | Unambiguous training |
| Dense prediction | Single prediction |

## Code Structure

```
├── config_vlg.py           # Configuration
├── ocid_vlg_loader.py      # Dataset loading
├── chat_formatter_vlg.py   # Format for Qwen-VL
├── torch_dataset_vlg.py    # PyTorch dataset
├── train_grasp_vlg.py      # Training script
├── evaluate_vlg.py         # Evaluation script
│
├── grasp_quantizer.py      # (from original) Continuous ↔ discrete
├── augmentations.py        # (from original) Data augmentation
└── model/
    ├── qwen3_grasp_model.py
    └── constrained_decoding.py
```

## Ablation Studies

### 1. With vs Without Language

Train without using the referring expression:
```bash
python train_grasp_vlg.py --config_variant no_language
```

### 2. Novel Classes Generalization

Test on unseen object classes:
```bash
python train_grasp_vlg.py --version novel-classes
python evaluate_vlg.py --version novel-classes
```

## Tips

1. **Start with `multiple` version** - it's the easiest and has the most data

2. **Use constrained decoding** - forces output to be digits only

3. **Grasp selection during training**:
   - `random`: Different grasp each epoch (augmentation effect)
   - `first`: Deterministic (for debugging)
   - `center`: Prefer central grasps

4. **Evaluation**: Success if prediction matches ANY valid grasp for target

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tziafas2023language,
  title={Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter},
  author={Tziafas, Georgios and others},
  booktitle={Conference on Robot Learning},
  year={2023}
}
```

## Troubleshooting

**Q: "No annotations found"**
A: Check your data path and ensure the pickle files exist.

**Q: "CUDA out of memory"**
A: Reduce batch size or use gradient checkpointing.

**Q: "Model outputs garbage"**
A: Ensure constrained decoding is enabled. Check that the model is loaded correctly.
