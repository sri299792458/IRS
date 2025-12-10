# VLM-Based Language-Guided Grasp Detection

A Vision-Language Model approach for language-guided robotic grasping using Qwen3-VL-8B with LoRA fine-tuning.

**Repository:** [github.com/sri299792458/vlm-grasp](https://github.com/sri299792458/vlm-grasp)

## Overview

This project fine-tunes Qwen3-VL-8B to predict grasp rectangles from natural language instructions. Given an image and a referring expression like *"grasp the red apple on the left"*, the model outputs an oriented grasp rectangle as 5 discrete tokens.

### Key Results on OCID-VLG Benchmark

| Method | Grasp Success Rate |
|--------|-------------------|
| CROG (CoRL 2023) | 77.2% |
| ETRG (ICRA 2025) | 82.3% |
| MapleGrasp (2025) | 88.2% |
| **Ours** | **93.4%** |

## Approach

### Architecture
```
Input: RGB Image (480×640) + Referring Expression
   ↓
[Qwen3-VL-8B Vision Encoder + Multimodal Transformer]
   ↓
[LoRA Adapters (r=16, ~4.5M trainable params)]
   ↓
[Constrained Text Generation]
   ↓
Output: "512 304 27 118 246" (x y θ w h)
   ↓
[Dequantization]
   ↓
Grasp Rectangle: {cx, cy, θ, w, h}
```

### Key Design Decisions

1. **Text-as-Action**: Grasp parameters encoded as discrete tokens (inspired by VLA-0)
2. **Quantization**: 1000 bins for positions (~0.6px error), 180 bins for angles (~1° error)
3. **Constrained Decoding**: Forces digit-only output, reducing invalid predictions to <1%
4. **LoRA Fine-tuning**: Only 0.51% of parameters trained, enabling efficient adaptation

## Installation

```bash
git clone https://github.com/sri299792458/vlm-grasp.git
cd vlm-grasp

pip install -r requirements.txt
```

**Requirements:**
- 1× A100 GPU (40GB) or equivalent


## Dataset Setup

Download OCID-VLG from the [official source](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view).

```bash
mkdir -p data/OCID-VLG
cd data/OCID-VLG
unzip ocid_vlg.zip
```

Expected structure:
```
OCID-VLG/
├── refer/
│   └── multiple/
│       ├── train_expressions.json
│       ├── val_expressions.json
│       └── test_expressions.json
└── [image sequences...]
```

Update the path in `OCID-VLG/config_vlg.py`:
```python
"ocid_vlg_path": "/path/to/your/OCID-VLG"
```

## Training

### Standard Training (Multiple Split)

```bash
python OCID-VLG/train_grasp_vlg.py \
    --data_dir /path/to/OCID-VLG \
    --checkpoint_dir ./checkpoints/vlg-multiple \
    --version multiple
```

### Novel Classes Generalization

```bash
python OCID-VLG/experiment_novel_classes.py train \
    --data_dir /path/to/OCID-VLG \
    --checkpoint_dir ./checkpoints/vlg-novel-classes
```

### Training Options

| Argument | Description |
|----------|-------------|
| `--data_dir` | Path to OCID-VLG dataset |
| `--checkpoint_dir` | Where to save checkpoints |
| `--version` | Dataset split: `multiple`, `unique`, `novel-instances`, `novel-classes` |
| `--resume_from_checkpoint` | Resume from a saved checkpoint |
| `--no_wandb` | Disable Weights & Biases logging |

**Training Time:** ~ 12 hours on A100 for 3 epochs (~60k samples)

## Evaluation

### Evaluate on Test Set

```bash
python OCID-VLG/evaluate_vlg.py \
    --checkpoint ./checkpoints/vlg-multiple/final \
    --data_dir /path/to/OCID-VLG \
    --split test \
    --version multiple
```

### Novel Classes Evaluation

```bash
python OCID-VLG/experiment_novel_classes.py evaluate \
    --checkpoint ./checkpoints/vlg-novel-classes/final \
    --data_dir /path/to/OCID-VLG
```


## Project Structure

```
vlm-grasp/
├── model/
│   ├── qwen3_grasp_model.py      # Model loading with LoRA
│   └── constrained_decoding.py   # Digit-only generation
├── train/
│   └── data_collator.py          # Loss masking for VLM training
├── inference/
│   └── visualize.py              # Grasp rectangle visualization
├── OCID-VLG/
│   ├── config_vlg.py             # Configuration
│   ├── ocid_vlg_loader.py        # Dataset loader
│   ├── torch_dataset_vlg.py      # PyTorch dataset wrapper
│   ├── chat_formatter_vlg.py     # Qwen3-VL chat formatting
│   ├── grasp_quantizer.py        # Continuous ↔ discrete conversion
│   ├── augmentations.py          # Data augmentation
│   ├── train_grasp_vlg.py        # Training script
│   ├── evaluate_vlg.py           # Evaluation script
│   ├── experiment_novel_classes.py  # Novel class generalization
│   ├── ablation_expression_type.py  # Expression type analysis
│   └── analyze_failures.py       # Failure case analysis
├── requirements.txt
└── README.md
```

## Configuration

Key parameters in `OCID-VLG/config_vlg.py`:

```python
CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    
    # Training
    "num_epochs": 3,
    "train_batch_size": 8,
    "learning_rate": 2e-4,
    
    # Quantization (grasp → tokens)
    "x_bins": 1000,
    "y_bins": 1000,
    "theta_bins": 180,
    
    # Evaluation
    "iou_threshold": 0.25,
    "angle_tolerance_deg": 30,
}
```



## Citation

```bibtex
@misc{vlm-grasp-2025,
  title={VLM-Based Language-Guided Grasp Detection with Qwen3-VL},
  author={Srinivas Kanth},
  year={2025},
  howpublished={\url{https://github.com/sri299792458/vlm-grasp}}
}
```

## Acknowledgments

- **OCID-VLG Dataset**: Tziafas et al., CoRL 2023
- **OCID-Grasp**: Ainetter & Fraundorfer, ICRA 2021
- **Qwen3-VL**: Alibaba Qwen Team
- **VLA Inspiration**: VLA-0

## License

MIT License