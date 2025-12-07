"""
Configuration for VLM language-guided grasp prediction.
Updated for OCID-VLG dataset.
"""

CONFIG = {
    # =========================================================================
    # Paths
    # =========================================================================
    "ocid_vlg_path": "/tmp/OCID-VLG",  # Placeholder (will be overwritten by SLURM)
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",  # Updated to Qwen3
    "checkpoint_dir": "checkpoints/qwen3-grasp-vlg-lora",
    "results_dir": "results",

    # =========================================================================
    # Data
    # =========================================================================
    "image_size": (480, 640),  # OCID-VLG native size
    "dataset_version": "multiple",  # 'multiple', 'unique', 'novel-instances', 'novel-classes'
    "grasp_selection": "random",  # How to select grasp when multiple exist

    # =========================================================================
    # Quantization bins (for converting continuous grasps to text)
    # =========================================================================
    "x_bins": 1000,    # [0..999] for x coordinate
    "y_bins": 1000,    # [0..999] for y coordinate
    "theta_bins": 180, # [0..179] mapping from [-90, 90) degrees
    "w_bins": 1000,    # [0..999] for width
    "h_bins": 1000,    # [0..999] for height

    # =========================================================================
    # Augmentation
    # =========================================================================
    "aug_rotation_deg": 15,      # ±15 degree rotation
    "aug_color_jitter": True,
    "aug_label_noise_prob": 0.2, # 20% of samples get ±1 bin jitter (VLA-0 style)
    "aug_mask_field_prob": 0.1,  # 10% mask one field (VLA-0 style)

    # =========================================================================
    # LoRA configuration
    # =========================================================================
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # =========================================================================
    # Training
    # =========================================================================
    "num_epochs": 3,             # Optimized: Reduced to 3
    "train_batch_size": 8,       # Optimized: Fits A40/A100 w/ LoRA
    "eval_batch_size": 16,       # Optimized: Faster eval
    "gradient_accumulation_steps": 2,  # Effective batch = 16
    "dataloader_num_workers": 8, # Optimized: Faster I/O
    
    "learning_rate": 2e-4,
    "warmup_steps": 500,
    "lr_scheduler": "cosine",
    "bf16": True,
    "gradient_checkpointing": True,
    
    "logging_steps": 50,
    "eval_steps": 3000,          # Optimized: Once per epoch
    "save_steps": 3000,          # Optimized: Once per epoch
    "save_total_limit": 3,

    # =========================================================================
    # Inference
    # =========================================================================
    "generation_max_tokens": 20,
    "generation_temperature": 0.1,  # Low temperature for deterministic output
    "generation_top_p": 0.9,
    "use_constrained_decoding": True,  # Force digits-only output
    
    # Ensemble settings (from VLA-0)
    "ensemble_size": 4,  # H=4 for ensemble prediction

    # =========================================================================
    # Evaluation metrics
    # =========================================================================
    "iou_threshold": 0.25,       # Standard grasp detection threshold
    "angle_tolerance_deg": 30,   # Standard angle tolerance

    # =========================================================================
    # Debug
    # =========================================================================
    "debug": False,
    "visualize_samples": 10,
}


# =========================================================================
# Config variations for different experiments
# =========================================================================

CONFIG_ABLATION_NO_LANGUAGE = {
    **CONFIG,
    "checkpoint_dir": "checkpoints/qwen-grasp-no-lang",
    # This variant ignores the referring expression
    # Useful to measure how much language helps
}

CONFIG_NOVEL_CLASSES = {
    **CONFIG,
    "dataset_version": "novel-classes",
    "checkpoint_dir": "checkpoints/qwen-grasp-novel-classes",
    # Tests generalization to unseen object classes
}

CONFIG_NOVEL_INSTANCES = {
    **CONFIG,
    "dataset_version": "novel-instances",
    "checkpoint_dir": "checkpoints/qwen-grasp-novel-instances",
    # Tests generalization to unseen object instances
}


def get_config(variant: str = "default"):
    """Get config by variant name."""
    configs = {
        "default": CONFIG,
        "no_language": CONFIG_ABLATION_NO_LANGUAGE,
        "novel_classes": CONFIG_NOVEL_CLASSES,
        "novel_instances": CONFIG_NOVEL_INSTANCES,
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown config variant: {variant}. "
                        f"Available: {list(configs.keys())}")
    
    return configs[variant].copy()


if __name__ == '__main__':
    import json
    
    print("OCID-VLG Training Configuration")
    print("="*60)
    print(json.dumps(CONFIG, indent=2))
