"""
Configuration for VLM grasp prediction.
Fast iteration research prototype style - just a simple dict!
"""

CONFIG = {
    # Paths
    "ocid_grasp_path": "./ocid-grasp",  # Adjust to your download path
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "checkpoint_dir": "./checkpoints/qwen3-grasp-lora",
    "results_dir": "./results",

    # Data
    "image_size": (480, 640),  # OCID-Grasp native size
    "train_split": 0.8,
    "val_split": 0.1,
    # test_split = 0.1 (remainder)

    # Quantization bins
    "x_bins": 1000,  # [0..1000] for x coordinate
    "y_bins": 1000,  # [0..1000] for y coordinate
    "theta_bins": 180,  # [0..179] mapping from [-90, 90) degrees
    "w_bins": 1000,  # [0..1000] for width
    "h_bins": 1000,  # [0..1000] for height

    # Augmentation
    "aug_rotation_deg": 15,  # ±15 degree rotation
    "aug_color_jitter": True,
    "aug_label_noise_prob": 0.2,  # 20% of samples get ±1 bin jitter
    "aug_mask_field_prob": 0.1,  # 10% mask one field

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],

    # Training
    "num_epochs": 10,
    "train_batch_size": 2,
    "eval_batch_size": 4,
    "gradient_accumulation_steps": 8,  # Effective batch = 16
    "learning_rate": 2e-4,
    "warmup_steps": 500,
    "lr_scheduler": "cosine",
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 3,

    # Inference
    "generation_max_tokens": 20,
    "generation_temperature": 0.2,
    "generation_top_p": 0.9,
    "ensemble_size": 4,  # H=4 for ensemble

    # Evaluation
    "iou_threshold": 0.25,
    "angle_tolerance_deg": 30,

    # Debug
    "debug": True,
    "visualize_samples": 10,
}
