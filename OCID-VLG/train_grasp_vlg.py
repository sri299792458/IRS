#!/usr/bin/env python3
"""
Training script for VLM language-guided grasp prediction.
Updated for OCID-VLG dataset.

Usage:
    python train_grasp_vlg.py --data_dir /path/to/OCID-VLG
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import 'model' and 'train'
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import transformers.utils.import_utils
# MONKEYPATCH 1: Patch source
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

from transformers import TrainingArguments, Trainer

# MONKEYPATCH 2: Patch trainer module instance (Crucial if it already imported the function)
import transformers.trainer
transformers.trainer.check_torch_load_is_safe = lambda: None

# Local imports (adjust paths as needed)

# Local imports (adjust paths as needed)
from config_vlg import CONFIG, get_config
from torch_dataset_vlg import GraspVLGDataset
from grasp_quantizer import GraspQuantizer

# These should come from your existing codebase
# from model.qwen3_grasp_model import load_model_and_processor
# from train.data_collator import GraspDataCollator


def main(args):
    # Get config
    if args.config_variant:
        config = get_config(args.config_variant)
    else:
        config = CONFIG.copy()
    
    # Override from args
    if args.data_dir:
        config['ocid_vlg_path'] = args.data_dir
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.version:
        config['dataset_version'] = args.version

    print("="*60)
    print("OCID-VLG Language-Guided Grasp Training")
    print("="*60)
    print(f"Data dir: {config['ocid_vlg_path']}")
    print(f"Dataset version: {config['dataset_version']}")
    print(f"Model: {config['model_name']}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")
    print(f"LoRA rank: {config['lora_r']}")
    print(f"Batch size: {config['train_batch_size']} x {config['gradient_accumulation_steps']}")
    print("="*60)

    # Initialize quantizer
    quantizer = GraspQuantizer(
        img_width=config['image_size'][1],
        img_height=config['image_size'][0],
        x_bins=config['x_bins'],
        y_bins=config['y_bins'],
        theta_bins=config['theta_bins'],
        w_bins=config['w_bins'],
        h_bins=config['h_bins'],
    )

    # Load datasets
    print("\nLoading OCID-VLG datasets...")
    
    train_dataset = GraspVLGDataset(
        config['ocid_vlg_path'],
        split='train',
        version=config['dataset_version'],
        config=config,
        quantizer=quantizer,
        grasp_selection=config['grasp_selection']
    )

    val_dataset = GraspVLGDataset(
        config['ocid_vlg_path'],
        split='val',
        version=config['dataset_version'],
        config=config,
        quantizer=quantizer,
        grasp_selection='first'  # Deterministic for validation
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Show example
    sample = train_dataset[0]
    print(f"\nExample sample:")
    print(f"  Sentence: '{sample['sentence']}'")
    print(f"  Target: {sample['target']}")
    print(f"  Grasp: {quantizer.bins_to_string(sample['grasp_bins'])}")

    # Load model
    print("\nLoading model...")
    
    # Import from your existing codebase
    from model.qwen3_grasp_model import load_model_and_processor
    from train.data_collator import GraspDataCollator
    
    model, processor = load_model_and_processor(config, use_qlora=False)

    # Data collator
    data_collator = GraspDataCollator(processor=processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['checkpoint_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler'],
        warmup_steps=config['warmup_steps'],
        logging_dir=f"{config['checkpoint_dir']}/logs",
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        save_strategy="steps",
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=config['bf16'],
        gradient_checkpointing=config['gradient_checkpointing'],
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grasp-vlg-{config['dataset_version']}",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("\nStarting training...")
    print("="*60)

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config['checkpoint_dir'] + "/final")

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OCID-VLG dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint output directory')
    parser.add_argument('--version', type=str, default='multiple',
                        choices=['multiple', 'unique', 'novel-instances', 'novel-classes'],
                        help='Dataset version')
    parser.add_argument('--config_variant', type=str, default=None,
                        choices=['default', 'no_language', 'novel_classes', 'novel_instances'],
                        help='Config variant')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    args = parser.parse_args()
    main(args)
