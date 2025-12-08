#!/usr/bin/env python3
"""
Novel Classes Generalization Experiment.

Tests the model's ability to generalize to object classes
that were never seen during training.

OCID-VLG provides a 'novel-classes' split where:
- Training: Only certain object classes
- Testing: Different object classes (zero-shot for classes)

This is a strong test of generalization - can the model
understand "grasp the apple" even if it never saw apples in training?
"""
import os
import sys
import argparse
import json
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).parent))
# Add parent directory (IRS) for 'model' and 'train' packages
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from config_vlg import CONFIG
from torch_dataset_vlg import GraspVLGDataset, GraspVLGEvalDataset
from grasp_quantizer import GraspQuantizer
from evaluate_vlg import GraspVLGEvaluator, print_metrics


def train_novel_classes(args):
    """Train on novel-classes split (excludes certain classes from training)."""
    from transformers import TrainingArguments, Trainer
    from model.qwen3_grasp_model import load_model_and_processor
    from train.data_collator import GraspDataCollator
    
    config = CONFIG.copy()
    config['ocid_vlg_path'] = args.data_dir
    config['dataset_version'] = 'novel-classes'  # KEY: Use novel-classes split
    config['checkpoint_dir'] = args.checkpoint_dir or 'checkpoints/qwen3-grasp-novel-classes'
    config['num_epochs'] = 6  # Override to 6 epochs per user request
    
    print("="*60)
    print("NOVEL CLASSES GENERALIZATION TRAINING")
    print("="*60)
    print("Training on 'novel-classes' split:")
    print("- Certain object classes are EXCLUDED from training")
    print("- Test set contains ONLY these excluded classes")
    print("- Tests zero-shot generalization to new object types")
    print("="*60)
    
    quantizer = GraspQuantizer(
        img_width=config['image_size'][1],
        img_height=config['image_size'][0],
    )
    
    # Load datasets with novel-classes version
    train_dataset = GraspVLGDataset(
        config['ocid_vlg_path'],
        split='train',
        version='novel-classes',
        config=config,
        quantizer=quantizer,
    )
    
    val_dataset = GraspVLGDataset(
        config['ocid_vlg_path'],
        split='val',
        version='novel-classes',
        config=config,
        quantizer=quantizer,
        grasp_selection='first',
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Check what classes are in training
    train_classes = set()
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        train_classes.add(sample['target'])
    
    print(f"\nClasses in training (sample): {sorted(train_classes)[:10]}...")
    
    # Load model
    from model.qwen3_grasp_model import load_model_and_processor
    from train.data_collator import GraspDataCollator
    
    model, processor = load_model_and_processor(config, use_qlora=False)
    data_collator = GraspDataCollator(processor=processor)
    
    training_args = TrainingArguments(
        output_dir=config['checkpoint_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        save_strategy="steps",
        save_steps=config['save_steps'],
        bf16=config['bf16'],
        gradient_checkpointing=config['gradient_checkpointing'],
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        run_name="grasp-vlg-NOVEL-CLASSES",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(config['checkpoint_dir'] + "/final")
    
    print(f"\nModel saved to: {config['checkpoint_dir']}/final")


def evaluate_novel_classes(args):
    """Evaluate on novel classes test set."""
    config = CONFIG.copy()
    config['ocid_vlg_path'] = args.data_dir
    
    print("="*60)
    print("NOVEL CLASSES EVALUATION")
    print("="*60)
    
    # Load test dataset with novel-classes version
    dataset = GraspVLGEvalDataset(
        args.data_dir,
        split='test',
        version='novel-classes',
        config=config,
    )
    
    print(f"Test samples: {len(dataset)}")
    
    # Check what classes are in test (should be DIFFERENT from training)
    test_classes = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        test_classes.add(sample['target'])
    
    print(f"Novel classes in test: {sorted(test_classes)}")
    
    # Evaluate
    evaluator = GraspVLGEvaluator(args.checkpoint, config)
    
    results = evaluator.evaluate(
        dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    
    print_metrics(results['metrics'], title="NOVEL CLASSES Results")
    
    # Per-class breakdown for novel classes
    print("\nPer Novel Class Performance:")
    print("-" * 50)
    
    per_class = results['metrics'].get('per_class', {})
    for cls in sorted(per_class.keys()):
        data = per_class[cls]
        print(f"  {cls:25s}: {data['success_rate']*100:5.1f}% ({data['num_samples']:3d} samples)")
    
    # Compare with multiple version (if available)
    if args.multiple_results:
        with open(args.multiple_results) as f:
            multiple = json.load(f)
        
        mult_sr = multiple['metrics']['success_rate'] * 100
        novel_sr = results['metrics']['success_rate'] * 100
        
        print("\n" + "="*60)
        print("COMPARISON: Seen Classes vs Novel Classes")
        print("="*60)
        print(f"  Seen classes (multiple):  {mult_sr:.1f}%")
        print(f"  Novel classes:            {novel_sr:.1f}%")
        print(f"  Generalization gap:       {mult_sr - novel_sr:+.1f}%")
        print()
        
        if novel_sr > 80:
            print("  → Strong generalization to novel classes!")
        elif novel_sr > 60:
            print("  → Moderate generalization - some drop expected")
        else:
            print("  → Significant generalization gap")
        
        print("="*60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "eval_novel_classes.json"
    
    save_data = {
        'metrics': results['metrics'],
        'novel_classes': list(test_classes),
        'num_samples': len(results['successes']),
        # Detailed data for ablation analysis
        'sentences': results['sentences'],
        'successes': results['successes'],
        'ious': results['ious'],
        'predictions': results['predictions'],
        'targets': results['targets'] if 'targets' in results else [],
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def analyze_class_splits(args):
    """Analyze which classes are in train vs test for novel-classes split."""
    config = CONFIG.copy()
    config['ocid_vlg_path'] = args.data_dir
    
    print("="*60)
    print("NOVEL-CLASSES SPLIT ANALYSIS")
    print("="*60)
    
    # Load train and test
    from ocid_vlg_loader import OCIDVLGDataset
    
    train_ds = OCIDVLGDataset(args.data_dir, split='train', version='novel-classes')
    test_ds = OCIDVLGDataset(args.data_dir, split='test', version='novel-classes')
    
    # Collect classes
    train_classes = set()
    for sample in train_ds.samples:
        train_classes.add(sample.get('target', 'unknown'))
    
    test_classes = set()
    for sample in test_ds.samples:
        test_classes.add(sample.get('target', 'unknown'))
    
    # Analysis
    print(f"\nTraining classes ({len(train_classes)}):")
    for cls in sorted(train_classes):
        print(f"  - {cls}")
    
    print(f"\nTest classes ({len(test_classes)}):")
    for cls in sorted(test_classes):
        print(f"  - {cls}")
    
    # Overlap check
    overlap = train_classes & test_classes
    if overlap:
        print(f"\n⚠️  WARNING: Overlap found: {overlap}")
    else:
        print(f"\n✓ No overlap - clean novel-classes split")
    
    train_only = train_classes - test_classes
    test_only = test_classes - train_classes
    
    print(f"\nClasses only in train: {len(train_only)}")
    print(f"Classes only in test (NOVEL): {len(test_only)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('--data_dir', type=str, required=True)
    train_p.add_argument('--checkpoint_dir', type=str, default=None)
    train_p.add_argument('--no_wandb', action='store_true')
    
    # Evaluate
    eval_p = subparsers.add_parser('evaluate')
    eval_p.add_argument('--checkpoint', type=str, required=True)
    eval_p.add_argument('--data_dir', type=str, required=True)
    eval_p.add_argument('--max_samples', type=int, default=None)
    eval_p.add_argument('--batch_size', type=int, default=32)
    eval_p.add_argument('--multiple_results', type=str, default=None,
                        help='Path to multiple-split results for comparison')
    eval_p.add_argument('--output_dir', type=str, default='results')
    
    # Analyze
    analyze_p = subparsers.add_parser('analyze')
    analyze_p.add_argument('--data_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_novel_classes(args)
    elif args.command == 'evaluate':
        evaluate_novel_classes(args)
    elif args.command == 'analyze':
        analyze_class_splits(args)
    else:
        parser.print_help()