#!/usr/bin/env python3
"""
Failure Analysis Script for OCID-VLG Results.

Analyzes and visualizes failure cases to understand model limitations.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add script directory (OCID-VLG)
sys.path.insert(0, str(Path(__file__).parent))
# Add parent directory (IRS) for 'model' package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from config_vlg import CONFIG
from torch_dataset_vlg import GraspVLGEvalDataset
from grasp_quantizer import GraspQuantizer
from evaluate_vlg import GraspVLGEvaluator, check_grasp_success, angle_difference


def draw_grasp_rectangle(img, grasp, color=(0, 255, 0), thickness=2):
    """Draw oriented rectangle for grasp."""
    cx, cy = grasp['cx'], grasp['cy']
    w, h = grasp['w'], grasp['h']
    theta = grasp['theta']
    
    # Get rotated rectangle points
    rect = ((cx, cy), (w, h), theta)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    cv2.drawContours(img, [box], 0, color, thickness)
    
    # Draw center point
    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)
    
    # Draw orientation line
    angle_rad = np.radians(theta)
    line_len = max(w, h) / 2
    x2 = int(cx + line_len * np.cos(angle_rad))
    y2 = int(cy + line_len * np.sin(angle_rad))
    cv2.line(img, (int(cx), int(cy)), (x2, y2), color, thickness)
    
    return img


def analyze_class_failures(results_json_path: str, 
                           checkpoint_path: str,
                           data_dir: str,
                           target_classes: list,
                           output_dir: str,
                           max_per_class: int = 20):
    """
    Detailed failure analysis for specific classes.
    
    Args:
        results_json_path: Path to eval results JSON
        checkpoint_path: Model checkpoint
        data_dir: OCID-VLG data directory
        target_classes: List of class names to analyze
        output_dir: Where to save visualizations
        max_per_class: Max failures to visualize per class
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = {}
    if results_json_path and os.path.exists(results_json_path):
        print(f"Loading results from {results_json_path}...")
        with open(results_json_path) as f:
            results = json.load(f)
    else:
        print("No results JSON provided. Analyzing from scratch...")
        # Try to find a default if not provided? No, just proceed.
    
    config = CONFIG.copy()
    config['ocid_vlg_path'] = data_dir
    
    # Load dataset
    print("Loading dataset...")
    dataset = GraspVLGEvalDataset(
        data_dir,
        split='test',
        version='multiple',
        config=config,
    )
    
    # Load model
    print("Loading model...")
    evaluator = GraspVLGEvaluator(checkpoint_path, config)
    
    # Analyze each target class
    class_analysis = {}
    
    for target_class in target_classes:
        print(f"\n{'='*60}")
        print(f"Analyzing: {target_class}")
        print(f"{'='*60}")
        
        class_dir = output_dir / target_class
        class_dir.mkdir(exist_ok=True)
        
        # Find samples of this class
        class_samples = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample['target'] == target_class:
                class_samples.append((idx, sample))
        
        print(f"Found {len(class_samples)} samples")
        
        # Evaluate and categorize
        successes = []
        failures = []
        
        for idx, sample in tqdm(class_samples, desc=f"Evaluating {target_class}"):
            # Predict
            pred_grasp = evaluator.predict_batch([sample['image']], [sample['sentence']])[0]
            
            # Check success
            success, iou = check_grasp_success(
                pred_grasp, sample['grasps'],
                iou_threshold=config['iou_threshold'],
                angle_tolerance=config['angle_tolerance_deg']
            )
            
            result = {
                'idx': idx,
                'sample': sample,
                'pred': pred_grasp,
                'success': success,
                'iou': iou,
            }
            
            if success:
                successes.append(result)
            else:
                failures.append(result)
        
        # Analysis stats
        total = len(class_samples)
        n_success = len(successes)
        n_fail = len(failures)
        
        class_analysis[target_class] = {
            'total': total,
            'success': n_success,
            'fail': n_fail,
            'success_rate': n_success / total if total > 0 else 0,
        }
        
        print(f"Success: {n_success}/{total} ({100*n_success/total:.1f}%)")
        print(f"Failures: {n_fail}")
        
        # Visualize failures
        print(f"\nVisualizing failures...")
        
        for i, result in enumerate(failures[:max_per_class]):
            sample = result['sample']
            pred = result['pred']
            iou = result['iou']
            
            img = sample['image'].copy()
            
            # Draw all GT grasps in blue
            for gt in sample['grasps']:
                img = draw_grasp_rectangle(img, gt, color=(255, 0, 0), thickness=2)
            
            # Draw prediction in red (failure)
            img = draw_grasp_rectangle(img, pred, color=(0, 0, 255), thickness=3)
            
            # Add text annotations
            cv2.putText(img, f"FAILURE - IoU: {iou:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, f"Target: {sample['target']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Truncate sentence if too long
            sentence = sample['sentence'][:80] + "..." if len(sample['sentence']) > 80 else sample['sentence']
            cv2.putText(img, f"Query: {sentence}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Prediction details
            pred_str = f"Pred: ({pred['cx']:.0f}, {pred['cy']:.0f}, θ={pred['theta']:.0f}°)"
            cv2.putText(img, pred_str, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # GT details (first one)
            if sample['grasps']:
                gt = sample['grasps'][0]
                gt_str = f"GT[0]: ({gt['cx']:.0f}, {gt['cy']:.0f}, θ={gt['theta']:.0f}°)"
                cv2.putText(img, gt_str, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Legend
            cv2.putText(img, "Blue=GT, Red=Pred", (10, img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save
            save_path = class_dir / f"fail_{i:03d}_iou{iou:.2f}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Also save a few successes for comparison
        success_dir = class_dir / "successes"
        success_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(successes[:5]):
            sample = result['sample']
            pred = result['pred']
            iou = result['iou']
            
            img = sample['image'].copy()
            
            # Draw first GT in blue
            if sample['grasps']:
                img = draw_grasp_rectangle(img, sample['grasps'][0], color=(255, 0, 0), thickness=2)
            
            # Draw prediction in green (success)
            img = draw_grasp_rectangle(img, pred, color=(0, 255, 0), thickness=3)
            
            cv2.putText(img, f"SUCCESS - IoU: {iou:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            save_path = success_dir / f"success_{i:03d}_iou{iou:.2f}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Failure pattern analysis
        print(f"\nFailure Pattern Analysis for {target_class}:")
        
        if failures:
            # Analyze angle errors
            angle_errors = []
            position_errors = []
            size_errors = []
            
            for result in failures:
                pred = result['pred']
                sample = result['sample']
                
                # Find closest GT
                min_angle_err = 180
                min_pos_err = float('inf')
                min_size_err = float('inf')
                
                for gt in sample['grasps']:
                    angle_err = abs(angle_difference(pred['theta'], gt['theta']))
                    pos_err = np.sqrt((pred['cx'] - gt['cx'])**2 + (pred['cy'] - gt['cy'])**2)
                    size_err = abs(pred['w'] - gt['w']) + abs(pred['h'] - gt['h'])
                    
                    if angle_err < min_angle_err:
                        min_angle_err = angle_err
                    if pos_err < min_pos_err:
                        min_pos_err = pos_err
                    if size_err < min_size_err:
                        min_size_err = size_err
                
                angle_errors.append(min_angle_err)
                position_errors.append(min_pos_err)
                size_errors.append(min_size_err)
            
            print(f"  Angle Error:    mean={np.mean(angle_errors):.1f}°, median={np.median(angle_errors):.1f}°")
            print(f"  Position Error: mean={np.mean(position_errors):.1f}px, median={np.median(position_errors):.1f}px")
            print(f"  Size Error:     mean={np.mean(size_errors):.1f}px, median={np.median(size_errors):.1f}px")
            
            # Categorize failure types
            angle_failures = sum(1 for e in angle_errors if e > 30)
            position_failures = sum(1 for e in position_errors if e > 50)
            
            print(f"\n  Primary failure causes:")
            print(f"    Angle errors (>30°):     {angle_failures}/{len(failures)} ({100*angle_failures/len(failures):.1f}%)")
            print(f"    Position errors (>50px): {position_failures}/{len(failures)} ({100*position_failures/len(failures):.1f}%)")
            
            class_analysis[target_class]['angle_errors'] = {
                'mean': float(np.mean(angle_errors)),
                'median': float(np.median(angle_errors)),
            }
            class_analysis[target_class]['position_errors'] = {
                'mean': float(np.mean(position_errors)),
                'median': float(np.median(position_errors)),
            }
    
    # Save analysis summary
    summary_path = output_dir / "failure_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(class_analysis, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    return class_analysis


def create_failure_summary_figure(analysis: dict, output_path: str):
    """Create a summary figure comparing failure patterns across classes."""
    
    classes = list(analysis.keys())
    success_rates = [analysis[c]['success_rate'] * 100 for c in classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success rate bar chart
    ax1 = axes[0, 0]
    colors = ['green' if sr > 80 else 'orange' if sr > 60 else 'red' for sr in success_rates]
    bars = ax1.bar(classes, success_rates, color=colors)
    ax1.axhline(y=92.6, color='blue', linestyle='--', label='Overall (92.6%)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate by Class')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Sample count
    ax2 = axes[0, 1]
    totals = [analysis[c]['total'] for c in classes]
    ax2.bar(classes, totals, color='steelblue')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count by Class')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Angle error distribution (if available)
    ax3 = axes[1, 0]
    angle_means = []
    for c in classes:
        if 'angle_errors' in analysis[c]:
            angle_means.append(analysis[c]['angle_errors']['mean'])
        else:
            angle_means.append(0)
    
    ax3.bar(classes, angle_means, color='coral')
    ax3.axhline(y=30, color='red', linestyle='--', label='Threshold (30°)')
    ax3.set_ylabel('Mean Angle Error (°)')
    ax3.set_title('Angle Error by Class (Failures Only)')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Position error distribution
    ax4 = axes[1, 1]
    pos_means = []
    for c in classes:
        if 'position_errors' in analysis[c]:
            pos_means.append(analysis[c]['position_errors']['mean'])
        else:
            pos_means.append(0)
    
    ax4.bar(classes, pos_means, color='mediumpurple')
    ax4.axhline(y=50, color='red', linestyle='--', label='50px threshold')
    ax4.set_ylabel('Mean Position Error (px)')
    ax4.set_title('Position Error by Class (Failures Only)')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OCID-VLG dataset')
    parser.add_argument('--results_json', type=str, default=None,
                        help='Path to evaluation results JSON (optional)')
    parser.add_argument('--output_dir', type=str, default='results/failure_analysis',
                        help='Output directory for visualizations')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['bell_pepper_1', 'bowl_1', 'binder_1', 
                                 'coffee_mug_1', 'coffee_mug_2', 'hand_towel_1'],
                        help='Classes to analyze')
    parser.add_argument('--max_per_class', type=int, default=20,
                        help='Max failure visualizations per class')
    
    args = parser.parse_args()
    
    analysis = analyze_class_failures(
        results_json_path=args.results_json,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        target_classes=args.classes,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
    )
    
    # Create summary figure
    create_failure_summary_figure(
        analysis,
        output_path=str(Path(args.output_dir) / "failure_summary.png")
    )
