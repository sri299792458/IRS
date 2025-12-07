#!/usr/bin/env python3
"""
Evaluation script for language-guided grasp detection on OCID-VLG.

Metrics:
- Grasp success rate (IoU > 0.25, angle diff < 30°) against ANY valid grasp
- Per-class breakdown
- Language grounding accuracy (optional: mask IoU)
"""
import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

# Local imports
from config_vlg import CONFIG, get_config
from torch_dataset_vlg import GraspVLGEvalDataset
from grasp_quantizer import GraspQuantizer
from chat_formatter_vlg import format_inference_messages

# Metrics (from your existing codebase)
# from eval.metrics import rotated_iou, angle_difference, compute_metrics


def rotated_iou(pred, gt, angle_tolerance=30.0):
    """
    Compute IoU for oriented rectangles.
    Returns 0 if angle difference > tolerance.
    """
    import cv2
    from shapely.geometry import Polygon
    
    # Check angle tolerance
    angle_diff = abs(angle_difference(pred['theta'], gt['theta']))
    if angle_diff > angle_tolerance:
        return 0.0
    
    try:
        # Convert to polygons using OpenCV RotatedRect
        def grasp_to_polygon(g):
            center = (float(g['cx']), float(g['cy']))
            size = (float(g['w']), float(g['h']))
            angle = float(g['theta'])
            rect = ((center[0], center[1]), (size[0], size[1]), angle)
            box = cv2.boxPoints(rect)
            return Polygon(box)
        
        poly_pred = grasp_to_polygon(pred)
        poly_gt = grasp_to_polygon(gt)
        
        if not poly_pred.is_valid or not poly_gt.is_valid:
            return 0.0
        
        intersection = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area
        
        if union < 1e-6:
            return 0.0
        
        return float(intersection / union)
    
    except Exception as e:
        return 0.0


def angle_difference(a1, a2):
    """Compute angle difference handling parallel-jaw symmetry."""
    a1 = ((a1 + 90) % 180) - 90
    a2 = ((a2 + 90) % 180) - 90
    diff = abs(a1 - a2)
    if diff > 90:
        diff = 180 - diff
    return diff


def check_grasp_success(pred_grasp, gt_grasps, iou_threshold=0.25, angle_tolerance=30.0):
    """
    Check if predicted grasp matches ANY ground truth grasp.
    
    This is the key difference from single-object evaluation:
    success if pred matches ANY valid grasp for the target.
    """
    for gt in gt_grasps:
        iou = rotated_iou(pred_grasp, gt, angle_tolerance=angle_tolerance)
        if iou >= iou_threshold:
            return True, iou
    return False, 0.0


class GraspVLGEvaluator:
    """Evaluator for language-guided grasp detection."""
    
    def __init__(self, checkpoint_path, config):
        self.config = config
        
        # Quantizer
        self.quantizer = GraspQuantizer(
            img_width=config['image_size'][1],
            img_height=config['image_size'][0],
            x_bins=config['x_bins'],
            y_bins=config['y_bins'],
            theta_bins=config['theta_bins'],
            w_bins=config['w_bins'],
            h_bins=config['h_bins'],
        )
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        from model.qwen3_grasp_model import load_trained_model
        self.model, self.processor = load_trained_model(checkpoint_path, config)
        self.model.eval()
        
        # Constrained decoding
        from model.constrained_decoding import DigitsOnlyLogitsProcessor
        self.logits_processor = [DigitsOnlyLogitsProcessor(self.processor.tokenizer)]
    
    @torch.no_grad()
    def predict(self, image, sentence):
        """
        Predict grasp for given image and referring expression.
        
        Args:
            image: numpy array (H, W, 3)
            sentence: str, referring expression
            
        Returns:
            grasp dict {cx, cy, theta, w, h}
        """
        # Format messages
        messages = format_inference_messages(image, instruction=sentence)
        
        # Process
        text_prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        inputs = self.processor(
            text=[text_prompt],
            images=[pil_image],
            return_tensors='pt'
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.get('generation_max_tokens', 20),
            temperature=self.config.get('generation_temperature', 0.1),
            do_sample=self.config.get('generation_temperature', 0.1) > 0,
            logits_processor=self.logits_processor if self.config.get('use_constrained_decoding', True) else None,
        )
        
        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse output
        from model.constrained_decoding import parse_grasp_output
        bins = parse_grasp_output(generated_text)
        
        if bins is None:
            # Return center of image as fallback
            return {
                'cx': self.config['image_size'][1] / 2,
                'cy': self.config['image_size'][0] / 2,
                'theta': 0,
                'w': 50,
                'h': 100,
            }
        
        # Decode bins to continuous
        grasp = self.quantizer.decode(bins)
        return grasp
    
    def evaluate(self, dataset, max_samples=None):
        """
        Evaluate on dataset.
        
        Args:
            dataset: GraspVLGEvalDataset
            max_samples: Max samples to evaluate (None = all)
            
        Returns:
            dict with metrics
        """
        results = {
            'predictions': [],
            'ground_truths': [],
            'sentences': [],
            'targets': [],
            'successes': [],
            'ious': [],
        }
        
        n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        print(f"\nEvaluating on {n_samples} samples...")
        
        for idx in tqdm(range(n_samples)):
            sample = dataset[idx]
            
            image = sample['image']
            sentence = sample['sentence']
            gt_grasps = sample['grasps']
            target = sample['target']
            
            # Skip if no ground truth
            if len(gt_grasps) == 0:
                continue
            
            # Predict
            try:
                pred_grasp = self.predict(image, sentence)
            except Exception as e:
                print(f"\nError on sample {idx}: {e}")
                continue
            
            # Check success against ANY valid grasp
            success, best_iou = check_grasp_success(
                pred_grasp, gt_grasps,
                iou_threshold=self.config.get('iou_threshold', 0.25),
                angle_tolerance=self.config.get('angle_tolerance_deg', 30)
            )
            
            results['predictions'].append(pred_grasp)
            results['ground_truths'].append(gt_grasps)
            results['sentences'].append(sentence)
            results['targets'].append(target)
            results['successes'].append(success)
            results['ious'].append(best_iou)
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        results['metrics'] = metrics
        
        return results
    
    def _compute_metrics(self, results):
        """Compute aggregate metrics."""
        successes = results['successes']
        ious = results['ious']
        targets = results['targets']
        
        # Overall metrics
        metrics = {
            'num_samples': len(successes),
            'success_rate': np.mean(successes) if successes else 0.0,
            'mean_iou': np.mean(ious) if ious else 0.0,
            'median_iou': np.median(ious) if ious else 0.0,
        }
        
        # Per-class metrics
        per_class = {}
        for target, success, iou in zip(targets, successes, ious):
            if target not in per_class:
                per_class[target] = {'successes': [], 'ious': []}
            per_class[target]['successes'].append(success)
            per_class[target]['ious'].append(iou)
        
        metrics['per_class'] = {}
        for target, data in per_class.items():
            metrics['per_class'][target] = {
                'num_samples': len(data['successes']),
                'success_rate': np.mean(data['successes']),
                'mean_iou': np.mean(data['ious']),
            }
        
        return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print(f"\nOverall:")
    print(f"  Samples:      {metrics['num_samples']}")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Mean IoU:     {metrics['mean_iou']:.3f}")
    print(f"  Median IoU:   {metrics['median_iou']:.3f}")
    
    if 'per_class' in metrics and len(metrics['per_class']) > 0:
        print(f"\nPer-Class Breakdown:")
        
        # Sort by success rate
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for target, data in sorted_classes[:15]:  # Top 15
            print(f"  {target:25s}: {data['success_rate']*100:5.1f}% ({data['num_samples']:3d} samples)")
        
        if len(sorted_classes) > 15:
            print(f"  ... and {len(sorted_classes) - 15} more classes")
    
    print("="*60)


def main(args):
    # Get config
    config = get_config(args.config_variant) if args.config_variant else CONFIG.copy()
    
    if args.data_dir:
        config['ocid_vlg_path'] = args.data_dir
    if args.version:
        config['dataset_version'] = args.version

    print("="*60)
    print("OCID-VLG Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {config['ocid_vlg_path']}")
    print(f"Version: {config['dataset_version']}")
    print(f"Split: {args.split}")
    print("="*60)

    # Load dataset
    dataset = GraspVLGEvalDataset(
        config['ocid_vlg_path'],
        split=args.split,
        version=config['dataset_version'],
        config=config,
    )
    
    print(f"Loaded {len(dataset)} samples")

    # Create evaluator
    evaluator = GraspVLGEvaluator(args.checkpoint, config)

    # Evaluate
    results = evaluator.evaluate(dataset, max_samples=args.max_samples)

    # Print results
    print_metrics(results['metrics'], title=f"OCID-VLG {config['dataset_version']}/{args.split}")

    # Save results
    output_dir = Path(config['results_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"eval_{config['dataset_version']}_{args.split}.json"
    
    # Convert to serializable format
    save_results = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'metrics': results['metrics'],
        'num_samples': len(results['predictions']),
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

    # Optional: visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        from inference.visualize import visualize_batch, draw_grasp_rectangle
        
        # Visualize first N samples
        n_vis = min(10, len(results['predictions']))
        
        vis_samples = []
        for i in range(n_vis):
            sample = dataset[i]
            pred = results['predictions'][i]
            success = results['successes'][i]
            
            vis_samples.append({
                'image': sample['image'],
                'pred': pred,
                'gt': sample['grasps'][0] if sample['grasps'] else None,
                'sentence': sample['sentence'],
                'success': success,
            })
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, s in enumerate(vis_samples):
            img = s['image'].copy()
            
            # Draw GT in blue
            if s['gt']:
                img = draw_grasp_rectangle(img, s['gt'], color=(0, 0, 255), thickness=2)
            
            # Draw pred in green (or red if failed)
            pred_color = (0, 255, 0) if s['success'] else (255, 0, 0)
            img = draw_grasp_rectangle(img, s['pred'], color=pred_color, thickness=2)
            
            axes[i].imshow(img)
            status = "✓" if s['success'] else "✗"
            axes[i].set_title(f"{status} {s['sentence'][:30]}...", fontsize=8)
            axes[i].axis('off')
        
        plt.tight_layout()
        vis_path = output_dir / f"vis_{config['dataset_version']}_{args.split}.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {vis_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OCID-VLG dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--version', type=str, default='multiple',
                        choices=['multiple', 'unique', 'novel-instances', 'novel-classes'],
                        help='Dataset version')
    parser.add_argument('--config_variant', type=str, default=None,
                        help='Config variant')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to evaluate (for quick tests)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')

    args = parser.parse_args()
    main(args)
