#!/usr/bin/env python3
"""
Full evaluation script.
Runs model + baselines on test set, computes metrics.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from tqdm import tqdm
import pickle

from configs.config import CONFIG
from data.ocid_grasp_loader import OCIDGraspDataset
from data.grasp_quantizer import GraspQuantizer
from inference.predict_grasp import GraspPredictor
from eval.metrics import compute_metrics, compute_per_class_metrics, print_metrics
from eval.baselines import pca_grasp_baseline, minarea_rect_baseline, random_grasp_baseline
from inference.visualize import visualize_batch


def evaluate_model(predictor: GraspPredictor,
                   test_dataset: OCIDGraspDataset,
                   ensemble_size: int = 1,
                   max_samples: int = None) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        predictor: GraspPredictor instance
        test_dataset: Test dataset
        ensemble_size: Ensemble size for prediction
        max_samples: Max samples to evaluate (for speed)

    Returns:
        Dict with predictions, ground_truths, metrics
    """
    predictions = []
    ground_truths = []
    labels = []
    images_vis = []

    n_samples = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))

    print(f"\nEvaluating model on {n_samples} test samples...")

    for idx in tqdm(range(n_samples)):
        sample = test_dataset[idx]

        image = sample['image']
        grasps_gt = sample['grasps']

        if len(grasps_gt) == 0:
            continue

        # Use first grasp as target (OCID-Grasp has multiple grasps per object)
        gt_grasp = grasps_gt[0]

        instruction = sample.get('sentence', '')

        # Predict
        try:
            if ensemble_size == 1:
                pred_grasp = predictor.predict_single(image, instruction)
            else:
                pred_grasp = predictor.predict_ensemble(
                    image, instruction, ensemble_size=ensemble_size
                )

            predictions.append(pred_grasp)
            ground_truths.append(gt_grasp)
            labels.append(sample.get('target', 'unknown'))

            # Save a few for visualization
            if len(images_vis) < 20:
                images_vis.append({
                    'image': image,
                    'pred': pred_grasp,
                    'gt': gt_grasp,
                })

        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            continue

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Per-class metrics
    per_class = compute_per_class_metrics(predictions, ground_truths, labels)

    return {
        'predictions': predictions,
        'ground_truths': ground_truths,
        'labels': labels,
        'metrics': metrics,
        'per_class_metrics': per_class,
        'images_vis': images_vis,
    }


def evaluate_baseline(baseline_name: str,
                     test_dataset: OCIDGraspDataset,
                     max_samples: int = None) -> dict:
    """
    Evaluate a baseline method.

    Args:
        baseline_name: 'pca', 'minarea', or 'random'
        test_dataset: Test dataset
        max_samples: Max samples to evaluate

    Returns:
        Results dict
    """
    predictions = []
    ground_truths = []
    labels = []

    n_samples = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))

    print(f"\nEvaluating {baseline_name} baseline on {n_samples} samples...")

    for idx in tqdm(range(n_samples)):
        sample = test_dataset[idx]

        image = sample['image']
        mask = sample.get('mask', None)
        grasps_gt = sample['grasps']

        if len(grasps_gt) == 0:
            continue

        gt_grasp = grasps_gt[0]

        # Predict with baseline
        try:
            if baseline_name == 'pca':
                if mask is None:
                    continue
                pred_grasp = pca_grasp_baseline(mask)

            elif baseline_name == 'minarea':
                if mask is None:
                    continue
                pred_grasp = minarea_rect_baseline(mask)

            elif baseline_name == 'random':
                pred_grasp = random_grasp_baseline(image, mask)

            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            predictions.append(pred_grasp)
            ground_truths.append(gt_grasp)
            labels.append(sample.get('target', 'unknown'))

        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            continue

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    return {
        'predictions': predictions,
        'ground_truths': ground_truths,
        'labels': labels,
        'metrics': metrics,
    }


def main(args):
    config = CONFIG.copy()

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = OCIDGraspDataset(
        config['ocid_grasp_path'],
        split='test'
    )

    print(f"Test dataset: {len(test_dataset)} images")

    results = {}

    # Evaluate model (if checkpoint provided)
    if args.checkpoint:
        predictor = GraspPredictor(args.checkpoint, config)

        # H=1 (single prediction)
        results['model_h1'] = evaluate_model(
            predictor, test_dataset,
            ensemble_size=1,
            max_samples=args.max_samples
        )

        # H=4 (ensemble)
        if not args.no_ensemble:
            results['model_h4'] = evaluate_model(
                predictor, test_dataset,
                ensemble_size=4,
                max_samples=args.max_samples
            )

    # Evaluate baselines
    if args.baselines:
        for baseline_name in ['pca', 'minarea', 'random']:
            results[f'baseline_{baseline_name}'] = evaluate_baseline(
                baseline_name, test_dataset,
                max_samples=args.max_samples
            )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for method_name, result in results.items():
        print_metrics(result['metrics'], title=method_name.upper())

    # Save results
    output_dir = Path(config['results_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save full results (without images, too large)
    results_to_save = {}
    for key, val in results.items():
        results_to_save[key] = {
            'metrics': val['metrics'],
            'per_class_metrics': val.get('per_class_metrics', {}),
        }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

    # Save visualizations
    if 'model_h1' in results and 'images_vis' in results['model_h1']:
        print("\nGenerating visualizations...")
        vis_samples = results['model_h1']['images_vis'][:10]

        from inference.visualize import visualize_batch

        visualize_batch(
            [s['image'] for s in vis_samples],
            [s['pred'] for s in vis_samples],
            [s['gt'] for s in vis_samples],
            save_path=str(output_dir / 'predictions_vis.png')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--baselines', action='store_true',
                        help='Evaluate baseline methods')
    parser.add_argument('--no_ensemble', action='store_true',
                        help='Skip ensemble evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to evaluate (for quick tests)')

    args = parser.parse_args()

    if not args.checkpoint and not args.baselines:
        print("Error: Must provide --checkpoint and/or --baselines")
        sys.exit(1)

    main(args)
