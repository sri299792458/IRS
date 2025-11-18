#!/usr/bin/env python3
"""
Quick dataset inspection script.
Shows dataset statistics and sample visualizations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from configs.config import CONFIG
from data.ocid_grasp_loader import OCIDGraspDataset


def inspect_dataset(split='train'):
    """Inspect dataset and print statistics."""
    print("="*60)
    print(f"Dataset Inspection: {split} split")
    print("="*60)

    # Load dataset
    dataset = OCIDGraspDataset(CONFIG['ocid_grasp_path'], split=split)

    print(f"\nTotal images: {len(dataset)}")

    # Collect statistics
    n_grasps_per_image = []
    all_classes = []
    grasp_angles = []
    grasp_widths = []
    grasp_heights = []

    for i in range(len(dataset)):
        sample = dataset[i]

        n_grasps_per_image.append(len(sample['grasps']))
        all_classes.append(sample['target'])

        for grasp in sample['grasps']:
            grasp_angles.append(grasp['theta'])
            grasp_widths.append(grasp['w'])
            grasp_heights.append(grasp['h'])

    # Print statistics
    print(f"\nGrasps per image:")
    print(f"  Mean: {np.mean(n_grasps_per_image):.1f}")
    print(f"  Median: {np.median(n_grasps_per_image):.1f}")
    print(f"  Total grasps: {sum(n_grasps_per_image)}")

    print(f"\nGrasp angles (degrees):")
    print(f"  Mean: {np.mean(grasp_angles):.1f}°")
    print(f"  Std: {np.std(grasp_angles):.1f}°")
    print(f"  Range: [{np.min(grasp_angles):.1f}, {np.max(grasp_angles):.1f}]")

    print(f"\nGrasp widths (pixels):")
    print(f"  Mean: {np.mean(grasp_widths):.1f}")
    print(f"  Std: {np.std(grasp_widths):.1f}")

    print(f"\nGrasp heights (pixels):")
    print(f"  Mean: {np.mean(grasp_heights):.1f}")
    print(f"  Std: {np.std(grasp_heights):.1f}")

    print(f"\nObject classes:")
    class_counts = Counter(all_classes)
    for class_name, count in class_counts.most_common(10):
        print(f"  {class_name:20s}: {count:4d} images")

    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes")

    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Grasps per image
    axes[0, 0].hist(n_grasps_per_image, bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Grasps per image')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Grasps per Image Distribution')

    # Angles
    axes[0, 1].hist(grasp_angles, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Angle (degrees)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Grasp Angle Distribution')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Widths
    axes[1, 0].hist(grasp_widths, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Grasp Width Distribution')

    # Heights
    axes[1, 1].hist(grasp_heights, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Height (pixels)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Grasp Height Distribution')

    plt.tight_layout()
    plt.savefig(f'dataset_stats_{split}.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to dataset_stats_{split}.png")

    # Show sample images
    print("\nGenerating sample visualizations...")
    from inference.visualize import draw_grasp_rectangle

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        sample = dataset[i]
        img = sample['image']

        # Draw all grasps for this image
        img_vis = img.copy()
        for grasp in sample['grasps'][:3]:  # Max 3 grasps per image
            img_vis = draw_grasp_rectangle(img_vis, grasp, color=(0, 255, 0))

        axes[i].imshow(img_vis)
        axes[i].set_title(f"{sample['target']}\n{len(sample['grasps'])} grasps")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'dataset_samples_{split}.png', dpi=150, bbox_inches='tight')
    print(f"Sample images saved to dataset_samples_{split}.png")

    print("\n" + "="*60)
    print("Inspection complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to inspect')

    args = parser.parse_args()

    inspect_dataset(args.split)
