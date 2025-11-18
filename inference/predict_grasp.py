#!/usr/bin/env python3
"""
Inference script with ensemble support.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import cv2

from configs.config import CONFIG
from data.grasp_quantizer import GraspQuantizer
from model.qwen3_grasp_model import load_trained_model, GraspModelWrapper
from model.constrained_decoding import parse_grasp_output
from data.chat_formatter import format_inference_messages


class GraspPredictor:
    """Ensemble grasp predictor with TTA."""

    def __init__(self, checkpoint_path: str, config: Optional[Dict] = None):
        """
        Args:
            checkpoint_path: Path to trained LoRA checkpoint
            config: Config dict
        """
        self.config = config or CONFIG
        self.checkpoint_path = checkpoint_path

        # Initialize quantizer
        self.quantizer = GraspQuantizer(
            img_width=self.config['image_size'][1],
            img_height=self.config['image_size'][0],
            x_bins=self.config['x_bins'],
            y_bins=self.config['y_bins'],
            theta_bins=self.config['theta_bins'],
            w_bins=self.config['w_bins'],
            h_bins=self.config['h_bins'],
        )

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model, self.processor = load_trained_model(checkpoint_path, self.config)
        self.model.eval()

        self.wrapper = GraspModelWrapper(
            self.model, self.processor, self.quantizer, self.config
        )

    def predict_single(self, image, instruction: str = "",
                       temperature: float = 0.2) -> Dict:
        """
        Single prediction (H=1).

        Args:
            image: PIL Image or numpy array
            instruction: Optional language instruction
            temperature: Sampling temperature

        Returns:
            Grasp dict
        """
        grasp, text = self.wrapper.predict_grasp(
            image,
            instruction=instruction,
            use_constrained=True,
            temperature=temperature
        )

        return grasp

    def predict_ensemble(self, image, instruction: str = "",
                        ensemble_size: int = 4,
                        temperature: float = 0.2) -> Dict:
        """
        Ensemble prediction with slight augmentations (H=4).

        Args:
            image: PIL Image or numpy array
            instruction: Optional language instruction
            ensemble_size: Number of predictions to ensemble
            temperature: Sampling temperature

        Returns:
            Averaged grasp dict
        """
        predictions = []

        for i in range(ensemble_size):
            # Slight augmentation for diversity
            aug_image = self._slight_augment(image, seed=i)

            grasp = self.predict_single(
                aug_image,
                instruction=instruction,
                temperature=temperature
            )

            predictions.append(grasp)

        # Average in continuous space
        avg_grasp = self._average_grasps(predictions)

        return avg_grasp

    def _slight_augment(self, image, seed: int = 0):
        """
        Very slight augmentation for ensemble diversity.
        Just small crops/shifts, no heavy rotation.
        """
        np.random.seed(seed)

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        h, w = img_array.shape[:2]

        # Small random crop (98% of image)
        crop_h = int(h * 0.99)
        crop_w = int(w * 0.99)

        y_start = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        x_start = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

        cropped = img_array[y_start:y_start+crop_h, x_start:x_start+crop_w]

        # Resize back to original size
        resized = cv2.resize(cropped, (w, h))

        return Image.fromarray(resized)

    def _average_grasps(self, grasps: List[Dict]) -> Dict:
        """
        Average multiple grasp predictions.
        Handle circular mean for angles.
        """
        if len(grasps) == 0:
            raise ValueError("No grasps to average")

        if len(grasps) == 1:
            return grasps[0]

        # Average positions
        cx = np.mean([g['cx'] for g in grasps])
        cy = np.mean([g['cy'] for g in grasps])
        w = np.mean([g['w'] for g in grasps])
        h = np.mean([g['h'] for g in grasps])

        # Circular mean for angle
        angles = [g['theta'] for g in grasps]
        theta = self._circular_mean_deg(angles)

        return {
            'cx': float(cx),
            'cy': float(cy),
            'theta': float(theta),
            'w': float(w),
            'h': float(h),
        }

    def _circular_mean_deg(self, angles: List[float]) -> float:
        """Circular mean for angles in degrees."""
        # Convert to radians
        angles_rad = np.deg2rad(angles)

        # Compute mean of unit vectors
        sin_mean = np.mean(np.sin(angles_rad))
        cos_mean = np.mean(np.cos(angles_rad))

        # Convert back
        mean_rad = np.arctan2(sin_mean, cos_mean)
        mean_deg = np.rad2deg(mean_rad)

        # Normalize to [-90, 90)
        mean_deg = ((mean_deg + 90) % 180) - 90

        return mean_deg


def main():
    """Quick test of inference."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LoRA checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Ensemble size (1 = no ensemble)')
    parser.add_argument('--instruction', type=str, default='',
                        help='Language instruction')
    parser.add_argument('--output', type=str, default='prediction.png',
                        help='Output visualization path')

    args = parser.parse_args()

    # Load predictor
    predictor = GraspPredictor(args.checkpoint)

    # Load image
    image = Image.open(args.image).convert('RGB')

    # Predict
    print(f"\nPredicting grasp (ensemble={args.ensemble})...")
    if args.ensemble == 1:
        grasp = predictor.predict_single(image, args.instruction)
    else:
        grasp = predictor.predict_ensemble(
            image, args.instruction, ensemble_size=args.ensemble
        )

    print(f"\nPredicted grasp:")
    print(f"  Center: ({grasp['cx']:.1f}, {grasp['cy']:.1f})")
    print(f"  Angle: {grasp['theta']:.1f}°")
    print(f"  Size: {grasp['w']:.1f} x {grasp['h']:.1f}")

    # Visualize
    from inference.visualize import draw_grasp_rectangle
    img_vis = draw_grasp_rectangle(np.array(image), grasp)

    Image.fromarray(img_vis).save(args.output)
    print(f"\nVisualization saved to {args.output}")


if __name__ == '__main__':
    main()
