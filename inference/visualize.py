"""
Visualization utilities for grasp rectangles.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from PIL import Image


def draw_grasp_rectangle(image: np.ndarray,
                         grasp: Dict[str, float],
                         color: tuple = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    Draw oriented rectangle on image.

    Args:
        image: (H, W, 3) RGB image
        grasp: Dict with cx, cy, theta, w, h
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        Image with rectangle drawn
    """
    img = image.copy()

    cx, cy = grasp['cx'], grasp['cy']
    w, h = grasp['w'], grasp['h']
    theta = grasp['theta']

    # Convert to OpenCV RotatedRect format
    # OpenCV uses (center, size, angle) where angle is in degrees
    center = (float(cx), float(cy))
    size = (float(w), float(h))
    angle = float(theta)

    # Get rectangle corners
    rect = cv2.RotatedRect(center, size, angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Draw rectangle
    cv2.drawContours(img, [box], 0, color, thickness)

    # Draw center point
    cv2.circle(img, (int(cx), int(cy)), 3, (255, 0, 0), -1)

    # Draw orientation line (from center along h direction)
    angle_rad = np.deg2rad(theta)
    dx = (h / 2) * np.cos(angle_rad)
    dy = (h / 2) * np.sin(angle_rad)

    end_x = int(cx + dx)
    end_y = int(cy + dy)

    cv2.line(img, (int(cx), int(cy)), (end_x, end_y), (255, 0, 255), thickness)

    return img


def visualize_grasp_sample(image: np.ndarray,
                           pred_grasp: Dict,
                           gt_grasp: Optional[Dict] = None,
                           title: str = "Grasp Prediction") -> np.ndarray:
    """
    Visualize prediction vs ground truth.

    Args:
        image: RGB image
        pred_grasp: Predicted grasp dict
        gt_grasp: Ground truth grasp dict (optional)
        title: Plot title

    Returns:
        Visualization image
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw predictions
    img_vis = image.copy()

    if gt_grasp is not None:
        # Ground truth in blue
        img_vis = draw_grasp_rectangle(img_vis, gt_grasp, color=(0, 0, 255), thickness=2)

    # Prediction in green
    img_vis = draw_grasp_rectangle(img_vis, pred_grasp, color=(0, 255, 0), thickness=2)

    ax.imshow(img_vis)
    ax.set_title(title)
    ax.axis('off')

    if gt_grasp is not None:
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Ground Truth'),
            Patch(facecolor='green', label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Convert to numpy
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img_array


def visualize_batch(images: List[np.ndarray],
                   pred_grasps: List[Dict],
                   gt_grasps: Optional[List[Dict]] = None,
                   save_path: str = "batch_vis.png"):
    """
    Visualize a batch of predictions.

    Args:
        images: List of RGB images
        pred_grasps: List of predicted grasps
        gt_grasps: List of ground truth grasps (optional)
        save_path: Where to save visualization
    """
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n):
        img_vis = images[i].copy()

        if gt_grasps is not None:
            img_vis = draw_grasp_rectangle(img_vis, gt_grasps[i],
                                          color=(0, 0, 255), thickness=2)

        img_vis = draw_grasp_rectangle(img_vis, pred_grasps[i],
                                       color=(0, 255, 0), thickness=2)

        axes[i].imshow(img_vis)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved batch visualization to {save_path}")
