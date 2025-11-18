"""
Baseline methods for grasp detection.
- SAM + min-area rectangle
- PCA-based orientation
"""
import numpy as np
import cv2
from typing import Dict, Optional
from sklearn.decomposition import PCA


def pca_grasp_baseline(mask: np.ndarray) -> Dict[str, float]:
    """
    Baseline: fit oriented rectangle using PCA on mask.

    Args:
        mask: (H, W) binary mask of object

    Returns:
        Grasp dict with cx, cy, theta, w, h
    """
    # Get points from mask
    points = np.argwhere(mask > 0)

    if len(points) < 10:
        # Degenerate case: return dummy grasp
        h, w = mask.shape
        return {
            'cx': w / 2,
            'cy': h / 2,
            'theta': 0.0,
            'w': 50.0,
            'h': 100.0,
        }

    # points are (y, x), convert to (x, y)
    points_xy = points[:, [1, 0]]

    # Compute center
    cx, cy = points_xy.mean(axis=0)

    # PCA for orientation
    pca = PCA(n_components=2)
    pca.fit(points_xy)

    # Principal axis
    principal_axis = pca.components_[0]  # (x, y) direction

    # Angle from x-axis
    theta_rad = np.arctan2(principal_axis[1], principal_axis[0])
    theta_deg = np.rad2deg(theta_rad)

    # Normalize to [-90, 90)
    theta_deg = ((theta_deg + 90) % 180) - 90

    # Size: use explained variance
    # Project points onto principal axes
    projected = pca.transform(points_xy)

    # Width and height from range along each axis
    w = (projected[:, 1].max() - projected[:, 1].min())  # Short axis (gripper opening)
    h = (projected[:, 0].max() - projected[:, 0].min())  # Long axis

    # Ensure w < h (gripper convention)
    if w > h:
        w, h = h, w
        theta_deg = (theta_deg + 90) % 180 - 90

    return {
        'cx': float(cx),
        'cy': float(cy),
        'theta': float(theta_deg),
        'w': float(w),
        'h': float(h),
    }


def minarea_rect_baseline(mask: np.ndarray) -> Dict[str, float]:
    """
    Baseline: OpenCV minAreaRect on mask contour.

    Args:
        mask: (H, W) binary mask

    Returns:
        Grasp dict
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        # Degenerate
        h, w = mask.shape
        return {
            'cx': w / 2,
            'cy': h / 2,
            'theta': 0.0,
            'w': 50.0,
            'h': 100.0,
        }

    # Use largest contour
    contour = max(contours, key=cv2.contourArea)

    # Min area rectangle
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # OpenCV angle convention: [0, 90)
    # Convert to our [-90, 90) convention
    # OpenCV: angle is from horizontal to the first side
    # We want: angle from horizontal to long axis

    # Ensure w < h (gripper convention)
    if w > h:
        w, h = h, w
        angle = (angle + 90) % 180

    # Normalize to [-90, 90)
    theta_deg = ((angle + 90) % 180) - 90

    return {
        'cx': float(cx),
        'cy': float(cy),
        'theta': float(theta_deg),
        'w': float(w),
        'h': float(h),
    }


def random_grasp_baseline(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Baseline: random grasp within image/mask bounds.

    Args:
        image: (H, W, 3) RGB image
        mask: Optional (H, W) mask to constrain grasp

    Returns:
        Random grasp dict
    """
    h, w = image.shape[:2]

    if mask is not None and mask.sum() > 0:
        # Sample from mask
        points = np.argwhere(mask > 0)
        idx = np.random.randint(len(points))
        cy, cx = points[idx]
    else:
        # Sample from image
        cx = np.random.uniform(w * 0.2, w * 0.8)
        cy = np.random.uniform(h * 0.2, h * 0.8)

    # Random angle
    theta = np.random.uniform(-90, 90)

    # Random size (reasonable defaults)
    grasp_w = np.random.uniform(30, 100)
    grasp_h = np.random.uniform(50, 150)

    return {
        'cx': float(cx),
        'cy': float(cy),
        'theta': float(theta),
        'w': float(grasp_w),
        'h': float(grasp_h),
    }
