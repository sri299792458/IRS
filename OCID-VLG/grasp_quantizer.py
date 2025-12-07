"""
Grasp quantization: continuous rectangles <-> discrete bins.
Simple and hackable for fast iteration.
"""
import numpy as np
from typing import Dict, Tuple


class GraspQuantizer:
    """Convert between continuous grasp params and discrete bins."""

    def __init__(self, img_width=640, img_height=480,
                 x_bins=1000, y_bins=1000, theta_bins=180,
                 w_bins=1000, h_bins=1000):
        self.img_width = img_width
        self.img_height = img_height
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.theta_bins = theta_bins
        self.w_bins = w_bins
        self.h_bins = h_bins

    def encode(self, grasp: Dict[str, float]) -> Dict[str, int]:
        """
        Continuous grasp -> discrete bins.

        Args:
            grasp: {'cx': float, 'cy': float, 'theta': float, 'w': float, 'h': float}
                   cx, cy in pixels, theta in degrees [-90, 90), w, h in pixels

        Returns:
            bins: {'x': int, 'y': int, 'theta': int, 'w': int, 'h': int}
        """
        # Normalize coords to [0, 1] then scale to bins
        x_norm = np.clip(grasp['cx'] / self.img_width, 0, 1)
        y_norm = np.clip(grasp['cy'] / self.img_height, 0, 1)

        x_bin = int(x_norm * (self.x_bins - 1))
        y_bin = int(y_norm * (self.y_bins - 1))

        # Theta: [-90, 90) -> [0, 180)
        theta_normalized = (grasp['theta'] + 90) / 180  # [0, 1)
        theta_bin = int(theta_normalized * self.theta_bins)
        theta_bin = np.clip(theta_bin, 0, self.theta_bins - 1)

        # Width and height: normalize by image width for scale invariance
        w_norm = np.clip(grasp['w'] / self.img_width, 0, 1)
        h_norm = np.clip(grasp['h'] / self.img_width, 0, 1)

        w_bin = int(w_norm * (self.w_bins - 1))
        h_bin = int(h_norm * (self.h_bins - 1))

        # Clamp to valid range
        return {
            'x': np.clip(x_bin, 0, self.x_bins - 1),
            'y': np.clip(y_bin, 0, self.y_bins - 1),
            'theta': theta_bin,
            'w': np.clip(w_bin, 0, self.w_bins - 1),
            'h': np.clip(h_bin, 0, self.h_bins - 1),
        }

    def decode(self, bins: Dict[str, int]) -> Dict[str, float]:
        """
        Discrete bins -> continuous grasp.

        Args:
            bins: {'x': int, 'y': int, 'theta': int, 'w': int, 'h': int}

        Returns:
            grasp: {'cx': float, 'cy': float, 'theta': float, 'w': float, 'h': float}
        """
        # Denormalize coordinates
        x_norm = (bins['x'] + 0.5) / self.x_bins  # +0.5 for bin center
        y_norm = (bins['y'] + 0.5) / self.y_bins

        cx = x_norm * self.img_width
        cy = y_norm * self.img_height

        # Theta: [0, 180) -> [-90, 90)
        theta_norm = (bins['theta'] + 0.5) / self.theta_bins
        theta = theta_norm * 180 - 90

        # Width and height
        w_norm = (bins['w'] + 0.5) / self.w_bins
        h_norm = (bins['h'] + 0.5) / self.h_bins

        w = w_norm * self.img_width
        h = h_norm * self.img_width

        return {
            'cx': float(cx),
            'cy': float(cy),
            'theta': float(theta),
            'w': float(w),
            'h': float(h),
        }

    def bins_to_string(self, bins: Dict[str, int]) -> str:
        """Format bins as space-separated string for LM output."""
        return f"{bins['x']} {bins['y']} {bins['theta']} {bins['w']} {bins['h']}"

    def string_to_bins(self, s: str) -> Dict[str, int]:
        """Parse LM output string to bins."""
        parts = s.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 integers, got {len(parts)}: {s}")

        try:
            x, y, theta, w, h = map(int, parts)
        except ValueError as e:
            raise ValueError(f"Could not parse integers from: {s}") from e

        # Clamp to valid ranges
        return {
            'x': np.clip(x, 0, self.x_bins - 1),
            'y': np.clip(y, 0, self.y_bins - 1),
            'theta': np.clip(theta, 0, self.theta_bins - 1),
            'w': np.clip(w, 0, self.w_bins - 1),
            'h': np.clip(h, 0, self.h_bins - 1),
        }


def angle_diff(a1: float, a2: float) -> float:
    """Compute angular difference handling parallel-jaw symmetry."""
    # Normalize both to [-90, 90)
    a1 = ((a1 + 90) % 180) - 90
    a2 = ((a2 + 90) % 180) - 90

    diff = abs(a1 - a2)
    # Handle wrap-around at ±90
    if diff > 90:
        diff = 180 - diff
    return diff


def rotate_grasp(grasp: Dict[str, float], angle_deg: float,
                 img_center: Tuple[float, float]) -> Dict[str, float]:
    """
    Rotate grasp rectangle around image center.
    Useful for augmentation.

    Args:
        grasp: Grasp dict with cx, cy, theta, w, h
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)
        img_center: (cx_img, cy_img) in pixels

    Returns:
        Rotated grasp dict
    """
    cx, cy = grasp['cx'], grasp['cy']
    cx_img, cy_img = img_center

    # Translate to origin
    dx = cx - cx_img
    dy = cy - cy_img

    # Rotate
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # Translate back
    cx_new = dx_rot + cx_img
    cy_new = dy_rot + cy_img

    # Rotate theta
    theta_new = grasp['theta'] + angle_deg
    # Normalize to [-90, 90)
    theta_new = ((theta_new + 90) % 180) - 90

    return {
        'cx': cx_new,
        'cy': cy_new,
        'theta': theta_new,
        'w': grasp['w'],
        'h': grasp['h'],
    }
