"""
Evaluation metrics for grasp detection.
Rotated IoU, angle error, center error.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Dict, List, Tuple
import cv2


def grasp_to_polygon(grasp: Dict[str, float]) -> Polygon:
    """
    Convert grasp dict to Shapely Polygon.

    Args:
        grasp: Dict with cx, cy, theta, w, h

    Returns:
        Shapely Polygon
    """
    cx, cy = grasp['cx'], grasp['cy']
    w, h = grasp['w'], grasp['h']
    theta = grasp['theta']

    # Get rectangle corners using OpenCV
    center = (float(cx), float(cy))
    size = (float(w), float(h))
    angle = float(theta)

    rect = cv2.RotatedRect(center, size, angle)
    box = cv2.boxPoints(rect)

    # Convert to Polygon
    polygon = Polygon(box)

    return polygon


def rotated_iou(pred: Dict, gt: Dict, angle_tolerance: float = 30.0) -> float:
    """
    Compute IoU for oriented rectangles.

    Args:
        pred: Predicted grasp dict
        gt: Ground truth grasp dict
        angle_tolerance: Max angle difference in degrees (return 0 if exceeded)

    Returns:
        IoU in [0, 1]
    """
    # Check angle tolerance first
    angle_diff = abs(angle_difference(pred['theta'], gt['theta']))
    if angle_diff > angle_tolerance:
        return 0.0

    # Convert to polygons
    try:
        poly_pred = grasp_to_polygon(pred)
        poly_gt = grasp_to_polygon(gt)

        # Compute intersection and union
        if not poly_pred.is_valid or not poly_gt.is_valid:
            return 0.0

        intersection = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area

        if union < 1e-6:
            return 0.0

        iou = intersection / union

        return float(iou)

    except Exception as e:
        # Handle degenerate cases
        print(f"Warning: IoU computation failed: {e}")
        return 0.0


def angle_difference(a1: float, a2: float) -> float:
    """
    Compute angle difference handling parallel-jaw symmetry.
    Angles in degrees, normalized to [-90, 90).

    Args:
        a1, a2: Angles in degrees

    Returns:
        Absolute difference in [0, 90]
    """
    # Normalize both to [-90, 90)
    a1 = ((a1 + 90) % 180) - 90
    a2 = ((a2 + 90) % 180) - 90

    diff = abs(a1 - a2)

    # Handle wrap-around at ±90
    if diff > 90:
        diff = 180 - diff

    return diff


def center_distance(pred: Dict, gt: Dict) -> float:
    """
    Euclidean distance between centers.

    Args:
        pred, gt: Grasp dicts

    Returns:
        Distance in pixels
    """
    dx = pred['cx'] - gt['cx']
    dy = pred['cy'] - gt['cy']

    return float(np.sqrt(dx**2 + dy**2))


def compute_metrics(predictions: List[Dict],
                   ground_truths: List[Dict],
                   iou_threshold: float = 0.25,
                   angle_tolerance: float = 30.0) -> Dict[str, float]:
    """
    Compute all metrics for a set of predictions.

    Args:
        predictions: List of predicted grasps
        ground_truths: List of ground truth grasps
        iou_threshold: IoU threshold for success
        angle_tolerance: Angle tolerance in degrees

    Returns:
        Dict of metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(ground_truths)}")

    n = len(predictions)

    # Compute per-sample metrics
    ious = []
    angle_errors = []
    center_errors = []
    successes = []

    for pred, gt in zip(predictions, ground_truths):
        # IoU
        iou = rotated_iou(pred, gt, angle_tolerance=angle_tolerance)
        ious.append(iou)

        # Success (IoU above threshold)
        successes.append(iou >= iou_threshold)

        # Angle error
        angle_err = angle_difference(pred['theta'], gt['theta'])
        angle_errors.append(angle_err)

        # Center error
        center_err = center_distance(pred, gt)
        center_errors.append(center_err)

    # Aggregate metrics
    metrics = {
        'mean_iou': float(np.mean(ious)),
        'median_iou': float(np.median(ious)),
        f'success_rate@{iou_threshold}': float(np.mean(successes)),
        'mean_angle_error_deg': float(np.mean(angle_errors)),
        'median_angle_error_deg': float(np.median(angle_errors)),
        'mean_center_error_px': float(np.mean(center_errors)),
        'median_center_error_px': float(np.median(center_errors)),
        'num_samples': n,
    }

    return metrics


def compute_per_class_metrics(predictions: List[Dict],
                              ground_truths: List[Dict],
                              labels: List[str],
                              iou_threshold: float = 0.25) -> Dict[str, Dict]:
    """
    Compute metrics per object class.

    Args:
        predictions: List of predicted grasps
        ground_truths: List of ground truth grasps
        labels: List of class labels (same length as predictions)
        iou_threshold: IoU threshold

    Returns:
        Dict mapping class name to metrics dict
    """
    # Group by class
    class_groups = {}
    for pred, gt, label in zip(predictions, ground_truths, labels):
        if label not in class_groups:
            class_groups[label] = {'preds': [], 'gts': []}

        class_groups[label]['preds'].append(pred)
        class_groups[label]['gts'].append(gt)

    # Compute metrics per class
    per_class_metrics = {}
    for class_name, group in class_groups.items():
        metrics = compute_metrics(
            group['preds'],
            group['gts'],
            iou_threshold=iou_threshold
        )
        per_class_metrics[class_name] = metrics

    return per_class_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key or 'iou' in key:
                print(f"  {key:30s}: {value:6.3f} ({value*100:.1f}%)")
            else:
                print(f"  {key:30s}: {value:6.2f}")
        else:
            print(f"  {key:30s}: {value}")

    print("="*60)
