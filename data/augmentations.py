"""
Data augmentations for grasp detection.
Rotate, crop, color jitter, label noise.
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from typing import Dict, Tuple
import random


def augment_rotation(image: np.ndarray, grasp: Dict[str, float],
                     max_angle: float = 15.0) -> Tuple[np.ndarray, Dict]:
    """
    Randomly rotate image and grasp.

    Args:
        image: (H, W, 3) RGB image
        grasp: Grasp dict with cx, cy, theta, w, h
        max_angle: Maximum rotation angle in degrees

    Returns:
        (rotated_image, rotated_grasp)
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    # Rotate image
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rot = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

    # Rotate grasp
    from data.grasp_quantizer import rotate_grasp
    grasp_rot = rotate_grasp(grasp, angle, center)

    # Clamp to image bounds
    grasp_rot['cx'] = np.clip(grasp_rot['cx'], 0, w)
    grasp_rot['cy'] = np.clip(grasp_rot['cy'], 0, h)

    return image_rot, grasp_rot


def augment_color_jitter(image: np.ndarray,
                         brightness: float = 0.2,
                         contrast: float = 0.2,
                         saturation: float = 0.2) -> np.ndarray:
    """
    Random color jittering (doesn't affect grasp labels).

    Args:
        image: (H, W, 3) RGB image
        brightness, contrast, saturation: Jitter range [1-x, 1+x]

    Returns:
        Jittered image
    """
    img_pil = Image.fromarray(image)

    # Brightness
    if random.random() < 0.8:
        factor = random.uniform(1 - brightness, 1 + brightness)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(factor)

    # Contrast
    if random.random() < 0.8:
        factor = random.uniform(1 - contrast, 1 + contrast)
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(factor)

    # Saturation
    if random.random() < 0.8:
        factor = random.uniform(1 - saturation, 1 + saturation)
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(factor)

    return np.array(img_pil)


def augment_label_noise(bins: Dict[str, int],
                        noise_prob: float = 0.2) -> Dict[str, int]:
    """
    Add ±1 bin jitter to a random field.
    Makes model robust to small quantization errors.

    Args:
        bins: Quantized grasp bins
        noise_prob: Probability of adding noise

    Returns:
        Noisy bins
    """
    if random.random() > noise_prob:
        return bins

    # Pick a random field
    field = random.choice(['x', 'y', 'theta', 'w', 'h'])
    delta = random.choice([-1, 1])

    bins_noisy = bins.copy()
    bins_noisy[field] = bins[field] + delta

    # Clamp (assuming standard ranges)
    if field in ['x', 'y', 'w', 'h']:
        bins_noisy[field] = np.clip(bins_noisy[field], 0, 999)
    elif field == 'theta':
        bins_noisy[field] = np.clip(bins_noisy[field], 0, 179)

    return bins_noisy


def augment_mask_field(text: str, mask_prob: float = 0.1) -> str:
    """
    Randomly mask one field in the output string.
    VLA-style robustness trick.

    Args:
        text: "x y theta w h" string
        mask_prob: Probability of masking

    Returns:
        Possibly masked string like "x y [MASK] w h"
    """
    if random.random() > mask_prob:
        return text

    parts = text.split()
    if len(parts) != 5:
        return text

    # Mask a random field
    idx = random.randint(0, 4)
    parts[idx] = '[MASK]'

    return ' '.join(parts)


def augment_sample(image: np.ndarray, grasp: Dict[str, float],
                   config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Apply full augmentation pipeline.

    Args:
        image: RGB image array
        grasp: Grasp dict
        config: Config dict with augmentation params

    Returns:
        (augmented_image, augmented_grasp)
    """
    # Rotation (affects both image and grasp)
    if config.get('aug_rotation_deg', 0) > 0:
        image, grasp = augment_rotation(
            image, grasp, max_angle=config['aug_rotation_deg']
        )

    # Color jitter (image only)
    if config.get('aug_color_jitter', False):
        image = augment_color_jitter(image)

    return image, grasp
