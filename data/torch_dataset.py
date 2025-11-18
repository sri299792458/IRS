"""
PyTorch Dataset wrapper for OCID-Grasp.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional
import random

from data.ocid_grasp_loader import OCIDGraspDataset
from data.grasp_quantizer import GraspQuantizer
from data.augmentations import augment_sample, augment_label_noise
from data.chat_formatter import format_chat_sample


class GraspVLMDataset(Dataset):
    """
    PyTorch dataset for VLM grasp training.
    Returns formatted chat messages ready for Qwen3-VL.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 config: Optional[Dict] = None,
                 quantizer: Optional[GraspQuantizer] = None):
        """
        Args:
            data_dir: Path to OCID-Grasp data
            split: 'train', 'val', or 'test'
            config: Config dict
            quantizer: GraspQuantizer instance
        """
        self.split = split
        self.config = config or {}
        self.is_train = (split == 'train')

        # Load raw data
        self.raw_dataset = OCIDGraspDataset(data_dir, split)

        # Quantizer
        if quantizer is None:
            self.quantizer = GraspQuantizer(
                img_width=config.get('image_size', (480, 640))[1],
                img_height=config.get('image_size', (480, 640))[0],
                x_bins=config.get('x_bins', 1000),
                y_bins=config.get('y_bins', 1000),
                theta_bins=config.get('theta_bins', 180),
                w_bins=config.get('w_bins', 1000),
                h_bins=config.get('h_bins', 1000),
            )
        else:
            self.quantizer = quantizer

        # Flatten: one sample per grasp
        self.samples = []
        for idx in range(len(self.raw_dataset)):
            item = self.raw_dataset[idx]
            image = item['image']
            grasps = item['grasps']
            instruction = item.get('sentence', '')

            for grasp in grasps:
                self.samples.append({
                    'image': image,
                    'grasp': grasp,
                    'instruction': instruction,
                    'image_id': item['image_id'],
                })

        print(f"{split}: {len(self.samples)} grasp samples from {len(self.raw_dataset)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with:
                'messages': Chat format for Qwen3-VL
                'image': PIL Image
                'grasp_bins': Quantized bins dict
                'grasp_continuous': Original continuous grasp
        """
        sample = self.samples[idx]

        image = sample['image'].copy()
        grasp = sample['grasp'].copy()
        instruction = sample['instruction']

        # Augmentation (only on train)
        if self.is_train:
            image, grasp = augment_sample(image, grasp, self.config)

        # Quantize grasp
        grasp_bins = self.quantizer.encode(grasp)

        # Label noise (only on train)
        if self.is_train and self.config.get('aug_label_noise_prob', 0) > 0:
            grasp_bins = augment_label_noise(
                grasp_bins,
                noise_prob=self.config['aug_label_noise_prob']
            )

        # Format as string
        grasp_string = self.quantizer.bins_to_string(grasp_bins)

        # Create chat messages
        messages = format_chat_sample(image, grasp_string, instruction)

        return {
            'messages': messages,
            'image_id': sample['image_id'],
            'grasp_bins': grasp_bins,
            'grasp_continuous': grasp,
        }


def collate_fn(batch, processor):
    """
    Custom collate function for Qwen3-VL.

    Args:
        batch: List of dicts from dataset
        processor: Qwen3VLProcessor

    Returns:
        Batch dict ready for model
    """
    # Extract messages
    messages_list = [item['messages'] for item in batch]

    # Process with Qwen processor
    # This handles image encoding, tokenization, etc.
    batch_encoded = processor(
        messages_list,
        padding=True,
        return_tensors='pt'
    )

    # Add metadata
    batch_encoded['image_ids'] = [item['image_id'] for item in batch]
    batch_encoded['grasp_bins'] = [item['grasp_bins'] for item in batch]

    return batch_encoded
