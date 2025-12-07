"""
PyTorch Dataset wrapper for OCID-VLG.
Language-conditioned grasp detection.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional
import random

from ocid_vlg_loader import OCIDVLGDataset
from grasp_quantizer import GraspQuantizer
from augmentations import augment_sample, augment_label_noise
from chat_formatter_vlg import format_chat_sample


class GraspVLGDataset(Dataset):
    """
    PyTorch dataset for VLM language-guided grasp training.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 version: str = 'multiple',
                 config: Optional[Dict] = None,
                 quantizer: Optional[GraspQuantizer] = None,
                 grasp_selection: str = 'random'):
        """
        Args:
            data_dir: Path to OCID-VLG data
            split: 'train', 'val', or 'test'
            version: 'multiple', 'unique', 'novel-instances', 'novel-classes'
            config: Config dict
            quantizer: GraspQuantizer instance
            grasp_selection: How to select grasp when multiple exist
        """
        self.split = split
        self.version = version
        self.config = config or {}
        self.is_train = (split == 'train')
        self.grasp_selection = grasp_selection

        # Load raw data (JSON annotations only at this stage)
        self.raw_dataset = OCIDVLGDataset(data_dir, split=split, version=version)

        # Quantizer
        if quantizer is None:
            img_size = config.get('image_size', (480, 640))
            self.quantizer = GraspQuantizer(
                img_width=img_size[1],
                img_height=img_size[0],
                x_bins=config.get('x_bins', 1000),
                y_bins=config.get('y_bins', 1000),
                theta_bins=config.get('theta_bins', 180),
                w_bins=config.get('w_bins', 1000),
                h_bins=config.get('h_bins', 1000),
            )
        else:
            self.quantizer = quantizer

        # Optimized Filtering:
        # Instead of loading every image (OOM risk), we iterate the metadata only.
        # OCIDVLGDataset exposes .samples which is the list of JSON dicts.
        self.valid_indices = []
        for idx, item in enumerate(self.raw_dataset.samples):
            # Check for valid grasps without loading image
            # Note: The loader might store grasps as lists or arrays in the json dict
            grasps = item.get('grasps', [])
            if len(grasps) > 0:
                self.valid_indices.append(idx)

        print(f"OCID-VLG {version}/{split}: {len(self.valid_indices)} samples (filtered from {len(self.raw_dataset.samples)})")
        print(f"  (Grasp selection: {grasp_selection})")

    def _select_grasp(self, grasps: list) -> Dict:
        """
        Select one grasp from the list of valid grasps.
        """
        # Grasps come from __getitem__ already formatted as dicts
        if len(grasps) == 1:
            return grasps[0]
        
        if self.grasp_selection == 'random':
            return random.choice(grasps)
        
        elif self.grasp_selection == 'first':
            return grasps[0]
        
        elif self.grasp_selection == 'center':
            img_cx = self.quantizer.img_width / 2
            img_cy = self.quantizer.img_height / 2
            def dist_to_center(g):
                return (g['cx'] - img_cx)**2 + (g['cy'] - img_cy)**2
            return min(grasps, key=dist_to_center)
        
        else:
            return grasps[0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Lazy loading of sample.
        """
        # Map to real index in raw dataset
        real_idx = self.valid_indices[idx]
        
        # This triggers the image load ONLY now
        sample = self.raw_dataset[real_idx]

        image = sample['image'].copy()
        sentence = sample['sentence']
        grasps = sample['grasps']

        # Select one grasp (for training target)
        grasp = self._select_grasp(grasps).copy()

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

        # Create chat messages with the referring expression
        messages = format_chat_sample(image, grasp_string, instruction=sentence)

        return {
            'messages': messages,
            'image_id': sample['image_id'],
            'grasp_bins': grasp_bins,
            'grasp_continuous': grasp,
            'sentence': sentence,
            'target': sample['target'],
            'all_grasps': grasps,  # Keep all for evaluation
        }


class GraspVLGEvalDataset(Dataset):
    """
    Evaluation dataset that returns ALL grasps for proper metric computation.
    
    During evaluation, we want to check if the predicted grasp matches
    ANY of the valid grasps for the target object.
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'test',
                 version: str = 'multiple',
                 config: Optional[Dict] = None,
                 quantizer: Optional[GraspQuantizer] = None):
        """
        Args:
            data_dir: Path to OCID-VLG data
            split: 'val' or 'test'
            version: Dataset version
            config: Config dict
            quantizer: GraspQuantizer instance
        """
        self.split = split
        self.version = version
        self.config = config or {}

        # Load raw data
        self.raw_dataset = OCIDVLGDataset(data_dir, split=split, version=version)

        # Quantizer
        if quantizer is None:
            img_size = config.get('image_size', (480, 640))
            self.quantizer = GraspQuantizer(
                img_width=img_size[1],
                img_height=img_size[0],
                x_bins=config.get('x_bins', 1000),
                y_bins=config.get('y_bins', 1000),
                theta_bins=config.get('theta_bins', 180),
                w_bins=config.get('w_bins', 1000),
                h_bins=config.get('h_bins', 1000),
            )
        else:
            self.quantizer = quantizer

        # Build samples (no filtering, keep all)
        self.samples = []
        for idx in range(len(self.raw_dataset)):
            item = self.raw_dataset[idx]
            if len(item['grasps']) > 0:
                self.samples.append(item)

        print(f"OCID-VLG Eval {version}/{split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns sample with ALL ground truth grasps.
        """
        sample = self.samples[idx]

        return {
            'image': sample['image'],
            'sentence': sample['sentence'],
            'grasps': sample['grasps'],  # ALL grasps for this target
            'target': sample['target'],
            'target_idx': sample['target_idx'],
            'mask': sample['mask'],
            'image_id': sample['image_id'],
        }


def collate_fn_vlg(batch, processor):
    """
    Custom collate function for Qwen3-VL with OCID-VLG data.

    Args:
        batch: List of dicts from dataset
        processor: Qwen3VLProcessor

    Returns:
        Batch dict ready for model
    """
    # Extract messages
    messages_list = [item['messages'] for item in batch]

    # Process with Qwen processor
    batch_encoded = processor(
        messages_list,
        padding=True,
        return_tensors='pt'
    )

    # Add metadata
    batch_encoded['image_ids'] = [item['image_id'] for item in batch]
    batch_encoded['grasp_bins'] = [item['grasp_bins'] for item in batch]
    batch_encoded['sentences'] = [item['sentence'] for item in batch]
    batch_encoded['all_grasps'] = [item['all_grasps'] for item in batch]

    return batch_encoded


# ============================================================================
# Testing and inspection
# ============================================================================

def test_vlg_dataset(data_dir: str, config: Dict):
    """Test the VLG dataset loading."""
    print("="*60)
    print("Testing GraspVLGDataset")
    print("="*60)
    
    # Initialize quantizer
    quantizer = GraspQuantizer(
        img_width=config.get('image_size', (480, 640))[1],
        img_height=config.get('image_size', (480, 640))[0],
        x_bins=config.get('x_bins', 1000),
        y_bins=config.get('y_bins', 1000),
        theta_bins=config.get('theta_bins', 180),
        w_bins=config.get('w_bins', 1000),
        h_bins=config.get('h_bins', 1000),
    )
    
    # Load dataset
    dataset = GraspVLGDataset(
        data_dir,
        split='train',
        version='multiple',
        config=config,
        quantizer=quantizer,
        grasp_selection='random'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Show some samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Sentence: '{sample['sentence']}'")
        print(f"  Target: {sample['target']}")
        print(f"  Grasp string: {quantizer.bins_to_string(sample['grasp_bins'])}")
        print(f"  Num valid grasps: {len(sample['all_grasps'])}")
        
        # Extract user text from messages
        user_content = sample['messages'][1]['content']
        for item in user_content:
            if item['type'] == 'text':
                print(f"  User prompt:\n    {item['text'][:100]}...")
                break
    
    print("\n" + "="*60)
    print("Dataset test passed!")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OCID-VLG dataset')
    
    args = parser.parse_args()
    
    # Default config
    config = {
        'image_size': (480, 640),
        'x_bins': 1000,
        'y_bins': 1000,
        'theta_bins': 180,
        'w_bins': 1000,
        'h_bins': 1000,
        'aug_rotation_deg': 15,
        'aug_color_jitter': True,
        'aug_label_noise_prob': 0.2,
    }
    
    test_vlg_dataset(args.data_dir, config)
