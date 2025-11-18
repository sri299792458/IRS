"""
Custom data collator for Qwen3-VL grasp training.
Handles loss masking for chat format.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer


@dataclass
class GraspDataCollator:
    """
    Collator for Qwen3-VL chat format.
    Masks loss on system/user turns, only computes on assistant response.
    """

    processor: Any  # Qwen3VLProcessor
    tokenizer: PreTrainedTokenizer = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and create labels with masking.

        Args:
            features: List of dicts from dataset

        Returns:
            Batch dict with input_ids, attention_mask, labels, etc.
        """
        # Extract messages
        messages_list = [f['messages'] for f in features]

        # Process with Qwen processor (handles images, tokenization, padding)
        batch = self.processor(
            messages_list,
            padding=True,
            return_tensors='pt',
        )

        # Create labels by masking non-assistant tokens
        labels = batch['input_ids'].clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask system and user turns (only compute loss on assistant)
        # We need to find where the assistant response starts
        for i in range(len(labels)):
            input_ids = batch['input_ids'][i]
            masked_labels = self._mask_non_assistant(input_ids)
            labels[i] = masked_labels

        batch['labels'] = labels

        # Add metadata
        batch['image_ids'] = [f['image_id'] for f in features]
        batch['grasp_bins'] = [f['grasp_bins'] for f in features]

        return batch

    def _mask_non_assistant(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mask tokens before assistant response.

        Args:
            input_ids: (seq_len,)

        Returns:
            labels: (seq_len,) with -100 for masked positions
        """
        labels = input_ids.clone()

        # Find assistant turn
        # Qwen uses special format with <|im_start|>assistant\n
        # We'll look for the assistant role token

        # Decode to find pattern (hacky but works for research)
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find "assistant" in decoded text
        # Then find corresponding token index
        assistant_markers = ['assistant', '<|im_start|>assistant']

        assistant_start_idx = None
        for marker in assistant_markers:
            # Encode marker and search
            marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if len(marker_ids) == 0:
                continue

            # Search for marker in input_ids
            for i in range(len(input_ids) - len(marker_ids) + 1):
                if torch.all(input_ids[i:i+len(marker_ids)] == torch.tensor(marker_ids)):
                    # Found it! Mask everything before end of marker
                    assistant_start_idx = i + len(marker_ids)
                    break

            if assistant_start_idx is not None:
                break

        if assistant_start_idx is None:
            # Fallback: mask first 80% (crude but avoids loss on prompt)
            assistant_start_idx = int(len(input_ids) * 0.8)

        # Mask everything before assistant response
        labels[:assistant_start_idx] = -100

        return labels


def compute_grasp_metrics(eval_pred):
    """
    Compute metrics during evaluation.
    For now just perplexity; full metrics done post-training.
    """
    predictions, labels = eval_pred
    # predictions are logits, shape (batch, seq_len, vocab)

    # Just return loss as metric
    # (Trainer computes this automatically, but we can add more here)

    return {}
