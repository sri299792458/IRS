"""
Constrained decoding: force model to output only digits and spaces.
Prevents garbage outputs like "The grasp is..." or emoji.
"""
import torch
from transformers import LogitsProcessor
from typing import List


class DigitsOnlyLogitsProcessor(LogitsProcessor):
    """
    Mask all tokens except 0-9 and space.
    Ensures LM outputs clean "x y theta w h" format.
    """

    def __init__(self, tokenizer, allow_eos: bool = True):
        """
        Args:
            tokenizer: Qwen tokenizer
            allow_eos: Whether to allow EOS token (end generation)
        """
        self.tokenizer = tokenizer
        self.allow_eos = allow_eos

        # Get digit token IDs
        self.allowed_tokens = self._get_allowed_tokens()

    def _get_allowed_tokens(self) -> List[int]:
        """Find token IDs for digits 0-9 and space."""
        allowed = []

        # Digits 0-9
        for digit in '0123456789':
            # Try different tokenizations
            for prefix in ['', ' ', '  ']:  # Handle space prefixes
                token_id = self.tokenizer.encode(prefix + digit, add_special_tokens=False)
                if token_id:
                    allowed.extend(token_id)

        # Space
        space_tokens = self.tokenizer.encode(' ', add_special_tokens=False)
        allowed.extend(space_tokens)

        # Newline (sometimes needed)
        newline_tokens = self.tokenizer.encode('\n', add_special_tokens=False)
        allowed.extend(newline_tokens)

        # EOS token
        if self.allow_eos:
            allowed.append(self.tokenizer.eos_token_id)

        # Remove duplicates
        allowed = list(set(allowed))

        print(f"Allowed tokens for digits-only: {len(allowed)} tokens")
        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask logits for non-digit tokens.

        Args:
            input_ids: (batch, seq_len)
            scores: (batch, vocab_size)

        Returns:
            Masked scores
        """
        # Create mask: -inf for disallowed tokens
        mask = torch.full_like(scores, float('-inf'))

        # Allow only digit tokens
        mask[:, self.allowed_tokens] = 0

        return scores + mask


class GraspFormatValidator(LogitsProcessor):
    """
    More sophisticated: enforce "int int int int int" format.
    Tracks state and only allows valid next tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = 'start'  # start, digit, space
        self.num_count = 0  # How many numbers emitted

        # Precompute token sets
        self.digit_tokens = self._get_digit_tokens()
        self.space_tokens = self._get_space_tokens()
        self.eos_token = tokenizer.eos_token_id

    def _get_digit_tokens(self) -> List[int]:
        tokens = []
        for digit in '0123456789':
            for prefix in ['', ' ']:
                ids = self.tokenizer.encode(prefix + digit, add_special_tokens=False)
                tokens.extend(ids)
        return list(set(tokens))

    def _get_space_tokens(self) -> List[int]:
        return self.tokenizer.encode(' ', add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Enforce state machine for "int space int space ..." pattern."""
        mask = torch.full_like(scores, float('-inf'))

        # Allow digits if we haven't finished all 5 numbers
        if self.num_count < 5:
            mask[:, self.digit_tokens] = 0

        # Allow space after a digit (if not the last number)
        if self.num_count < 4:
            mask[:, self.space_tokens] = 0

        # Allow EOS if we've emitted 5 numbers
        if self.num_count >= 5:
            mask[:, self.eos_token] = 0

        return scores + mask


def parse_grasp_output(text: str) -> dict:
    """
    Parse model output to extract 5 integers.

    Args:
        text: Model generated text

    Returns:
        Dict with x, y, theta, w, h or None if invalid
    """
    # Clean text: extract only digits and spaces
    import re
    clean = re.sub(r'[^0-9\s]', '', text)
    parts = clean.split()

    # Filter valid integers
    nums = []
    for part in parts:
        try:
            nums.append(int(part))
        except ValueError:
            continue

    if len(nums) < 5:
        return None

    # Take first 5
    return {
        'x': nums[0],
        'y': nums[1],
        'theta': nums[2],
        'w': nums[3],
        'h': nums[4],
    }
