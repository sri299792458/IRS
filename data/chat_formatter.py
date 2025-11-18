"""
Format samples into Qwen3-VL chat format.
"""
from typing import Dict, List
from PIL import Image
import numpy as np


SYSTEM_PROMPT = "You are a robot grasp planner. Output only numbers."

USER_PROMPT_TEMPLATE = """Output exactly 5 integers separated by spaces: x y theta w h

Format: x y theta w h
- x: horizontal center [0-1000] (left to right)
- y: vertical center [0-1000] (top to bottom)
- theta: angle bin [0-179] (maps from -90° to +90°)
- w: width [0-1000] (gripper opening)
- h: height [0-1000] (rectangle length)

Output only the 5 numbers, space-separated. No other text."""


def format_chat_sample(image: np.ndarray,
                       grasp_string: str,
                       instruction: str = "") -> List[Dict]:
    """
    Create Qwen3-VL chat format.

    Args:
        image: (H, W, 3) RGB numpy array
        grasp_string: "x y theta w h" target string
        instruction: Optional language instruction (for OCID-VLG)

    Returns:
        List of message dicts for Qwen3-VL
    """
    # Convert numpy to PIL
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image

    # Build user message
    user_text = USER_PROMPT_TEMPLATE
    if instruction:
        user_text = f"Instruction: {instruction}\n\n{user_text}"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": user_text}
            ]
        },
        {
            "role": "assistant",
            "content": grasp_string  # "512 304 27 118 246"
        }
    ]

    return messages


def format_inference_messages(image: np.ndarray,
                               instruction: str = "") -> List[Dict]:
    """
    Create chat messages for inference (no assistant response).

    Args:
        image: RGB image
        instruction: Optional language instruction

    Returns:
        Messages list (without assistant turn)
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image

    user_text = USER_PROMPT_TEMPLATE
    if instruction:
        user_text = f"Instruction: {instruction}\n\n{user_text}"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": user_text}
            ]
        }
    ]

    return messages


def extract_assistant_tokens(input_ids, tokenizer):
    """
    Find indices of assistant response tokens for loss masking.

    Args:
        input_ids: Full sequence token IDs
        tokenizer: Qwen tokenizer

    Returns:
        Boolean mask (True = compute loss, False = ignore)
    """
    # Find the assistant header token(s)
    # Qwen uses special tokens like <|im_start|>assistant
    assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    # Find where assistant turn starts
    assistant_start = None
    for i in range(len(input_ids)):
        if input_ids[i] == assistant_token_id:
            assistant_start = i + 1  # Start after the header
            break

    if assistant_start is None:
        # Fallback: compute loss on everything (not ideal)
        return [True] * len(input_ids)

    # Mask: False before assistant, True after
    mask = [False] * assistant_start + [True] * (len(input_ids) - assistant_start)

    return mask
