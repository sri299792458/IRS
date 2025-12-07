"""
Format samples into Qwen3-VL chat format.
Updated for language-conditioned grasp detection (OCID-VLG).
"""
from typing import Dict, List
from PIL import Image
import numpy as np


SYSTEM_PROMPT = """You are a robot grasp planner. Given an image and a description of a target object, output the grasp pose for that specific object.

Output exactly 5 integers separated by spaces: x y theta w h
- x: horizontal center position [0-999]
- y: vertical center position [0-999]  
- theta: rotation angle bin [0-179] (maps from -90° to +90°)
- w: gripper opening width [0-999]
- h: grasp rectangle length [0-999]

Output only the 5 numbers, space-separated. Nothing else."""


USER_PROMPT_TEMPLATE = """Grasp the object: {instruction}

Output: x y theta w h"""


USER_PROMPT_NO_INSTRUCTION = """Identify the most graspable object and output a grasp pose.

Output: x y theta w h"""


def format_chat_sample(image: np.ndarray,
                       grasp_string: str,
                       instruction: str = "") -> List[Dict]:
    """
    Create Qwen3-VL chat format for training.

    Args:
        image: (H, W, 3) RGB numpy array
        grasp_string: "x y theta w h" target string (e.g., "512 304 27 118 246")
        instruction: Referring expression (e.g., "the red apple on the left")

    Returns:
        List of message dicts for Qwen3-VL
    """
    # Convert numpy to PIL
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image

    # Build user message with instruction
    if instruction and instruction.strip():
        user_text = USER_PROMPT_TEMPLATE.format(instruction=instruction)
    else:
        user_text = USER_PROMPT_NO_INSTRUCTION

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
        instruction: Referring expression for target object

    Returns:
        Messages list (without assistant turn)
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image

    if instruction and instruction.strip():
        user_text = USER_PROMPT_TEMPLATE.format(instruction=instruction)
    else:
        user_text = USER_PROMPT_NO_INSTRUCTION

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


# ============================================================================
# Alternative prompt styles (for ablation studies)
# ============================================================================

SYSTEM_PROMPT_MINIMAL = "You are a robot grasp planner. Output only numbers."

SYSTEM_PROMPT_DETAILED = """You are an expert robot grasp planner for a parallel-jaw gripper.

Given:
1. An RGB image of a cluttered scene with multiple objects
2. A natural language description identifying a specific target object

Your task:
Predict a grasp pose that would allow a parallel-jaw gripper to successfully pick up ONLY the described target object.

Output format: 5 space-separated integers
- x: horizontal center of grasp [0-999] (0=left edge, 999=right edge)
- y: vertical center of grasp [0-999] (0=top edge, 999=bottom edge)
- theta: rotation bin [0-179] (0=-90°, 89=0°, 179=+89°)
- w: gripper opening width [0-999] (scaled relative to image width)
- h: grasp rectangle length [0-999] (along the gripper fingers)

Important: Output ONLY the 5 numbers. No explanation, no text, no punctuation."""


def format_chat_sample_vla0_style(image: np.ndarray,
                                   grasp_string: str,
                                   instruction: str = "") -> List[Dict]:
    """
    VLA-0 style formatting: more minimal prompt.
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image

    # VLA-0 uses very minimal prompts
    if instruction and instruction.strip():
        user_text = f"Grasp: {instruction}"
    else:
        user_text = "Grasp the object."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_MINIMAL
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
            "content": grasp_string
        }
    ]

    return messages


# ============================================================================
# Utilities for tokenization and loss masking
# ============================================================================

def extract_assistant_tokens(input_ids, tokenizer):
    """
    Find indices of assistant response tokens for loss masking.
    
    Note: This is typically handled by the data collator now,
    but kept here for reference.

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


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == '__main__':
    # Test the chat formatting
    import numpy as np
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test with instruction
    messages = format_chat_sample(
        dummy_image,
        grasp_string="512 304 89 118 246",
        instruction="the red apple on the left side of the table"
    )
    
    print("="*60)
    print("Chat Format with Instruction:")
    print("="*60)
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        if isinstance(msg['content'], str):
            print(msg['content'])
        else:
            for item in msg['content']:
                if item['type'] == 'text':
                    print(item['text'])
                else:
                    print(f"<{item['type']}>")
    
    # Test without instruction
    print("\n" + "="*60)
    print("Chat Format without Instruction:")
    print("="*60)
    messages_no_inst = format_chat_sample(
        dummy_image,
        grasp_string="512 304 89 118 246",
        instruction=""
    )
    
    for msg in messages_no_inst:
        print(f"\n[{msg['role'].upper()}]")
        if isinstance(msg['content'], str):
            print(msg['content'])
        else:
            for item in msg['content']:
                if item['type'] == 'text':
                    print(item['text'])
                else:
                    print(f"<{item['type']}>")
