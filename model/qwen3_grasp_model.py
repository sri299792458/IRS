"""
Qwen3-VL model setup with LoRA for grasp prediction.
"""
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Optional


def load_model_and_processor(config: Dict, use_qlora: bool = True):
    """
    Load Qwen3-VL-8B with QLoRA configuration.

    Args:
        config: Config dict
        use_qlora: Whether to use 4-bit quantization

    Returns:
        (model, processor)
    """
    model_name = config['model_name']

    print(f"Loading {model_name}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Quantization config
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Double quantization for more memory savings
        )
        print("Using QLoRA (4-bit NF4)")
    else:
        bnb_config = None
        print("Using full precision")

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training (required for QLoRA)
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=config.get('lora_target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")

    return model, processor


def load_trained_model(checkpoint_path: str, config: Dict):
    """
    Load a trained LoRA model from checkpoint.

    Args:
        checkpoint_path: Path to saved LoRA adapter
        config: Config dict

    Returns:
        (model, processor)
    """
    from peft import PeftModel

    model_name = config['model_name']

    # Load base model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # For inference, we can load in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    print(f"Loaded trained model from {checkpoint_path}")

    return model, processor


class GraspModelWrapper:
    """
    Convenience wrapper for inference.
    """

    def __init__(self, model, processor, quantizer, config: Dict):
        self.model = model
        self.processor = processor
        self.quantizer = quantizer
        self.config = config

        self.model.eval()

    @torch.no_grad()
    def predict_grasp(self, image, instruction: str = "",
                      use_constrained: bool = True,
                      temperature: float = 0.2):
        """
        Predict grasp for a single image.

        Args:
            image: PIL Image or numpy array
            instruction: Optional language instruction
            use_constrained: Use digits-only logits processor
            temperature: Sampling temperature

        Returns:
            Grasp dict with cx, cy, theta, w, h
        """
        from data.chat_formatter import format_inference_messages
        from model.constrained_decoding import (
            DigitsOnlyLogitsProcessor,
            parse_grasp_output
        )

        # Format messages
        messages = format_inference_messages(image, instruction)

        # Tokenize
        inputs = self.processor(messages, return_tensors='pt').to(self.model.device)

        # Setup generation
        gen_kwargs = {
            'max_new_tokens': self.config.get('generation_max_tokens', 20),
            'temperature': temperature,
            'top_p': self.config.get('generation_top_p', 0.9),
            'do_sample': temperature > 0,
        }

        # Add constrained decoding
        if use_constrained:
            logits_processor = [DigitsOnlyLogitsProcessor(self.processor.tokenizer)]
            gen_kwargs['logits_processor'] = logits_processor

        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract grasp from text
        bins = parse_grasp_output(generated_text)

        if bins is None:
            print(f"Warning: Failed to parse output: {generated_text}")
            # Return dummy grasp
            return {'cx': 320, 'cy': 240, 'theta': 0, 'w': 50, 'h': 100}

        # Decode bins to continuous
        grasp = self.quantizer.decode(bins)

        return grasp, generated_text
