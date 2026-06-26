"""Model loading utilities with cpu/cuda device switching."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device(config_device: str | None = None) -> torch.device:
    """Determine device: config override > auto-detect (cuda > mps > cpu)."""
    if config_device:
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    """Load a causal LM and its tokenizer from HuggingFace hub.

    Returns (model, tokenizer) with pad_token set to eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
