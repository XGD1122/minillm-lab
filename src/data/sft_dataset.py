"""SFT dataset: instruction formatting and tokenization with response-only loss."""

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Alpaca-style instruction template
PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
NO_INPUT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def _format_example(item: dict) -> tuple[str, str]:
    """Format an instruction example into (prompt, full_text).

    prompt = everything before the response
    full_text = prompt + response (with EOS)
    """
    instruction = item["instruction"]
    inp = item.get("input", "")
    output = item["output"]

    if inp:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    full_text = prompt + output
    return prompt, full_text


class SFTDataset(Dataset):
    """Tokenizes instruction data with labels masked for the prompt portion."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, full_text = _format_example(item)

        # Tokenize prompt and full text separately
        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        full_tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length)["input_ids"]

        prompt_len = len(prompt_tokens)
        full_len = len(full_tokens)

        # Pad full_tokens to max_length
        input_ids = full_tokens + [self.tokenizer.eos_token_id] if len(full_tokens) < self.max_length else full_tokens[:self.max_length]
        # Pad
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # Labels: -100 for prompt portion and padding, keep response tokens
        response_len = min(full_len - prompt_len, self.max_length - prompt_len)
        labels = (
            [-100] * prompt_len
            + full_tokens[prompt_len : prompt_len + response_len]
        )
        if len(labels) < self.max_length:
            labels = labels + [-100] * (self.max_length - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
