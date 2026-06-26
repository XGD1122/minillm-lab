"""DPO dataset: preference pair (chosen/rejected) tokenization."""

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DPODataset(Dataset):
    """Tokenizes (prompt, chosen, rejected) triples for DPO training.

    Each item returns:
        prompt_input_ids, prompt_attention_mask
        chosen_input_ids, chosen_attention_mask, chosen_labels
        rejected_input_ids, rejected_attention_mask, rejected_labels
    """

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize each component
        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, padding="max_length",
            return_tensors="pt",
        )

        chosen_text = prompt + " " + chosen
        rejected_text = prompt + " " + rejected

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length, padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length, padding="max_length",
            return_tensors="pt",
        )

        prompt_len = prompt_enc["attention_mask"].sum().item()

        # Build labels: prompt portion = -100
        chosen_labels = chosen_enc["input_ids"].clone()
        chosen_labels[:, :prompt_len] = -100

        rejected_labels = rejected_enc["input_ids"].clone()
        rejected_labels[:, :prompt_len] = -100

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        }
