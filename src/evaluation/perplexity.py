"""Perplexity computation."""

import math
import torch
from torch.utils.data import DataLoader


def compute_perplexity(model, dataloader: DataLoader, device: torch.device) -> float:
    """Compute perplexity on an evaluation dataset.

    PPL = exp(average cross-entropy loss).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
