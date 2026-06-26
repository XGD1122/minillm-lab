"""Experiment C: DPO — Direct Preference Optimization.

Implements DPO loss manually to demonstrate understanding of the algorithm.
DPO loss = -log(sigma(beta * (log_pi_chosen - log_pi_rejected) - beta * (log_ref_chosen - log_ref_rejected)))
"""

import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.config import load_config
from src.utils.model import load_model_and_tokenizer, get_device
from src.utils.logging import TrainingLogger
from src.data.dpo_dataset import DPODataset
from src.evaluation.generation_compare import generate_text


def compute_dpo_loss(
    model_chosen_logps: torch.Tensor,
    model_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and rewards.

    Args:
        model_chosen_logps: log-probs of chosen response under policy model
        model_rejected_logps: log-probs of rejected response under policy model
        ref_chosen_logps: log-probs of chosen response under reference model
        ref_rejected_logps: log-probs of rejected response under reference model
        beta: temperature parameter

    Returns:
        (loss, chosen_reward, rejected_reward)
    """
    model_logratios = model_chosen_logps - model_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = model_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    chosen_reward = beta * (model_chosen_logps - ref_chosen_logps).detach().mean()
    rejected_reward = beta * (model_rejected_logps - ref_rejected_logps).detach().mean()

    return loss, chosen_reward, rejected_reward


def get_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities for the labeled (non -100) tokens.

    Returns average log-prob per sequence in the batch.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log-probs of the correct tokens
    per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding (-100)
    mask = (shift_labels != -100).float()
    seq_logps = (per_token_logps * mask).sum(dim=1)
    seq_lengths = mask.sum(dim=1).clamp(min=1)
    return seq_logps / seq_lengths


def main():
    cfg = load_config()
    device = get_device(cfg.device)
    mode = "smoke" if cfg.smoke else "full"
    print(f"[DPO] device={device}, mode={mode}, max_steps={cfg.max_steps}")

    # Load policy model (the one being trained)
    if cfg.sft_model_path and os.path.exists(cfg.sft_model_path):
        print(f"  Loading SFT model from {cfg.sft_model_path}")
        model, tokenizer = load_model_and_tokenizer(cfg.sft_model_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg.model)
    model.to(device)

    # Load reference model (frozen copy)
    if cfg.sft_model_path and os.path.exists(cfg.sft_model_path):
        ref_model, _ = load_model_and_tokenizer(cfg.sft_model_path)
    else:
        ref_model, _ = load_model_and_tokenizer(cfg.model)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model_before = copy.deepcopy(model).cpu()

    # Data
    train_ds = DPODataset(cfg.data_path, tokenizer, max_length=cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps * cfg.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)

    # Logger
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TrainingLogger(cfg.log_dir)

    # Sample prompts for before/after comparison
    eval_prompts = [item["prompt"] for item in train_ds.data[:3]]

    model.train()
    global_step = 0
    accum_loss = 0.0
    train_loss = 0.0

    print(f"\n{'='*50}")
    print(f"DPO Training — before generation samples:")
    for p in eval_prompts[:2]:
        print(f"  Prompt: {p}")
        print(f"  Output: {generate_text(model_before, tokenizer, p, device=device)}")
    print(f"{'='*50}\n")

    for step, batch in enumerate(train_loader):
        if global_step >= cfg.max_steps:
            break

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        # Compute log-probs for policy model
        model_chosen_logps = get_logprobs(model, chosen_ids, chosen_mask, chosen_labels)
        model_rejected_logps = get_logprobs(model, rejected_ids, rejected_mask, rejected_labels)

        # Compute log-probs for reference model (no grad)
        with torch.no_grad():
            ref_chosen_logps = get_logprobs(ref_model, chosen_ids, chosen_mask, chosen_labels)
            ref_rejected_logps = get_logprobs(ref_model, rejected_ids, rejected_mask, rejected_labels)

        loss, chosen_reward, rejected_reward = compute_dpo_loss(
            model_chosen_logps, model_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=cfg.beta,
        )
        loss = loss / cfg.gradient_accumulation_steps
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            train_loss += accum_loss
            accum_loss = 0.0

            logger.log_scalar("loss/train", train_loss / global_step, global_step)
            logger.log_scalar("chosen_reward", chosen_reward.item(), global_step)
            logger.log_scalar("rejected_reward", rejected_reward.item(), global_step)
            logger.log_scalar("reward_margin", (chosen_reward - rejected_reward).item(), global_step)

            if global_step % 10 == 0:
                print(
                    f"  step {global_step:4d}/{cfg.max_steps} | loss: {train_loss / global_step:.4f}"
                    f" | chosen_r: {chosen_reward.item():.3f} | rejected_r: {rejected_reward.item():.3f}"
                )

            if global_step % cfg.save_steps == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  --- saved checkpoint to {ckpt_path} ---")

    # Final save
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.close()

    # After training: generation comparison
    model.cpu()
    ref_model.cpu()
    print(f"\n{'='*50}")
    print(f"DPO Training complete. Final loss: {train_loss / global_step:.4f}")
    print(f"\nBefore vs After generation comparison:")
    for p in eval_prompts:
        before = generate_text(model_before, tokenizer, p, device=torch.device("cpu"), max_new_tokens=50)
        after = generate_text(model, tokenizer, p, device=torch.device("cpu"), max_new_tokens=50)
        print(f"  Prompt: {p}")
        print(f"  Before: {before}")
        print(f"  After:  {after}")
        print()
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
