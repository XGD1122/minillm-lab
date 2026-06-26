"""Experiment A: Mini Pretrain — Causal LM continued training on WikiText-2."""

import os
import copy
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.config import load_config
from src.utils.model import load_model_and_tokenizer, get_device
from src.utils.logging import TrainingLogger
from src.data.causal_lm_dataset import get_wikitext_data
from src.evaluation.perplexity import compute_perplexity
from src.evaluation.generation_compare import generate_text


EVAL_PROMPTS = [
    "The weather today is",
    "The history of artificial intelligence",
    "In recent years, scientists have discovered",
    "The most important skill for a programmer is",
    "Once upon a time in a distant land",
]


def main():
    cfg = load_config()
    device = get_device(cfg.device)
    mode = "smoke" if cfg.smoke else "full"
    print(f"[Mini Pretrain] device={device}, mode={mode}, max_steps={cfg.max_steps}")

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model.to(device)
    model_before = copy.deepcopy(model).cpu()  # snapshot for generation comparison

    # Data
    train_ds, eval_ds = get_wikitext_data(tokenizer, block_size=cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps * cfg.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)

    # Logger
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TrainingLogger(cfg.log_dir)

    # Training loop
    model.train()
    global_step = 0
    accum_loss = 0.0
    train_loss = 0.0

    print(f"\n{'='*50}")
    print(f"Training — before generation samples:")
    for p in EVAL_PROMPTS[:2]:
        print(f"  Prompt: {p}")
        print(f"  Output: {generate_text(model_before, tokenizer, p, device=device)}")
    print(f"{'='*50}\n")

    for step, batch in enumerate(train_loader):
        if global_step >= cfg.max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
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

            if global_step % 10 == 0:
                print(f"  step {global_step:4d}/{cfg.max_steps} | loss: {train_loss / global_step:.4f}")

            # Evaluation
            if global_step % cfg.eval_steps == 0:
                ppl = compute_perplexity(model, eval_loader, device)
                logger.log_scalar("perplexity/eval", ppl, global_step)
                print(f"  --- eval @ step {global_step}: perplexity={ppl:.2f} ---")

            # Save checkpoint
            if global_step % cfg.save_steps == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  --- saved checkpoint to {ckpt_path} ---")

    # Final save
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.close()

    # After training generation comparison
    model.cpu()
    print(f"\n{'='*50}")
    print(f"Training complete. Final loss: {train_loss / global_step:.4f}")
    print(f"\nBefore vs After generation comparison:")
    for p in EVAL_PROMPTS:
        before = generate_text(model_before, tokenizer, p, device=torch.device("cpu"))
        after = generate_text(model, tokenizer, p, device=torch.device("cpu"))
        print(f"  Prompt: {p}")
        print(f"  Before: {before}")
        print(f"  After:  {after}")
        print()
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
