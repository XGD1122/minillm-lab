"""Generation comparison: before/after training text generation."""

import torch


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, device=None) -> str:
    """Generate text from a prompt using the model."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device or model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def compare_generations(
    model_before,
    model_after,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    device=None,
) -> list[dict]:
    """Generate text for each prompt with both models and return comparison.

    Returns list of dicts with keys: prompt, before, after.
    """
    results = []
    for prompt in prompts:
        before = generate_text(model_before, tokenizer, prompt, max_new_tokens, device)
        after = generate_text(model_after, tokenizer, prompt, max_new_tokens, device)
        results.append({"prompt": prompt, "before": before, "after": after})
    return results
