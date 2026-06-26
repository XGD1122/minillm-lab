"""Tests for data loading and formatting."""

import json
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sft_data_format():
    """Verify SFT JSON has the correct structure."""
    data_path = "data/samples/sft_samples.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "SFT data should be a list"
    assert len(data) > 0, "SFT data should not be empty"

    for item in data:
        assert "instruction" in item, f"Missing 'instruction' in {item}"
        assert "output" in item, f"Missing 'output' in {item}"
        # 'input' is optional
        assert isinstance(item["instruction"], str)
        assert isinstance(item["output"], str)


def test_dpo_data_format():
    """Verify DPO JSON has the correct structure."""
    data_path = "data/samples/dpo_samples.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "DPO data should be a list"
    assert len(data) > 0, "DPO data should not be empty"

    for item in data:
        assert "prompt" in item, f"Missing 'prompt' in {item}"
        assert "chosen" in item, f"Missing 'chosen' in {item}"
        assert "rejected" in item, f"Missing 'rejected' in {item}"
        assert isinstance(item["prompt"], str)
        assert isinstance(item["chosen"], str)
        assert isinstance(item["rejected"], str)
        assert len(item["chosen"]) > 0, "chosen should not be empty"
        assert len(item["rejected"]) > 0, "rejected should not be empty"


def test_sft_dataset_instantiation():
    """Test SFTDataset can be created with a tokenizer."""
    from transformers import AutoTokenizer
    from src.data.sft_dataset import SFTDataset

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = SFTDataset("data/samples/sft_samples.json", tokenizer, max_length=128)
    assert len(ds) > 0

    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 128
    assert item["labels"].shape[0] == 128


def test_dpo_dataset_instantiation():
    """Test DPODataset can be created with a tokenizer."""
    from transformers import AutoTokenizer
    from src.data.dpo_dataset import DPODataset

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = DPODataset("data/samples/dpo_samples.json", tokenizer, max_length=128)
    assert len(ds) > 0

    item = ds[0]
    for key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids",
                "chosen_labels", "rejected_labels"]:
        assert key in item, f"Missing key: {key}"
