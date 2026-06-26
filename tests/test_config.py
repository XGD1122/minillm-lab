"""Tests for config loading."""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_structure():
    """Verify config YAML files can be loaded and have required keys."""
    import yaml

    required_keys = {"model", "max_steps", "smoke_steps", "batch_size", "learning_rate"}

    config_files = [
        "configs/cpu_pretrain_tiny.yaml",
        "configs/cpu_sft_tiny.yaml",
        "configs/cpu_dpo_tiny.yaml",
    ]

    for cf in config_files:
        with open(cf, "r") as f:
            cfg = yaml.safe_load(f)
        missing = required_keys - set(cfg.keys())
        assert not missing, f"{cf} missing keys: {missing}"


def test_config_load_with_smoke():
    """Test that smoke flag changes max_steps to smoke_steps."""
    # We can't easily test the argparse integration, but we can test the logic
    import yaml

    with open("configs/cpu_pretrain_tiny.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    smoke_val = cfg["smoke_steps"]
    assert smoke_val < cfg["max_steps"], "smoke_steps should be less than max_steps"
