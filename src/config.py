"""Configuration loading: YAML + CLI argument merging."""

import argparse
import yaml
from types import SimpleNamespace


def load_config() -> SimpleNamespace:
    """Load config from YAML file, override with CLI args.

    CLI supports:
        --config PATH     YAML config file (required)
        --smoke           Use smoke_steps instead of max_steps
    """
    parser = argparse.ArgumentParser(description="MiniLLM-Lab training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with reduced steps")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["smoke"] = args.smoke
    cfg["max_steps"] = cfg.get("smoke_steps", 50) if args.smoke else cfg.get("max_steps", 500)

    return SimpleNamespace(**cfg)
