#!/usr/bin/env python3
"""
Helper script to read ablation configuration from config.yml for shell scripts.

Usage:
    python scripts/read_ablation_config.py <dataset>

Prints (one per line):
    1. train path
    2. val path
    3. test path
    4. default top_k
    5. default llm_model
    6. model_variants (space-separated)
    7. reward_variants (space-separated)
    8. model_ablation results base dir (with dataset name)
    9. reward_ablation results base dir (with dataset name)
    10. num_epochs
    11. early_stopping_patience
"""

import sys
import os
import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_ablation_config.py <dataset>", file=sys.stderr)
        sys.exit(1)

    dataset = sys.argv[1]

    # Find config.yml relative to this script (one level up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yml")

    if not os.path.exists(config_path):
        config_path = "config.yml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datasets = cfg["datasets"]

    if dataset not in datasets:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(datasets.keys())}", file=sys.stderr)
        sys.exit(1)

    ds = datasets[dataset]
    print(ds["train"])
    print(ds["val"])
    print(ds["test"])
    print(cfg["defaults"]["top_k"])
    print(cfg["defaults"]["llm_model"])

    model_variants = cfg["experiments"]["model_variants"]
    print(" ".join(model_variants))

    reward_variants = cfg["experiments"]["reward_variants"]
    print(" ".join(reward_variants))

    # Results base dirs (append dataset name for per-dataset output)
    model_base = cfg["results"]["ablation2"]["model_ablation"].replace("cwq-", f"{dataset}-")
    reward_base = cfg["results"]["ablation2"]["reward_ablation"].replace("cwq-", f"{dataset}-")

    # If the paths already contain the dataset, use them as-is
    # Otherwise construct dataset-specific paths
    if dataset not in model_base:
        model_base = f"./results/ablation-2/{dataset}-model-ablation"
    if dataset not in reward_base:
        reward_base = f"./results/ablation-2/{dataset}-reward-ablation"

    print(model_base)
    print(reward_base)
    print(cfg["defaults"]["num_epochs"])
    print(cfg["defaults"]["early_stopping_patience"])


if __name__ == "__main__":
    main()
