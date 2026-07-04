#!/usr/bin/env python3
"""
Helper script to read config.yml for shell scripts.

Usage:
    python scripts/read_config.py <dataset>

Prints (one per line):
    1. train path
    2. val path
    3. test path
    4. k_values (space-separated)
    5. k_ablation results base dir
    6. default top_k
    7. default llm_model
"""

import sys
import os
import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_config.py <dataset>", file=sys.stderr)
        sys.exit(1)

    dataset = sys.argv[1]

    # Find config.yml relative to this script (one level up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yml")

    if not os.path.exists(config_path):
        # Try current working directory
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

    k_values = cfg["experiments"]["k_ablation"]["k_values"]
    print(" ".join(str(k) for k in k_values))

    print(cfg["results"]["k_ablation"])
    print(cfg["defaults"]["top_k"])
    print(cfg["defaults"]["llm_model"])


if __name__ == "__main__":
    main()
