#!/usr/bin/env python3
"""
Helper script to read config.yml for shell scripts.

Usage:
    python scripts/read_config.py <dataset>
    python scripts/read_config.py metaqa <hop>    (e.g., metaqa 2)

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
        print("Usage: python scripts/read_config.py <dataset> [hop]", file=sys.stderr)
        sys.exit(1)

    dataset = sys.argv[1]
    hop = sys.argv[2] if len(sys.argv) > 2 else None

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

    # For metaqa, resolve hop-specific paths dynamically
    if dataset == "metaqa" and hop:
        gen = cfg.get("generalization", {})
        processed_dir = gen.get("processed_dir", "data/metaqa/processed")
        train_path = os.path.join(processed_dir, f"metaqa-{hop}hop-train.pt")
        val_path = os.path.join(processed_dir, f"metaqa-{hop}hop-val.pt")
        test_path = os.path.join(processed_dir, f"metaqa-{hop}hop-test.pt")
        print(train_path)
        print(val_path)
        print(test_path)
    elif dataset in datasets:
        ds = datasets[dataset]
        print(ds["train"])
        print(ds["val"])
        print(ds["test"])
    else:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(datasets.keys())}", file=sys.stderr)
        sys.exit(1)

    k_values = cfg["experiments"]["k_ablation"]["k_values"]
    print(" ".join(str(k) for k in k_values))

    print(cfg["results"]["k_ablation"])
    print(cfg["defaults"]["top_k"])
    print(cfg["defaults"]["llm_model"])


if __name__ == "__main__":
    main()
