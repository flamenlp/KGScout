#!/usr/bin/env python3
"""
Helper script to read MetaQA generalization config from config.yml.

Usage:
    python scripts/read_metaqa_config.py <hop>

Prints (one per line):
    1. kb_path
    2. qa_train path
    3. qa_dev path
    4. qa_test path
    5. processed_dir
    6. embedding_model
"""

import sys
import os
import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_metaqa_config.py <hop>", file=sys.stderr)
        sys.exit(1)

    hop = sys.argv[1]

    # Find config.yml relative to this script (one level up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yml")

    if not os.path.exists(config_path):
        config_path = "config.yml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    g = cfg["generalization"]

    train_key = f"qa_train_{hop}hop"
    dev_key = f"qa_dev_{hop}hop"
    test_key = f"qa_test_{hop}hop"

    if train_key not in g:
        print(f"ERROR: Key '{train_key}' not found in config.yml generalization section.", file=sys.stderr)
        sys.exit(1)

    print(g["kb_path"])
    print(g[train_key])
    print(g[dev_key])
    print(g[test_key])
    print(g["processed_dir"])
    print(g["embedding_model"])


if __name__ == "__main__":
    main()
