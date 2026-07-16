#!/usr/bin/env python3
"""
Find the best or last checkpoint directory inside a training directory.

Checkpoint naming convention (from src/training/trainer.py):
  - Best:  checkpoint_best_epoch_{N}/path_ranker.pt
  - Regular: checkpoint_epoch_{N}/path_ranker.pt

Strategy:
  - If PICK_LAST_CHECKPOINT is true: find the highest-epoch checkpoint_epoch_N
  - If PICK_LAST_CHECKPOINT is false (default): find the highest-epoch checkpoint_best_epoch_N

Counts backward from max_epochs (default 30) to 1.

Usage:
    python scripts/find_checkpoint.py <train_dir>
    python scripts/find_checkpoint.py <train_dir> --pick-last
    python scripts/find_checkpoint.py <train_dir> --max-epochs 50

Prints the path to path_ranker.pt on success, exits with code 1 on failure.
"""

import os
import sys
import argparse
import yaml


def find_checkpoint(train_dir: str, pick_last: bool = False, max_epochs: int = 30) -> str:
    """
    Find checkpoint directory by counting backward from max_epochs.

    Args:
        train_dir: Path to the training directory containing checkpoint_* subdirs
        pick_last: If True, pick the highest-epoch checkpoint regardless of best or not.
                   If False, only look for checkpoint_best_epoch_N (best validation).
        max_epochs: Maximum epoch number to search from (counts down to 1)

    Returns:
        Path to path_ranker.pt inside the found checkpoint directory

    Raises:
        FileNotFoundError: If no valid checkpoint is found
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if pick_last:
        # Pick the highest epoch checkpoint, whether it's best or regular
        for epoch in range(max_epochs, 0, -1):
            for prefix in ("checkpoint_best_epoch_", "checkpoint_epoch_"):
                candidate_dir = os.path.join(train_dir, f"{prefix}{epoch}")
                candidate_file = os.path.join(candidate_dir, "path_ranker.pt")
                if os.path.isdir(candidate_dir) and os.path.isfile(candidate_file):
                    return candidate_file
    else:
        # Pick the highest-epoch best checkpoint only
        for epoch in range(max_epochs, 0, -1):
            candidate_dir = os.path.join(train_dir, f"checkpoint_best_epoch_{epoch}")
            candidate_file = os.path.join(candidate_dir, "path_ranker.pt")
            if os.path.isdir(candidate_dir) and os.path.isfile(candidate_file):
                return candidate_file

    raise FileNotFoundError(
        f"No checkpoint found in {train_dir}.\n"
        f"Looked for: checkpoint_*epoch_N/path_ranker.pt (N from {max_epochs} down to 1)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find best or last checkpoint in a training directory"
    )
    parser.add_argument(
        "train_dir",
        type=str,
        help="Path to training directory containing checkpoint_* subdirs"
    )
    parser.add_argument(
        "--pick-last",
        action="store_true",
        default=None,
        help="Pick last epoch checkpoint instead of best (overrides config.yml)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum epoch to search from (default: 30)"
    )
    args = parser.parse_args()

    # Determine pick_last from args or config.yml
    pick_last = args.pick_last
    if pick_last is None:
        # Read from config.yml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, "config.yml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            pick_last = cfg.get("defaults", {}).get("pick_last_checkpoint", False)
        else:
            pick_last = False

    try:
        ckpt_path = find_checkpoint(args.train_dir, pick_last, args.max_epochs)
        print(ckpt_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
