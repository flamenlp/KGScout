#!/usr/bin/env python3
"""
Ablation-2 Runner: Reversed Attention Architecture Ablation Studies.

Runs model architecture ablations (6) and reward function ablations (6)
on the reversed attention model.

Usage:
    python run_ablation_2.py                              # all 12 experiments
    python run_ablation_2.py --mode model                 # 6 model ablations
    python run_ablation_2.py --mode reward                # 6 reward ablations
    python run_ablation_2.py --mode model --experiments no-ppr no-gate
"""

import argparse
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to False to skip training and only run inference from saved checkpoints
IS_TRAIN_REQUIRED = False

# ============================================================================
# HARDCODED PATHS
# ============================================================================
TRAIN_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/train/train_jointrainer_path_dataset_v3_ppr.pt"
VAL_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
TEST_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"

MODEL_OUTPUT_DIR = "./results/ablation-2/model-ablation"
REWARD_OUTPUT_DIR = "./results/ablation-2/reward-ablation"
LOG_FILE = "ablation2_log.txt"


def setup_logging(log_file):
    logger = logging.getLogger("ablation2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Ablation-2: Reversed Attention ablation studies")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"])
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    logger = setup_logging(log_file)

    # Validate paths
    for name, path in [("train", TRAIN_DATA_PATH), ("val", VAL_DATA_PATH), ("test", TEST_DATA_PATH)]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}"); sys.exit(1)

    start = time.time()
    logger.info("=" * 70)
    logger.info("ABLATION-2: REVERSED ATTENTION ABLATION RUNNER")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Log: {log_file}")

    if args.mode in ("all", "model"):
        from importlib import import_module
        mod = import_module("ablation-2.model_ablation")
        t0 = time.time()
        mod.run_model_ablation(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODEL_OUTPUT_DIR, args.experiments if args.mode == "model" else None, train=IS_TRAIN_REQUIRED)
        logger.info(f"Model ablation time: {(time.time()-t0)/3600:.2f}h")

    if args.mode in ("all", "reward"):
        from importlib import import_module
        mod = import_module("ablation-2.reward_ablation")
        t0 = time.time()
        mod.run_reward_ablation(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, REWARD_OUTPUT_DIR, args.experiments if args.mode == "reward" else None, train=IS_TRAIN_REQUIRED)
        logger.info(f"Reward ablation time: {(time.time()-t0)/3600:.2f}h")

    logger.info("=" * 70)
    logger.info(f"ALL DONE. Total: {(time.time()-start)/3600:.2f}h")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
