#!/usr/bin/env python3
"""
Main Ablation Study Runner for KGScout.

Orchestrates all 12 ablation experiments:
  - 6 Model Architecture Ablations (saved to ./results/model-ablation/)
  - 6 Reward Function Ablations (saved to ./results/reward-ablation/)

Usage:
    # Run all experiments
    python run_ablation.py

    # Run only model ablations
    python run_ablation.py --mode model

    # Run only reward ablations
    python run_ablation.py --mode reward

    # Run specific model experiments
    python run_ablation.py --mode model --experiments no-ppr no-gate

    # Run specific reward experiments
    python run_ablation.py --mode reward --experiments no_pres only_conn

Directory Structure of Results:
    ./results/
    ├── model-ablation/
    │   ├── no-ppr/
    │   ├── no-rt/
    │   ├── no-tt/
    │   ├── no-gate/
    │   ├── no-ra/
    │   └── no-ta/
    └── reward-ablation/
        ├── no_pres/
        ├── no_conn/
        ├── no_path/
        ├── only_pres/
        ├── only_conn/
        └── only_cov/

Training Configuration (same for all experiments):
    Pretraining: n=500, 5 epochs, lr=1e-4, gradient_accumulation=8
    Training:    k=1000, sample_k=100, 30 epochs, lr=1e-4, gradient_accumulation=32,
                 cosine scheduler, warmup=100, early_stopping_patience=3
    Inference:   k=1000, top_k=100
"""

import argparse
import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation.model_ablation import run_model_ablation
from ablation.reward_ablation import run_reward_ablation

# ============================================================================
# HARDCODED DATA PATHS - Update these to match your environment
# ============================================================================
TRAIN_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/train/train_jointrainer_path_dataset_v3_ppr.pt"
VAL_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
TEST_DATA_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"

MODEL_OUTPUT_DIR = "./results/model-ablation"
REWARD_OUTPUT_DIR = "./results/reward-ablation"
# ============================================================================


def setup_logging(log_file="ablation_log.txt"):
    """
    Configure logging to write to both file and stdout.
    All output goes through the logger so it's captured in the log file
    even when running in a detached screen/tmux session.
    """
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler - captures everything to ablation_log.txt
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream handler - also prints to stdout (visible in screen if attached)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Run KGScout ablation studies (model architecture + reward function)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"],
                        help="Which ablation set to run: 'all' (default), 'model', or 'reward'")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (depends on --mode)")
    args = parser.parse_args()

    # Setup logging - all output goes to logs/ablation.log + stdout
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ablation.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)

    # Validate dataset paths
    for path_name, path_val in [("train_data", TRAIN_DATA_PATH),
                                 ("val_data", VAL_DATA_PATH),
                                 ("test_data", TEST_DATA_PATH)]:
        if not os.path.exists(path_val):
            logger.error(f"{path_name} file not found: {path_val}")
            sys.exit(1)

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("KGScout ABLATION STUDY RUNNER")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Train data: {TRAIN_DATA_PATH}")
    logger.info(f"Val data:   {VAL_DATA_PATH}")
    logger.info(f"Test data:  {TEST_DATA_PATH}")
    logger.info(f"Log file:   {log_file}")
    if args.experiments:
        logger.info(f"Experiments: {args.experiments}")

    # Run model ablations
    if args.mode in ("all", "model"):
        model_start = time.time()
        run_model_ablation(
            train_dataset_path=TRAIN_DATA_PATH,
            val_dataset_path=VAL_DATA_PATH,
            test_dataset_path=TEST_DATA_PATH,
            output_base_dir=MODEL_OUTPUT_DIR,
            experiments=args.experiments if args.mode == "model" else None
        )
        model_elapsed = time.time() - model_start
        logger.info(f"Model ablation total time: {model_elapsed / 3600:.2f} hours")

    # Run reward ablations
    if args.mode in ("all", "reward"):
        reward_start = time.time()
        run_reward_ablation(
            train_dataset_path=TRAIN_DATA_PATH,
            val_dataset_path=VAL_DATA_PATH,
            test_dataset_path=TEST_DATA_PATH,
            output_base_dir=REWARD_OUTPUT_DIR,
            experiments=args.experiments if args.mode == "reward" else None
        )
        reward_elapsed = time.time() - reward_start
        logger.info(f"Reward ablation total time: {reward_elapsed / 3600:.2f} hours")

    total_elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL ABLATION STUDIES COMPLETE")
    logger.info(f"Total time: {total_elapsed / 3600:.2f} hours")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
