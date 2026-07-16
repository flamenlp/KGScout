#!/usr/bin/env python3
"""
Triplet Coverage Analysis for Ablation Results.

For each ablation variant, loads the trained model checkpoint + test data,
runs the model to select top-100 triplets, and computes:
  1. Answer entity presence (any answer entity is a node in the graph)
  2. Reasoning path existence (path from any q_entity to any a_entity)

- Model ablation variants: uses ablation-2/model_ablation.py architectures
- Reward ablation variants: uses src/model/path_ranker.py (PathRankingModel)

Results saved to: ./results/<ablation-type>/<variant>/triplet-metrics/

Usage:
    python run_retrieval_ablation.py                          # all variants
    python run_retrieval_ablation.py --mode model             # model ablations only
    python run_retrieval_ablation.py --mode reward            # reward ablations only
    python run_retrieval_ablation.py --mode model --experiments no-ppr no-gate
"""

import os
import sys
import json
import re
import argparse
import logging
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Required for torch.load() to unpickle datasets saved from notebooks
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.utils.triplet_selector import select_triplets_kgscout, _extract_metadata_from_batch
from src.utils.metrics import compute_answer_coverage, compute_path_coverage
from src.model.path_ranker import PathRankingModel

# Import model architectures from ablation-2 modules (hyphenated dir requires importlib)
import importlib
_ablation2_model = importlib.import_module("ablation-2.model_ablation")
ReversedOriginal = _ablation2_model.ReversedOriginal
ReversedNoPPR = _ablation2_model.ReversedNoPPR
ReversedNoRT = _ablation2_model.ReversedNoRT
ReversedNoTT = _ablation2_model.ReversedNoTT
ReversedNoGate = _ablation2_model.ReversedNoGate
ReversedNoRA = _ablation2_model.ReversedNoRA
ReversedNoTA = _ablation2_model.ReversedNoTA

# ============================================================================
# CONFIGURATION
# ============================================================================
from src.utils.dir_config import (
    load_config, get_dataset_paths, get_results_dir,
    get_model_variants, get_reward_variants, get_defaults, get_log_path,
)

_config = load_config()
_defaults = get_defaults(_config)

MODEL_ABLATION_DIR = get_results_dir("ablation2", "model_ablation", _config)
REWARD_ABLATION_DIR = get_results_dir("ablation2", "reward_ablation", _config)
_, _, TEST_DATASET_PATH = get_dataset_paths("cwq", _config)
TOP_K = _defaults["top_k"]
SAMPLE_K = _defaults["sample_k"]
LOG_FILE = get_log_path("triplet_analysis", _config)

MODEL_VARIANTS = get_model_variants(_config)
REWARD_VARIANTS = get_reward_variants(_config)

# Variant → model class mapping (ablation-2 reversed attention variants)
MODEL_CLASS_MAP = {
    "no-ppr": ReversedNoPPR,
    "no-rt": ReversedNoRT,
    "no-tt": ReversedNoTT,
    "no-gate": ReversedNoGate,
    "no-ra": ReversedNoRA,
    "no-ta": ReversedNoTA,
}
# Reward ablation uses the standard PathRankingModel from src/
REWARD_MODEL_CLASS = PathRankingModel

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("ablation.triplet_analysis")


def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


# ============================================================================
# CHECKPOINT FINDER
# ============================================================================

def _find_last_checkpoint(train_dir: str) -> Optional[str]:
    """Find the last best checkpoint directory (counts down from epoch 30)."""
    if not os.path.exists(train_dir):
        return None
    # Count down from epoch 30 and return the first best checkpoint that exists
    for epoch in range(30, 0, -1):
        candidate = os.path.join(train_dir, f"checkpoint_best_epoch_{epoch}")
        if os.path.isdir(candidate):
            return candidate
    return None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_ablation_model(model_class, checkpoint_path: str, device: str):
    """
    Load an ablation model from checkpoint (save_pretrained format).

    For model ablation variants (ablation-2): component-level state dicts.
    For reward ablation (PathRankingModel from src): uses from_pretrained.
    """
    if model_class == PathRankingModel:
        # Reward ablation uses src PathRankingModel.from_pretrained
        ckpt_dir = os.path.dirname(checkpoint_path)
        model = PathRankingModel.from_pretrained(ckpt_dir, device=device)
    else:
        # Ablation-2 model variants: component-level state dicts
        model = model_class(device=device)
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        for key, val in state.items():
            if key in ('temperature', 'baseline'):
                getattr(model, key).data = val
            elif hasattr(model, key):
                getattr(model, key).load_state_dict(val)
            else:
                logger.warning(f"  Unexpected key: '{key}' in checkpoint")

    model.to(device)
    model.eval()
    return model


# ============================================================================
# COVERAGE EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_coverage(dataloader: DataLoader, model, device: str, top_k: int):
    """
    Run model on test data, select top_k triplets, compute coverage metrics.

    Uses select_triplets_kgscout from src.utils.triplet_selector and
    compute_answer_coverage / compute_path_coverage from src.utils.metrics.

    Returns (aggregate_metrics_dict, per_sample_list).
    """
    model.eval()
    ans_present_count = 0
    reasoning_path_count = 0
    total_count = 0
    per_sample_results = []

    for i, batch in enumerate(tqdm(dataloader, desc="    Coverage analysis")):
        try:
            # Use select_triplets_kgscout (handles DataLoader batch format)
            selected_triplets = select_triplets_kgscout(model, batch, top_k, device)

            if len(selected_triplets) == 0:
                continue

            # Extract metadata
            meta = _extract_metadata_from_batch(batch)
            q_ents = [e.lower() for e in meta["q_entity"]] if meta["q_entity"] else []
            a_ents = [e.lower() for e in meta["a_entity"]] if meta["a_entity"] else []

            if not a_ents:
                continue

            # Use metrics from src.utils.metrics
            ans_present = compute_answer_coverage(selected_triplets, a_ents)
            path_found = compute_path_coverage(selected_triplets, q_ents, a_ents)

            if ans_present:
                ans_present_count += 1
            if path_found:
                reasoning_path_count += 1

            total_count += 1

            per_sample_results.append({
                "id": i,
                "question": meta["question"],
                "answer_entity_present": ans_present,
                "reasoning_path_exists": path_found,
                "answer_entities": a_ents,
                "question_entities": q_ents,
                "num_selected_triplets": len(selected_triplets),
            })

        except Exception:
            continue

    if total_count == 0:
        return {"total_samples": 0}, []

    metrics = {
        "total_samples": total_count,
        "answer_entity_present": ans_present_count,
        "answer_entity_present_pct": round(ans_present_count / total_count * 100, 2),
        "reasoning_path_exists": reasoning_path_count,
        "reasoning_path_exists_pct": round(reasoning_path_count / total_count * 100, 2),
    }
    return metrics, per_sample_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Triplet coverage analysis for ablation results")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"])
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--test-data", type=str, default=TEST_DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--sample-k", type=int, default=SAMPLE_K)
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("TRIPLET COVERAGE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Sample-K: {args.sample_k}")
    logger.info(f"Test dataset: {args.test_data}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test dataset and create DataLoader (same pattern as ablation-2)
    logger.info("Loading test dataset...")
    test_data = torch.load(args.test_data, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)} samples")

    test_sampled = SampledJointTrainingDataset(test_data, k=args.sample_k)
    test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)

    # Build task list
    tasks = []
    if args.mode in ("all", "model"):
        variants = args.experiments if (args.experiments and args.mode == "model") else MODEL_VARIANTS
        for v in variants:
            tasks.append(("model-ablation", MODEL_ABLATION_DIR, v, MODEL_CLASS_MAP[v]))
    if args.mode in ("all", "reward"):
        variants = args.experiments if (args.experiments and args.mode == "reward") else REWARD_VARIANTS
        for v in variants:
            tasks.append(("reward-ablation", REWARD_ABLATION_DIR, v, REWARD_MODEL_CLASS))

    for ablation_type, base_dir, variant, model_class in tasks:
        logger.info(f"{'='*60}")
        logger.info(f"  Variant: {variant} ({ablation_type})")

        train_dir = os.path.join(base_dir, variant, "model", "trained")
        result_dir = os.path.join(base_dir, variant, "triplet-metrics")

        # Find checkpoint
        checkpoint_dir = _find_last_checkpoint(train_dir)
        if checkpoint_dir is None:
            logger.warning(f"  No checkpoint found in {train_dir}, skipping.")
            continue

        ckpt_path = os.path.join(checkpoint_dir, "path_ranker.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  path_ranker.pt not found in {checkpoint_dir}, skipping.")
            continue

        logger.info(f"  Checkpoint: {checkpoint_dir}")

        # Load model
        model = load_ablation_model(model_class, ckpt_path, device_str)

        # Run coverage analysis
        metrics, per_sample = evaluate_coverage(test_loader, model, device_str, args.top_k)

        # Save results
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "coverage_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(result_dir, "per_sample_metrics.jsonl"), "w") as f:
            for record in per_sample:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  Results: {metrics}")
        logger.info(f"  Saved to: {result_dir}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL COVERAGE ANALYSIS COMPLETE. Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
