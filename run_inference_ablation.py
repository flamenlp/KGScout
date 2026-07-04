#!/usr/bin/env python3
"""
LLaMA Inference on Ablation Results.

Loads LLM once, iterates over all ablation variants' selected_triplets.json files,
generates answers using top-k triplets with the latest prompt (format_prompt from
src/utils/llm_inference.py), and computes QA metrics.

Results saved to: ./results/<ablation-type>/<variant>/llama-inference/

Usage:
    python run_inference_ablation.py                          # all variants
    python run_inference_ablation.py --mode model             # model ablations only
    python run_inference_ablation.py --mode reward            # reward ablations only
    python run_inference_ablation.py --mode model --experiments no-ppr no-gate
"""

import os
import sys
import json
import argparse
import logging
import time

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_inference import load_llm_model, format_prompt, run_llm_inference
from src.utils.metrics import (
    extract_predictions_from_response,
    compute_hit_score,
    compute_hit_at_1,
    compute_precision,
    compute_recall,
    compute_f1_score,
    should_use_double_check,
    preprocess_date_answers,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
from src.utils.dir_config import (
    load_config, get_results_dir, get_model_variants,
    get_reward_variants, get_defaults, get_log_path,
)

_config = load_config()
_defaults = get_defaults(_config)

LLM_MODEL_NAME = _defaults["llm_model"]
MODEL_ABLATION_DIR = get_results_dir("ablation2", "model_ablation", _config)
REWARD_ABLATION_DIR = get_results_dir("ablation2", "reward_ablation", _config)
TOP_K = _defaults["top_k"]
LOG_FILE = get_log_path("inference", _config)

MODEL_VARIANTS = get_model_variants(_config)
REWARD_VARIANTS = get_reward_variants(_config)

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("ablation.inference")


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
# EVALUATION
# ============================================================================

def evaluate_dataset(data, output_dir, llm_model, tokenizer, top_k):
    """
    Run LLM inference on selected triplets and compute QA metrics.

    Uses format_prompt (v5) from src/utils/llm_inference and metrics from src/utils/metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    hit_list, hit1_list, f1_list = [], [], []
    precision_list, recall_list = [], []
    detailed_results = []

    for i, sample in enumerate(tqdm(data, desc="    LLM Inference")):
        try:
            question = sample["question"]
            ground_truth = sample.get("answer", sample.get("a_entity", []))
            q_entity = sample.get("q_entity", [])
            triplets = sample.get("reranker", [])

            if not ground_truth or not triplets:
                continue

            # Ensure ground_truth is a flat list of strings
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            ground_truth = [str(a) for a in ground_truth if a]
            if not ground_truth:
                continue

            # Format prompt using latest v5 from src
            triplets_used = triplets[:top_k]
            prompt = format_prompt(question, triplets_used, topk=top_k, q_entity=q_entity)

            # Run LLM inference
            raw_prediction = run_llm_inference(llm_model, tokenizer, prompt)
            prediction = extract_predictions_from_response(raw_prediction)
            prediction = [s for s in prediction if s != "" and s is not None]

            # Preprocess answers
            answer = preprocess_date_answers(question, ground_truth)
            double_check = should_use_double_check(question)

            # Compute metrics
            prec, _, num_pred = compute_precision(prediction, answer, double_check)
            rec, _, num_answer = compute_recall(prediction, answer, double_check)
            f1 = compute_f1_score(prec, rec)
            hit1 = compute_hit_at_1(prediction, answer, double_check)
            hit = compute_hit_score(prediction, answer, double_check)

            hit1_list.append(hit1)
            hit_list.append(hit)
            f1_list.append(f1)
            precision_list.append(prec)
            recall_list.append(rec)

            detailed_results.append({
                "id": i,
                "question": question,
                "prediction": prediction,
                "ground_truth": answer,
                "hit@1": hit1,
                "hit": hit,
                "f1": f1,
                "precision": prec,
                "recall": rec,
            })

        except Exception as e:
            logger.warning(f"  Error processing item {i}: {e}")
            continue

    if len(hit_list) == 0:
        logger.warning("  No valid predictions found!")
        return None

    # Aggregate metrics
    n = len(hit_list)
    metrics = {
        "hit": round(np.mean(hit_list) * 100, 2),
        "hit_at_1": round(np.mean(hit1_list) * 100, 2),
        "macro_f1": round(np.mean(f1_list) * 100, 2),
        "macro_precision": round(np.mean(precision_list) * 100, 2),
        "macro_recall": round(np.mean(recall_list) * 100, 2),
        "exact_match": round((np.array(f1_list) == 1).sum() / n * 100, 2),
        "totally_wrong": round((np.array(recall_list) == 0).sum() / n * 100, 2),
        "total_samples": n,
    }

    logger.info(f"    Hit: {metrics['hit']:.2f}%, Hit@1: {metrics['hit_at_1']:.2f}%, "
                f"F1: {metrics['macro_f1']:.2f}%, EM: {metrics['exact_match']:.2f}%")

    # Save results
    with open(os.path.join(output_dir, "llm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "llm_detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on ablation results")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"])
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL_NAME, choices=["llama", "qwen", "deepseek"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("ABLATION INFERENCE: LLM QA Evaluation")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"LLM: {args.llm_model}")
    logger.info(f"Top-K: {args.top_k}")

    # Load LLM model once (using src/utils/llm_inference)
    logger.info("Loading LLM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model, tokenizer = load_llm_model(args.llm_model, device)
    logger.info("  LLM model loaded.")

    # Build list of (base_dir, variant) to process
    tasks = []
    if args.mode in ("all", "model"):
        variants = args.experiments if (args.experiments and args.mode == "model") else MODEL_VARIANTS
        for v in variants:
            tasks.append((MODEL_ABLATION_DIR, v))
    if args.mode in ("all", "reward"):
        variants = args.experiments if (args.experiments and args.mode == "reward") else REWARD_VARIANTS
        for v in variants:
            tasks.append((REWARD_ABLATION_DIR, v))

    for base_dir, variant in tasks:
        input_path = os.path.join(base_dir, variant, "triplet-result", "selected_triplets.json")
        output_path = os.path.join(base_dir, variant, "llama-inference")

        logger.info(f"{'='*60}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")

        if not os.path.exists(input_path):
            logger.warning(f"  SKIPPED: {input_path} not found")
            continue

        with open(input_path, "r") as f:
            data = json.load(f)
        logger.info(f"  Loaded {len(data)} samples")

        evaluate_dataset(data, output_path, llm_model, tokenizer, args.top_k)

    # Cleanup
    del llm_model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL INFERENCE COMPLETE. Total time: {elapsed / 3600:.2f} hours")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
