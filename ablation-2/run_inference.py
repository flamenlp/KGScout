#!/usr/bin/env python3
"""
LLM Inference from selected_triplets.json.

Reads pre-computed model-ranked triplets (selected_triplets.json), runs LLM inference
using format_prompt_v5, and computes QA metrics (Hit, Hit@1, F1, Precision, Recall, EM).

This decouples LLM evaluation from model training — you can re-run inference
with different prompts without re-running the model.

Usage:
    python ablation-2/run_inference.py \
        --input results/ablation-2/cwq/triplet-result/selected_triplets.json \
        --output results/ablation-2/cwq/llm-results/ \
        --llm-model llama \
        --top-k 100

    # Or use shorthand for dataset:
    python ablation-2/run_inference.py --dataset cwq
    python ablation-2/run_inference.py --dataset webqsp
"""

import os
import sys
import json
import logging
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    extract_predictions_from_response, compute_hit_score,
    compute_hit_at_1, compute_precision, compute_recall,
    compute_f1_score, should_use_double_check, preprocess_date_answers,
)
from src.utils.llm_inference import load_llm_model, format_prompt_v5, run_llm_inference

# ============================================================================
# HARDCODED PATHS (for --dataset shorthand)
# ============================================================================

DATASET_CONFIGS = {
    "cwq": {
        "input": "./results/ablation-2/cwq/triplet-result/selected_triplets.json",
        "output": "./results/ablation-2/cwq/llm-results/",
    },
    "webqsp": {
        "input": "./results/ablation-2/webqsp/triplet-result/selected_triplets.json",
        "output": "./results/ablation-2/webqsp/llm-results/",
    },
}

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("run_inference")


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "inference.log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def run_evaluation(args):
    setup_logging(args.output)

    logger.info("=" * 70)
    logger.info("LLM INFERENCE FROM SELECTED TRIPLETS")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Model: {args.llm_model}")
    logger.info(f"  Top-k: {args.top_k}")
    logger.info("=" * 70)

    # --- Load selected triplets ---
    with open(args.input, "r") as f:
        data = json.load(f)
    logger.info(f"  Loaded {len(data)} samples from selected_triplets.json")

    # --- Load LLM ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Loading LLM ({args.llm_model}) on {device}...")
    llm_model, tokenizer = load_llm_model(args.llm_model, str(device))
    logger.info("  LLM loaded.")

    # --- Run inference ---
    hit_list, hit1_list, f1_list = [], [], []
    precision_list, recall_list = [], []
    detailed_results = []

    start_time = time.time()

    for i, sample in enumerate(tqdm(data, desc="  LLM Inference")):
        try:
            question = sample["question"]
            ground_truth = sample.get("answer", sample.get("a_entity", []))
            q_entity = sample.get("q_entity", [])
            a_entity = sample.get("a_entity", [])
            triplets = sample.get("reranker", [])
        except (KeyError, TypeError) as e:
            logger.warning(f"  Sample {i}: missing key or bad format: {e}")
            continue

        if not ground_truth or not triplets:
            continue

        # Ensure ground_truth is a flat list of strings
        if isinstance(ground_truth, dict):
            ground_truth = list(ground_truth.values())[0] if ground_truth else []
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        ground_truth = [str(a) for a in ground_truth if a]

        if not ground_truth:
            continue

        # Limit to top-k triplets
        triplets_used = triplets[:args.top_k]

        # Format prompt using v5
        prompt = format_prompt_v5(question, triplets_used, topk=args.top_k, q_entity=q_entity)

        # Run LLM inference
        try:
            raw_prediction = run_llm_inference(llm_model, tokenizer, prompt)
            prediction = extract_predictions_from_response(raw_prediction)
            prediction = [str(s) for s in prediction if s != "" and s is not None]
        except Exception as e:
            logger.warning(f"  LLM failed for sample {i}: {e}")
            prediction = []

        # Preprocess answers — ensure all items are strings
        answer = [str(a) for a in ground_truth if a is not None and a != ""]
        answer = preprocess_date_answers(question, answer)
        double_check = should_use_double_check(question)

        # Compute metrics
        prec, _, _ = compute_precision(prediction, answer, double_check)
        rec, _, _ = compute_recall(prediction, answer, double_check)
        f1 = compute_f1_score(prec, rec)
        hit = compute_hit_score(prediction, answer, double_check)
        hit1 = compute_hit_at_1(prediction, answer, double_check)

        hit_list.append(hit)
        hit1_list.append(hit1)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

        detailed_results.append({
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "q_entity": q_entity,
            "hit": hit,
            "hit_at_1": hit1,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "selected_triplets": triplets_used,
        })

    elapsed = time.time() - start_time
    n = len(hit_list)

    # --- Compute aggregate metrics ---
    metrics = {
        "hit": round(np.mean(hit_list) * 100, 2) if n else 0,
        "hit_at_1": round(np.mean(hit1_list) * 100, 2) if n else 0,
        "macro_f1": round(np.mean(f1_list) * 100, 2) if n else 0,
        "macro_precision": round(np.mean(precision_list) * 100, 2) if n else 0,
        "macro_recall": round(np.mean(recall_list) * 100, 2) if n else 0,
        "exact_match": round((np.array(f1_list) == 1).sum() / n * 100, 2) if n else 0,
        "total_samples": n,
        "elapsed_seconds": round(elapsed, 1),
        "llm_model": args.llm_model,
        "top_k": args.top_k,
        "input_file": args.input,
    }

    logger.info(f"\n{'=' * 70}")
    logger.info("RESULTS")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Samples: {n}")
    logger.info(f"  Hit:     {metrics['hit']:.2f}%")
    logger.info(f"  Hit@1:   {metrics['hit_at_1']:.2f}%")
    logger.info(f"  F1:      {metrics['macro_f1']:.2f}%")
    logger.info(f"  Prec:    {metrics['macro_precision']:.2f}%")
    logger.info(f"  Recall:  {metrics['macro_recall']:.2f}%")
    logger.info(f"  EM:      {metrics['exact_match']:.2f}%")
    logger.info(f"  Time:    {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/sample)")
    logger.info(f"{'=' * 70}")

    # --- Save results ---
    os.makedirs(args.output, exist_ok=True)

    metrics_path = os.path.join(args.output, "llm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to: {metrics_path}")

    detailed_path = os.path.join(args.output, "llm_detailed_results.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    logger.info(f"  Detailed results saved to: {detailed_path}")

    logger.info("\nDone.")
    return metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference on selected_triplets.json and compute QA metrics",
    )
    parser.add_argument("--dataset", type=str, choices=["cwq", "webqsp"],
                        help="Use preset paths for dataset (alternative to --input/--output)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to selected_triplets.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--llm-model", type=str, default="llama",
                        choices=["llama", "qwen", "deepseek"],
                        help="LLM model to use (default: llama)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of triplets to include in prompt (default: 100)")

    args = parser.parse_args()

    # Resolve paths
    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        if args.input is None:
            args.input = config["input"]
        if args.output is None:
            args.output = config["output"]
    else:
        if args.input is None or args.output is None:
            parser.error("Either --dataset or both --input and --output are required")

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()
