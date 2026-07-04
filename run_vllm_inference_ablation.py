#!/usr/bin/env python3
"""
vLLM-Accelerated LLaMA Inference on Ablation Results.

Same evaluation logic as run_inference_ablation.py, but uses vLLM offline batch
inference for ~5-8x throughput improvement over HuggingFace sequential generation.

Loads LLM once via vLLM, iterates over all ablation variants' selected_triplets.json,
generates answers using top-k triplets with format_prompt (v5), and computes QA metrics.

Results saved to: ./results/<ablation-type>/<variant>/llama-inference/

Requirements:
    pip install vllm

Usage:
    python run_vllm_inference_ablation.py                          # all variants
    python run_vllm_inference_ablation.py --mode model             # model ablations only
    python run_vllm_inference_ablation.py --mode reward            # reward ablations only
    python run_vllm_inference_ablation.py --mode model --experiments no-ppr no-gate
    python run_vllm_inference_ablation.py --gpu-memory-utilization 0.85  # tune memory
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_inference import format_prompt
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
MODEL_MAP = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen-7B-Chat",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
}
MODEL_ABLATION_DIR = "./results/ablation-2/model-ablation"
REWARD_ABLATION_DIR = "./results/reward-ablation"
TOP_K = 100
LOG_FILE = os.path.join("logs", "ablation_vllm_inference.log")

MODEL_VARIANTS = ["no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"]
REWARD_VARIANTS = ["no_pres", "no_conn", "no_path", "only_pres", "only_conn", "only_cov"]

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("ablation.vllm_inference")


def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
# PROMPT BUILDING
# ============================================================================

def build_prompts(data, top_k):
    """
    Build all prompts for a dataset upfront.

    Returns:
        prompts: List of formatted prompt strings
        valid_indices: List of indices into `data` that produced valid prompts
        ground_truths: List of ground truth answer lists (aligned with prompts)
        questions: List of question strings (aligned with prompts)
    """
    prompts = []
    valid_indices = []
    ground_truths = []
    questions = []

    for i, sample in enumerate(data):
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

        # Format prompt using v5 from src
        triplets_used = triplets[:top_k]
        prompt = format_prompt(question, triplets_used, topk=top_k, q_entity=q_entity)

        prompts.append(prompt)
        valid_indices.append(i)
        ground_truths.append(ground_truth)
        questions.append(question)

    return prompts, valid_indices, ground_truths, questions


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_dataset_vllm(data, output_dir, llm, sampling_params, top_k):
    """
    Run vLLM batch inference on selected triplets and compute QA metrics.

    Args:
        data: List of sample dicts from selected_triplets.json
        output_dir: Directory to save results
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams
        top_k: Number of triplets to use per sample
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build all prompts upfront
    logger.info("    Building prompts...")
    prompts, valid_indices, ground_truths, questions = build_prompts(data, top_k)

    if not prompts:
        logger.warning("  No valid prompts found!")
        return None

    logger.info(f"    Built {len(prompts)} prompts from {len(data)} samples")

    # Step 2: Batch inference via vLLM
    logger.info("    Running vLLM batch inference...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - t0
    logger.info(
        f"    vLLM inference done: {len(outputs)} outputs in {inference_time:.1f}s "
        f"({len(outputs) / inference_time:.1f} samples/sec)"
    )

    # Step 3: Compute metrics
    hit_list, hit1_list, f1_list = [], [], []
    precision_list, recall_list = [], []
    detailed_results = []

    for idx, (output, ground_truth, question) in enumerate(
        zip(outputs, ground_truths, questions)
    ):
        try:
            raw_prediction = output.outputs[0].text.strip()
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
                "id": valid_indices[idx],
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
            logger.warning(f"  Error processing output {idx}: {e}")
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
        "inference_time_seconds": round(inference_time, 2),
        "throughput_samples_per_sec": round(len(outputs) / inference_time, 2),
    }

    logger.info(
        f"    Hit: {metrics['hit']:.2f}%, Hit@1: {metrics['hit_at_1']:.2f}%, "
        f"F1: {metrics['macro_f1']:.2f}%, EM: {metrics['exact_match']:.2f}%"
    )

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
    parser = argparse.ArgumentParser(
        description="Run vLLM batch inference on ablation results"
    )
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "model", "reward"]
    )
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama",
        choices=["llama", "qwen", "deepseek"],
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max sequence length (prompt + generation). Default: 8192",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per sample (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("ABLATION INFERENCE (vLLM): LLM QA Evaluation")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"LLM: {args.llm_model} ({MODEL_MAP[args.llm_model]})")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    logger.info(f"Max Model Len: {args.max_model_len}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # Load vLLM model
    logger.info("Loading vLLM model...")
    from vllm import LLM, SamplingParams

    model_id = MODEL_MAP[args.llm_model]
    llm = LLM(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    logger.info("  vLLM model loaded.")

    # Build list of (base_dir, variant) to process
    tasks = []
    if args.mode in ("all", "model"):
        variants = (
            args.experiments
            if (args.experiments and args.mode == "model")
            else MODEL_VARIANTS
        )
        for v in variants:
            tasks.append((MODEL_ABLATION_DIR, v))
    if args.mode in ("all", "reward"):
        variants = (
            args.experiments
            if (args.experiments and args.mode == "reward")
            else REWARD_VARIANTS
        )
        for v in variants:
            tasks.append((REWARD_ABLATION_DIR, v))

    # Summary of all results
    all_metrics = {}

    for base_dir, variant in tasks:
        input_path = os.path.join(
            base_dir, variant, "triplet-result", "selected_triplets.json"
        )
        output_path = os.path.join(base_dir, variant, "llama-inference")

        logger.info(f"{'=' * 60}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")

        if not os.path.exists(input_path):
            logger.warning(f"  SKIPPED: {input_path} not found")
            continue

        with open(input_path, "r") as f:
            data = json.load(f)
        logger.info(f"  Loaded {len(data)} samples")

        metrics = evaluate_dataset_vllm(
            data, output_path, llm, sampling_params, args.top_k
        )
        if metrics:
            all_metrics[variant] = metrics

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"{'Variant':<15} {'Hit%':>8} {'Hit@1%':>8} {'F1%':>8} {'EM%':>8} {'Throughput':>12}"
    )
    logger.info("-" * 65)
    for variant, m in all_metrics.items():
        logger.info(
            f"{variant:<15} {m['hit']:>8.2f} {m['hit_at_1']:>8.2f} "
            f"{m['macro_f1']:>8.2f} {m['exact_match']:>8.2f} "
            f"{m['throughput_samples_per_sec']:>10.1f} s/s"
        )

    # Cleanup
    del llm

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL INFERENCE COMPLETE. Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
