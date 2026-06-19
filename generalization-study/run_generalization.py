#!/usr/bin/env python3
"""
Run Generalization Study: Evaluate KGScout on MetaQA.

Pipeline per hop:
1. Load preprocessed MetaQA .pt file (top-N=1000 cosine triplets per question)
2. Load KGScout model (trained on WebQSP or CWQ)
3. Use KGScout to re-rank and select top-k=100 triplets
4. Feed selected triplets + question to LLM (Llama-3.1-8b)
5. Extract predictions and compute metrics (Hit, Hit@1, F1, Precision, Recall, EM)
6. Save results

Usage:
    # Single hop
    python generalization-study/run_generalization.py \
        --model-path checkpoints/webqsp-k100/main/ \
        --dataset-name webqsp \
        --hop 1 \
        --output-dir results/generalization/

    # All hops
    python generalization-study/run_generalization.py \
        --model-path checkpoints/webqsp-k100/main/ \
        --dataset-name webqsp \
        --all-hops \
        --output-dir results/generalization/
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.path_ranker import PathRankingModel
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
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


def load_kgscout_model(model_path: str, device: str) -> PathRankingModel:
    """Load a trained KGScout model from checkpoint directory."""
    print(f"Loading KGScout model from: {model_path}")
    model = PathRankingModel.from_pretrained(model_path, device=device)
    model.eval()
    print(f"Model loaded. Hidden size: {model.hidden_size}")
    return model


def load_metaqa_data(data_path: str, device: str) -> List[Dict]:
    """
    Load preprocessed MetaQA .pt file and compute PPR features.
    Returns list of samples with graph_features added.
    """
    print(f"Loading preprocessed MetaQA data: {data_path}")
    raw_data = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(raw_data)} samples")

    # Use JointTrainingDatasetv3PPR to compute PPR graph features
    print("Computing PPR graph features...")
    dataset = JointTrainingDatasetv3PPR(raw_data, device="cpu")
    print(f"PPR computed. Final dataset size: {len(dataset)}")

    return dataset


def select_triplets_with_kgscout(
    model: PathRankingModel,
    sample: Dict,
    top_k: int,
    device: str
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Use KGScout model to re-rank triplets and select top-k.

    Returns:
        linearized: List of linearized triplet strings (for LLM prompt)
        structured: List of (s, r, o) tuples (for metrics)
    """
    question_embed = sample["question_embedding"].to(device)
    triplet_embeds = sample["topk_linearized_triplet_embeddings"].to(device)
    relation_embeds = sample["topK_rel_embeddings"].to(device)
    graph_features = sample["graph_features"].to(device)

    # Ensure correct dimensions
    if question_embed.dim() == 1:
        question_embed = question_embed.unsqueeze(0)
    if triplet_embeds.dim() == 3:
        triplet_embeds = triplet_embeds.squeeze(0)
    if relation_embeds.dim() == 3:
        relation_embeds = relation_embeds.squeeze(0)
    if graph_features.dim() == 3:
        graph_features = graph_features.squeeze(0)

    # Get triplets and linearized versions
    triplets_structured = [t[1] for t in sample["topk_rel_data"]]
    triplets_linearized = sample["topk_linearized_triplets"]

    num_available = len(triplets_structured)
    if num_available == 0:
        return [], []

    k = min(top_k, num_available)

    with torch.no_grad():
        ranking_scores, _ = model(
            question_embed, triplet_embeds, relation_embeds, graph_features
        )

    # Select top-k by ranking scores
    top_k_indices = torch.topk(ranking_scores.squeeze(), k).indices.cpu().tolist()

    selected_linearized = [triplets_linearized[i] for i in top_k_indices]
    selected_structured = [triplets_structured[i] for i in top_k_indices]

    return selected_linearized, selected_structured


def run_evaluation_for_hop(
    model: PathRankingModel,
    dataset,
    llm_model,
    tokenizer,
    hop: int,
    top_k: int,
    device: str,
    output_dir: str,
    dataset_name: str,
):
    """
    Run full evaluation pipeline for a single hop.
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATING {hop}-HOP ({len(dataset)} questions)")
    print(f"{'=' * 60}")

    results = []
    total_hit = 0.0
    total_hit_at_1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_exact_match = 0.0
    failed_llm = 0

    for idx in tqdm(range(len(dataset)), desc=f"MetaQA-{hop}hop"):
        sample = dataset[idx]
        question = sample["question"]
        ground_truth = sample["answer"]

        # Step 1: Select top-k triplets using KGScout
        selected_linearized, selected_structured = select_triplets_with_kgscout(
            model, sample, top_k, device
        )

        if len(selected_linearized) == 0:
            results.append({
                "question": question,
                "predicted": [],
                "ground_truth": ground_truth,
                "hit": 0.0, "hit_at_1": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "exact_match": 0.0, "num_triplets": 0,
            })
            continue

        # Step 2: Format prompt and run LLM inference
        prompt = format_prompt(question, selected_linearized, topk=top_k)

        try:
            response = run_llm_inference(llm_model, tokenizer, prompt)
            predicted = extract_predictions_from_response(response)
        except Exception as e:
            failed_llm += 1
            predicted = []

        # Step 3: Compute metrics
        double_check = should_use_double_check(question)
        gt_processed = preprocess_date_answers(question, ground_truth)

        hit = compute_hit_score(predicted, gt_processed, double_check)
        hit_at_1 = compute_hit_at_1(predicted, gt_processed, double_check)
        precision, _, _ = compute_precision(predicted, gt_processed, double_check)
        recall, _, _ = compute_recall(predicted, gt_processed, double_check)
        f1 = compute_f1_score(precision, recall)
        exact_match = 1.0 if (precision == 1.0 and recall == 1.0) else 0.0

        total_hit += hit
        total_hit_at_1 += hit_at_1
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_exact_match += exact_match

        results.append({
            "question": question,
            "predicted": predicted,
            "ground_truth": gt_processed,
            "hit": hit, "hit_at_1": hit_at_1,
            "precision": precision, "recall": recall, "f1": f1,
            "exact_match": exact_match,
            "num_triplets": len(selected_linearized),
        })

    # Compute averages
    n = len(results)
    metrics = {
        "hit": total_hit / n if n > 0 else 0.0,
        "hit_at_1": total_hit_at_1 / n if n > 0 else 0.0,
        "macro_precision": total_precision / n if n > 0 else 0.0,
        "macro_recall": total_recall / n if n > 0 else 0.0,
        "macro_f1": total_f1 / n if n > 0 else 0.0,
        "exact_match": total_exact_match / n if n > 0 else 0.0,
    }

    # Save results
    hop_dir = os.path.join(output_dir, f"{dataset_name}-on-metaqa-{hop}hop")
    os.makedirs(hop_dir, exist_ok=True)

    # Summary
    summary = {
        "dataset": f"metaqa-{hop}hop",
        "source_model": dataset_name,
        "retriever_type": "kgscout",
        "k": top_k,
        "total_questions": n,
        "failed_llm_inferences": failed_llm,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": metrics,
    }
    summary_path = os.path.join(hop_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Detailed results (JSONL)
    results_path = os.path.join(hop_dir, "results.jsonl")
    with open(results_path, 'w') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    # Predictions
    predictions_path = os.path.join(hop_dir, "predictions.txt")
    with open(predictions_path, 'w') as f:
        for r in results:
            if r["predicted"]:
                f.write(r["predicted"][0] + '\n')
            else:
                f.write('\n')

    # Print summary
    print(f"\n{'─' * 40}")
    print(f"Results for MetaQA {hop}-hop:")
    print(f"  Total Questions: {n}")
    print(f"  Hit:             {metrics['hit']:.4f}")
    print(f"  Hit@1:           {metrics['hit_at_1']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Exact Match:     {metrics['exact_match']:.4f}")
    print(f"  Failed LLM:      {failed_llm}")
    print(f"  Saved to:        {hop_dir}")
    print(f"{'─' * 40}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run KGScout generalization study on MetaQA"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained KGScout model checkpoint directory")
    parser.add_argument("--dataset-name", type=str, required=True, choices=["webqsp", "cwq"],
                        help="Name of the source dataset the model was trained on")
    parser.add_argument("--hop", type=int, default=None, choices=[1, 2, 3],
                        help="MetaQA hop to evaluate (1, 2, or 3)")
    parser.add_argument("--all-hops", action="store_true",
                        help="Evaluate all hops (1, 2, 3)")
    parser.add_argument("--data-dir", type=str, default="data/metaqa/processed/",
                        help="Directory containing preprocessed metaqa-Xhop-test.pt files")
    parser.add_argument("--output-dir", type=str, default="results/generalization/",
                        help="Output directory for results")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of triplets to select with KGScout (default: 100)")
    parser.add_argument("--llm-model", type=str, default="llama", choices=["llama", "qwen", "deepseek"],
                        help="LLM model to use for answer generation (default: llama)")
    args = parser.parse_args()

    if args.hop is None and not args.all_hops:
        parser.error("Must specify --hop or --all-hops")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Determine which hops to run
    hops_to_run = [1, 2, 3] if args.all_hops else [args.hop]

    # Load KGScout model (once, reuse for all hops)
    model = load_kgscout_model(args.model_path, device)

    # Load LLM (once, reuse for all hops)
    print(f"\nLoading LLM: {args.llm_model}")
    llm_model, tokenizer = load_llm_model(args.llm_model, device)

    # Run evaluation for each hop
    all_metrics = {}
    for hop in hops_to_run:
        data_path = os.path.join(args.data_dir, f"metaqa-{hop}hop-test.pt")
        if not os.path.exists(data_path):
            print(f"\nWARNING: Preprocessed data not found: {data_path}")
            print(f"Run preprocess_metaqa.py first for {hop}-hop data.")
            continue

        dataset = load_metaqa_data(data_path, device)
        metrics = run_evaluation_for_hop(
            model=model,
            dataset=dataset,
            llm_model=llm_model,
            tokenizer=tokenizer,
            hop=hop,
            top_k=args.top_k,
            device=device,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
        )
        all_metrics[f"{hop}-hop"] = metrics

    # Print overall summary
    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}")
        print(f"OVERALL GENERALIZATION RESULTS ({args.dataset_name} → MetaQA)")
        print(f"{'=' * 60}")
        print(f"{'Hop':<8}{'Hit':<10}{'Hit@1':<10}{'F1':<10}{'Prec':<10}{'Recall':<10}{'EM':<10}")
        print(f"{'─' * 58}")
        for hop_name, m in all_metrics.items():
            print(f"{hop_name:<8}{m['hit']:<10.4f}{m['hit_at_1']:<10.4f}"
                  f"{m['macro_f1']:<10.4f}{m['macro_precision']:<10.4f}"
                  f"{m['macro_recall']:<10.4f}{m['exact_match']:<10.4f}")
        print(f"{'=' * 60}")

        # Save combined summary
        combined_path = os.path.join(args.output_dir, f"{args.dataset_name}-metaqa-generalization-summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(combined_path, 'w') as f:
            json.dump({
                "source_model": args.dataset_name,
                "model_path": args.model_path,
                "top_k": args.top_k,
                "llm_model": args.llm_model,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "results": all_metrics
            }, f, indent=2)
        print(f"Combined summary saved to: {combined_path}")


if __name__ == "__main__":
    main()
