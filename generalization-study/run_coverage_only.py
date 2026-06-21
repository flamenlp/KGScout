#!/usr/bin/env python3
"""
Run coverage-only evaluation (no LLM needed).

Computes answer coverage and path coverage metrics for KGScout on MetaQA.
This is fast and doesn't require a GPU-heavy LLM, useful for quick validation.

Usage:
    python generalization-study/run_coverage_only.py \
        --model-path checkpoints/webqsp-k100/main/ \
        --data-dir data/metaqa/processed/ \
        --all-hops \
        --output-dir results/generalization/coverage/
"""

import os
import sys
import json
import argparse
import torch
import networkx as nx
from typing import List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.path_ranker import PathRankingModel
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.utils.metrics import compute_answer_coverage, compute_path_coverage


def load_kgscout_model(model_path: str, device: str) -> PathRankingModel:
    """Load trained KGScout model."""
    print(f"Loading model from: {model_path}")
    model = PathRankingModel.from_pretrained(model_path, device=device)
    model.eval()
    return model


def select_top_k(model, sample, top_k, device):
    """Select top-k triplets using KGScout ranking (sample_paths + sort by prob)."""
    question_embed = sample["question_embedding"].to(device)
    triplet_embeds = sample["topk_linearized_triplet_embeddings"].to(device)
    relation_embeds = sample["topK_rel_embeddings"].to(device)
    graph_features = sample["graph_features"].to(device)

    if question_embed.dim() == 1:
        question_embed = question_embed.unsqueeze(0)
    if triplet_embeds.dim() == 3:
        triplet_embeds = triplet_embeds.squeeze(0)
    if relation_embeds.dim() == 3:
        relation_embeds = relation_embeds.squeeze(0)
    if graph_features.dim() == 3:
        graph_features = graph_features.squeeze(0)

    triplets = [t[1] for t in sample["topk_rel_data"]]
    triplets_linearized = sample["topk_linearized_triplets"]
    if len(triplets) == 0:
        return []

    k = min(top_k, len(triplets))
    with torch.no_grad():
        ranking_scores, path_probs = model(
            question_embed, triplet_embeds, relation_embeds, graph_features
        )

    # sample_paths (stochastic) then sort by probability - matches notebook
    selected_paths, selected_probs, _, _ = model.sample_paths(
        path_probs, triplets, k, ranking_scores
    )
    sorted_indices = torch.argsort(selected_probs, descending=True)
    return [selected_paths[i] for i in sorted_indices.tolist()]


def run_coverage_evaluation(model, dataset, hop, top_k, device, output_dir):
    """Evaluate answer and path coverage for KGScout on a MetaQA hop."""
    print(f"\nEvaluating coverage for {hop}-hop ({len(dataset)} samples)...")

    total_answer_cov = 0.0
    total_path_cov = 0.0
    n = 0

    for idx in tqdm(range(len(dataset)), desc=f"Coverage {hop}-hop"):
        sample = dataset[idx]
        q_entities = sample["q_entity"]
        a_entities = sample["a_entity"]

        selected = select_top_k(model, sample, top_k, device)
        if len(selected) == 0:
            continue

        ans_cov = compute_answer_coverage(selected, a_entities)
        path_cov = compute_path_coverage(selected, q_entities, a_entities)

        total_answer_cov += float(ans_cov)
        total_path_cov += float(path_cov)
        n += 1

    metrics = {
        "answer_coverage": total_answer_cov / n if n > 0 else 0.0,
        "path_coverage": total_path_cov / n if n > 0 else 0.0,
        "total_evaluated": n,
    }

    # Also compute cosine baseline
    total_cos_ans = 0.0
    total_cos_path = 0.0
    n_cos = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        triplets = [t[1] for t in sample["topk_rel_data"]][:top_k]
        if len(triplets) == 0:
            continue
        q_entities = sample["q_entity"]
        a_entities = sample["a_entity"]
        total_cos_ans += float(compute_answer_coverage(triplets, a_entities))
        total_cos_path += float(compute_path_coverage(triplets, q_entities, a_entities))
        n_cos += 1

    cosine_metrics = {
        "answer_coverage": total_cos_ans / n_cos if n_cos > 0 else 0.0,
        "path_coverage": total_cos_path / n_cos if n_cos > 0 else 0.0,
        "total_evaluated": n_cos,
    }

    print(f"\n  KGScout (top-{top_k}):")
    print(f"    Answer Coverage: {metrics['answer_coverage']:.4f}")
    print(f"    Path Coverage:   {metrics['path_coverage']:.4f}")
    print(f"  Cosine Baseline (top-{top_k}):")
    print(f"    Answer Coverage: {cosine_metrics['answer_coverage']:.4f}")
    print(f"    Path Coverage:   {cosine_metrics['path_coverage']:.4f}")

    return {"kgscout": metrics, "cosine": cosine_metrics}


def main():
    parser = argparse.ArgumentParser(description="Coverage-only evaluation on MetaQA")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/metaqa/processed/")
    parser.add_argument("--hop", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--all-hops", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/generalization/coverage/")
    args = parser.parse_args()

    if args.hop is None and not args.all_hops:
        parser.error("Must specify --hop or --all-hops")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hops = [1, 2, 3] if args.all_hops else [args.hop]

    model = load_kgscout_model(args.model_path, device)

    all_results = {}
    for hop in hops:
        data_path = os.path.join(args.data_dir, f"metaqa-{hop}hop-test.pt")
        if not os.path.exists(data_path):
            print(f"WARNING: {data_path} not found. Run preprocess_metaqa.py first.")
            continue
        raw_data = torch.load(data_path, map_location="cpu", weights_only=False)
        dataset = JointTrainingDatasetv3PPR(raw_data, device="cpu")
        result = run_coverage_evaluation(model, dataset, hop, args.top_k, device, args.output_dir)
        all_results[f"{hop}-hop"] = result

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"coverage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "top_k": args.top_k,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
