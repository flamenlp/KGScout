#!/usr/bin/env python3
"""
Comprehensive Failure Analysis for KGScout Reversed Attention Results.

Analyzes model performance by:
  1. Computing hop distribution of the test set (1-hop, 2-hop, 3-hop classes)
     - Classification rule: if a question has both 1-hop and 2-hop paths,
       assign to 1-hop only. Same for 2-hop vs 3-hop → assign to lower hop.
  2. Computing Hit score per hop class
  3. Computing answer_entity presence and path_presence per hop class
  4. Generating failure.json with questions where answer entity is missing
  5. Additional diagnostic suggestions for prompt calibration

Usage:
    python analyze_results.py \
        --test-data /path/to/test.pt \
        --llm-results-dir results/ablation-2/cwq/llm-results/ \
        --coverage-results-dir results/ablation-2/cwq/coverage/ \
        --output-dir results/ablation-2/cwq/analysis/ \
        --top-k 100

    # For WebQSP:
    python analyze_results.py \
        --test-data /path/to/webqsp/test.pt \
        --llm-results-dir results/ablation-2/webqsp/llm-results/ \
        --coverage-results-dir results/ablation-2/webqsp/coverage/ \
        --output-dir results/ablation-2/webqsp/analysis/ \
        --top-k 100
"""

import os
import sys
import json
import argparse
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.utils.metrics import (
    compute_answer_coverage, compute_path_coverage,
    extract_predictions_from_response, compute_hit_score,
    compute_hit_at_1, compute_precision, compute_recall,
    compute_f1_score, should_use_double_check, preprocess_date_answers,
)

import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("analyze_results")


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "analysis.log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ============================================================================
# DATASET
# ============================================================================

class SampledDataset(Dataset):
    def __init__(self, dataset, k=1000):
        self.dataset = dataset
        self.k = k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_available = data["topk_linearized_triplet_embeddings"].shape[0]
        use_nums = min(self.k, num_available)
        return {
            "question": data["question"],
            "is_empty": data["is_empty"],
            "q_entity": data["q_entity"],
            "a_entity": data["a_entity"],
            "answer": data["answer"],
            "question_embedding": data["question_embedding"],
            "topk_linearized_triplets": data["topk_linearized_triplets"][:use_nums],
            "topk_linearized_triplet_embeddings": data["topk_linearized_triplet_embeddings"][:use_nums],
            "topk_rel_data": data["topk_rel_data"][:use_nums],
            "topK_rel_embeddings": data["topK_rel_embeddings"][:use_nums],
            "graph_features": data["graph_features"][:use_nums],
        }


# ============================================================================
# HOP CLASSIFICATION
# ============================================================================

def classify_question_hop(triplets: List[Tuple[str, str, str]],
                          q_entities: List[str],
                          a_entities: List[str]) -> str:
    """
    Classify a question into hop class based on the MINIMUM shortest path
    from any q_entity to any a_entity in an undirected graph built from triplets.

    Classification rule:
      - If min shortest path == 1 → "1-hop"
      - If min shortest path == 2 → "2-hop"
      - If min shortest path == 3 → "3-hop"
      - If min shortest path > 3 → "3+hop"
      - If no path exists but answer entity is present → "no-path"
      - If answer entity not present → "no-entity"

    This means if a question has both 1-hop and 2-hop q-a pairs,
    it is classified as 1-hop (the minimum).
    """
    if not triplets or not q_entities or not a_entities:
        return "no-entity"

    # Build undirected graph
    G = nx.Graph()
    for s, r, o in triplets:
        G.add_edge(s.lower(), o.lower(), relation=r.lower())

    # Check if any answer entity is present
    a_ents_lower = [a.lower() for a in a_entities]
    any_present = any(a in G.nodes for a in a_ents_lower)

    if not any_present:
        return "no-entity"

    # Find minimum shortest path
    min_path = float('inf')
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            if qn not in G or an not in G:
                continue
            try:
                d = nx.shortest_path_length(G, qn, an)
                min_path = min(min_path, d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if min_path == float('inf'):
        return "no-path"
    elif min_path == 1:
        return "1-hop"
    elif min_path == 2:
        return "2-hop"
    elif min_path == 3:
        return "3-hop"
    else:
        return "3+hop"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(args):
    setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("FAILURE ANALYSIS: Hop Distribution + Per-Class Metrics")
    logger.info("=" * 70)

    # --- Load test data ---
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = torch.load(args.test_data, weights_only=False, map_location="cpu")
    logger.info(f"  Test samples: {len(test_data)}")

    # --- Load LLM detailed results (if available) ---
    llm_detailed_path = os.path.join(args.llm_results_dir, "llm_detailed_results.json")
    llm_results = None
    if os.path.exists(llm_detailed_path):
        with open(llm_detailed_path, "r") as f:
            llm_results = json.load(f)
        logger.info(f"  Loaded LLM results: {len(llm_results)} samples")
    else:
        logger.warning(f"  LLM results not found at: {llm_detailed_path}")
        logger.warning("  Will compute hop distribution and coverage only.")

    # --- Step 1: Compute hop distribution and per-sample metrics ---
    logger.info("\n--- Step 1: Hop Distribution Analysis ---")

    test_sampled = SampledDataset(test_data, k=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_sample_analysis = []
    hop_counter = defaultdict(int)

    for idx in tqdm(range(len(test_sampled)), desc="  Classifying hops"):
        sample = test_sampled[idx]
        q_entities = sample["q_entity"]
        a_entities = sample["a_entity"]
        question = sample["question"]

        # Extract top-k structured triplets
        topk_rel_data = sample["topk_rel_data"]
        num_triplets = min(args.top_k, len(topk_rel_data))
        triplets = [(d[1][0], d[1][1], d[1][2]) for d in topk_rel_data[:num_triplets]]

        # Classify hop
        hop_class = classify_question_hop(triplets, q_entities, a_entities)
        hop_counter[hop_class] += 1

        # Compute answer coverage and path coverage for this sample
        ans_cov = compute_answer_coverage(triplets, a_entities)
        path_cov = compute_path_coverage(triplets, q_entities, a_entities)

        per_sample_analysis.append({
            "idx": idx,
            "question": question,
            "q_entity": q_entities,
            "a_entity": a_entities,
            "hop_class": hop_class,
            "answer_coverage": ans_cov,
            "path_coverage": path_cov,
            "num_triplets": num_triplets,
        })

    # Print hop distribution
    total = len(per_sample_analysis)
    logger.info(f"\n  Hop Distribution (test set, top-{args.top_k} triplets):")
    logger.info(f"  {'Class':<12} {'Count':<8} {'Percent':<10}")
    logger.info(f"  {'-'*30}")
    for cls in ["1-hop", "2-hop", "3-hop", "3+hop", "no-path", "no-entity"]:
        cnt = hop_counter.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        logger.info(f"  {cls:<12} {cnt:<8} {pct:.1f}%")
    logger.info(f"  {'Total':<12} {total:<8}")

    # --- Step 2 & 3: Per-class Hit scores and coverage ---
    logger.info("\n--- Step 2 & 3: Per-Class Hit Score + Coverage ---")

    # Build a question→index map from LLM results
    llm_map = {}
    if llm_results:
        for i, r in enumerate(llm_results):
            llm_map[r["question"]] = r

    # Aggregate per hop class
    class_metrics = defaultdict(lambda: {
        "hit_list": [], "hit1_list": [], "f1_list": [],
        "precision_list": [], "recall_list": [],
        "ans_coverage_count": 0, "path_coverage_count": 0, "total": 0,
    })

    for sample in per_sample_analysis:
        cls = sample["hop_class"]
        metrics = class_metrics[cls]
        metrics["total"] += 1

        if sample["answer_coverage"]:
            metrics["ans_coverage_count"] += 1
        if sample["path_coverage"]:
            metrics["path_coverage_count"] += 1

        # Match with LLM results
        if sample["question"] in llm_map:
            llm_r = llm_map[sample["question"]]
            metrics["hit_list"].append(llm_r.get("hit", 0))
            metrics["hit1_list"].append(llm_r.get("hit_at_1", 0))
            metrics["f1_list"].append(llm_r.get("f1", 0))
            metrics["precision_list"].append(llm_r.get("precision", 0))
            metrics["recall_list"].append(llm_r.get("recall", 0))

    # Print per-class table
    logger.info(f"\n  {'Class':<12} {'N':<6} {'Hit%':<8} {'Hit@1%':<8} {'F1%':<8} {'AnsCov%':<9} {'PathCov%':<9}")
    logger.info(f"  {'-'*62}")

    summary_table = {}
    for cls in ["1-hop", "2-hop", "3-hop", "3+hop", "no-path", "no-entity"]:
        m = class_metrics[cls]
        n = m["total"]
        if n == 0:
            continue
        hit = np.mean(m["hit_list"]) * 100 if m["hit_list"] else 0
        hit1 = np.mean(m["hit1_list"]) * 100 if m["hit1_list"] else 0
        f1 = np.mean(m["f1_list"]) * 100 if m["f1_list"] else 0
        ans_cov = m["ans_coverage_count"] / n * 100
        path_cov = m["path_coverage_count"] / n * 100

        logger.info(f"  {cls:<12} {n:<6} {hit:<8.1f} {hit1:<8.1f} {f1:<8.1f} {ans_cov:<9.1f} {path_cov:<9.1f}")

        summary_table[cls] = {
            "count": n,
            "hit": round(hit, 2),
            "hit_at_1": round(hit1, 2),
            "f1": round(f1, 2),
            "precision": round(np.mean(m["precision_list"]) * 100, 2) if m["precision_list"] else 0,
            "recall": round(np.mean(m["recall_list"]) * 100, 2) if m["recall_list"] else 0,
            "answer_coverage_pct": round(ans_cov, 2),
            "path_coverage_pct": round(path_cov, 2),
        }

    # --- Step 4: Generate failure.json ---
    logger.info("\n--- Step 4: Generating failure.json ---")

    failures = []
    for sample in per_sample_analysis:
        # Failure = answer entity is NOT present in top-k triplets
        if not sample["answer_coverage"]:
            # Get the top-k triplets as formatted strings
            idx = sample["idx"]
            raw_sample = test_sampled[idx]
            topk_rel_data = raw_sample["topk_rel_data"]
            num_triplets = min(args.top_k, len(topk_rel_data))
            triplets_formatted = [
                f"{d[1][0]}, {d[1][1].replace('.', ' ').replace('_', ' ')}, {d[1][2]}"
                for d in topk_rel_data[:num_triplets]
            ]

            failure_entry = {
                "question": sample["question"],
                "q_entity": sample["q_entity"],
                "a_entity": sample["a_entity"],
                "hop_class": sample["hop_class"],
                "answer_coverage": False,
                "path_coverage": sample["path_coverage"],
                "top_100_triplets": triplets_formatted,
            }

            # Attach LLM prediction if available
            if sample["question"] in llm_map:
                llm_r = llm_map[sample["question"]]
                failure_entry["llm_prediction"] = llm_r.get("prediction", [])
                failure_entry["llm_hit"] = llm_r.get("hit", 0)
                failure_entry["llm_f1"] = llm_r.get("f1", 0)

            failures.append(failure_entry)

    failure_path = os.path.join(args.output_dir, "failure.json")
    with open(failure_path, "w") as f:
        json.dump(failures, f, indent=2)
    logger.info(f"  Saved {len(failures)} failure cases to: {failure_path}")
    logger.info(f"  (These are questions where answer entity is NOT in top-{args.top_k} triplets)")

    # --- Also generate failures where entity IS present but Hit=0 ---
    llm_failures = []
    if llm_results:
        for sample in per_sample_analysis:
            if sample["answer_coverage"] and sample["question"] in llm_map:
                llm_r = llm_map[sample["question"]]
                if llm_r.get("hit", 0) == 0:
                    idx = sample["idx"]
                    raw_sample = test_sampled[idx]
                    topk_rel_data = raw_sample["topk_rel_data"]
                    num_triplets = min(args.top_k, len(topk_rel_data))
                    triplets_formatted = [
                        f"{d[1][0]}, {d[1][1].replace('.', ' ').replace('_', ' ')}, {d[1][2]}"
                        for d in topk_rel_data[:num_triplets]
                    ]
                    llm_failures.append({
                        "question": sample["question"],
                        "q_entity": sample["q_entity"],
                        "a_entity": sample["a_entity"],
                        "hop_class": sample["hop_class"],
                        "answer_coverage": True,
                        "path_coverage": sample["path_coverage"],
                        "llm_prediction": llm_r.get("prediction", []),
                        "ground_truth": llm_r.get("ground_truth", []),
                        "llm_hit": 0,
                        "llm_f1": llm_r.get("f1", 0),
                        "top_100_triplets": triplets_formatted,
                    })

        llm_failure_path = os.path.join(args.output_dir, "failure_llm_wrong.json")
        with open(llm_failure_path, "w") as f:
            json.dump(llm_failures, f, indent=2)
        logger.info(f"  Saved {len(llm_failures)} LLM failure cases (entity present but Hit=0) to: {llm_failure_path}")

    # --- Save summary ---
    summary = {
        "config": {
            "test_data": args.test_data,
            "llm_results_dir": args.llm_results_dir,
            "top_k": args.top_k,
            "total_test_samples": total,
        },
        "hop_distribution": {cls: hop_counter.get(cls, 0) for cls in
                             ["1-hop", "2-hop", "3-hop", "3+hop", "no-path", "no-entity"]},
        "per_class_metrics": summary_table,
        "failure_summary": {
            "total_missing_entity": len(failures),
            "total_llm_wrong_with_entity": len(llm_failures) if llm_results else "N/A",
        },
    }

    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Summary saved to: {summary_path}")

    # --- Step 5: Suggestions ---
    logger.info("\n" + "=" * 70)
    logger.info("SUGGESTIONS FOR PROMPT CALIBRATION & PERFORMANCE ANALYSIS")
    logger.info("=" * 70)

    suggestions = []

    # Suggestion based on entity coverage
    entity_miss_pct = len(failures) / total * 100 if total > 0 else 0
    if entity_miss_pct > 30:
        suggestions.append(
            f"[RETRIEVAL] {entity_miss_pct:.0f}% of test questions have answer entity MISSING "
            f"from top-{args.top_k} triplets. This is the retrieval ceiling — no prompt can fix "
            f"what the retriever doesn't supply. Consider: "
            f"(a) Increasing k beyond 100, (b) Improving PPR/BM25 retrieval, "
            f"(c) Adding answer-entity-aware re-ranking during training."
        )

    # Suggestion based on LLM failures with entity present
    if llm_results and llm_failures:
        llm_fail_pct = len(llm_failures) / total * 100
        suggestions.append(
            f"[PROMPT/LLM] {len(llm_failures)} questions ({llm_fail_pct:.1f}%) have the answer entity "
            f"in triplets but the LLM still gets Hit=0. This is a prompt/inference issue. "
            f"Examine failure_llm_wrong.json to identify patterns:\n"
            f"    - Are predictions empty or 'answer not available'? → LLM not finding the path.\n"
            f"    - Are predictions wrong entities? → LLM confused by distractor triplets.\n"
            f"    - Are predictions correct but not matching? → Normalization/format issue."
        )

    # Hop-specific suggestions
    for cls in ["2-hop", "3-hop"]:
        if cls in summary_table:
            cls_hit = summary_table[cls]["hit"]
            cls_cov = summary_table[cls]["answer_coverage_pct"]
            if cls_hit < 40 and cls_cov > 50:
                suggestions.append(
                    f"[{cls.upper()} PROMPT] Hit={cls_hit:.0f}% but coverage={cls_cov:.0f}% for {cls}. "
                    f"The triplets contain the answer but the LLM struggles with multi-hop chaining. "
                    f"Consider: (a) Adding more multi-hop ICL examples, "
                    f"(b) Explicitly prompting 'chain through intermediate entities', "
                    f"(c) Ordering triplets by graph proximity to highlight the reasoning path."
                )

    # Suggestion about path ordering
    suggestions.append(
        "[TRIPLET ORDERING] Currently triplets are ordered by model score. "
        "Consider experimenting with: (a) Grouping triplets that share entities together, "
        "(b) Ordering by graph distance from q_entity (closest first), "
        "(c) Placing the reasoning path at the top."
    )

    # Suggestion about answer format
    suggestions.append(
        "[ANSWER FORMAT] Check if the LLM is outputting answers in unexpected formats "
        "(e.g., full sentences instead of entity names, dates in different formats). "
        "Run: grep for 'answer not available' in predictions to see how often the LLM "
        "gives up vs produces wrong entities."
    )

    for i, s in enumerate(suggestions, 1):
        logger.info(f"\n  {i}. {s}")

    # Save suggestions to file
    suggestions_path = os.path.join(args.output_dir, "suggestions.txt")
    with open(suggestions_path, "w") as f:
        for i, s in enumerate(suggestions, 1):
            f.write(f"{i}. {s}\n\n")
    logger.info(f"\n  Suggestions saved to: {suggestions_path}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Failure analysis for KGScout reversed attention results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # CWQ analysis:
    python analyze_results.py \\
        --test-data /path/to/cwq/test_jointrainer_path_dataset_v3_ppr.pt \\
        --llm-results-dir results/ablation-2/cwq/llm-results/ \\
        --output-dir results/ablation-2/cwq/analysis/ \\
        --top-k 100

    # WebQSP analysis:
    python analyze_results.py \\
        --test-data /path/to/webqsp/test_jointrainer_path_dataset_v3_ppr.pt \\
        --llm-results-dir results/ablation-2/webqsp/llm-results/ \\
        --output-dir results/ablation-2/webqsp/analysis/ \\
        --top-k 100

Output files:
    analysis_summary.json    - Overall hop distribution + per-class metrics
    failure.json             - Questions where answer entity is MISSING from triplets
    failure_llm_wrong.json   - Questions where entity IS present but LLM gets Hit=0
    suggestions.txt          - Actionable suggestions for prompt calibration
    analysis.log             - Full log
        """
    )
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test .pt file")
    parser.add_argument("--llm-results-dir", type=str, required=True,
                        help="Path to directory containing llm_detailed_results.json")
    parser.add_argument("--output-dir", type=str, default="results/analysis/",
                        help="Output directory for analysis results")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top triplets used for evaluation (default: 100)")

    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}")
        sys.exit(1)

    run_analysis(args)


if __name__ == "__main__":
    main()
