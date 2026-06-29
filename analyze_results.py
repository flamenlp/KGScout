#!/usr/bin/env python3
"""
Comprehensive Failure Analysis for KGScout Reversed Attention Results.

Analyzes model performance by reading llm_detailed_results.json directly:
  1. Loads test data to enrich LLM results with q_entity (matched by question + ground_truth)
  2. Parses the selected_triplets from LLM results (model-ranked top-k triplets)
  3. Computes hop distribution of the test set (1-hop, 2-hop, 3-hop classes)
     - Classification rule: if a question has both 1-hop and 2-hop paths,
       assign to 1-hop only. Same for 2-hop vs 3-hop → assign to lower hop.
  4. Computing Hit score per hop class
  5. Computing answer_entity presence and path_presence per hop class
     (using the SAME model-ranked triplets that were fed to the LLM)
  6. Generating failure.json with questions where answer entity is missing
  7. Additional diagnostic suggestions for prompt calibration

Usage:
    python analyze_results.py --dataset cwq
    python analyze_results.py --dataset webqsp
"""

# ============================================================================
# HARDCODED CONFIGURATION
# ============================================================================

# --- CWQ Paths ---
CWQ_TEST_DATA = "/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/test/test_jointrainer_path_dataset_v3_ppr.pt"
CWQ_LLM_RESULTS_DIR = "./results/ablation-2/cwq/llm-results/"
CWQ_OUTPUT_DIR = "./results/ablation-2/cwq/analysis/"

# --- WebQSP Paths ---
WEBQSP_TEST_DATA = "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"
WEBQSP_LLM_RESULTS_DIR = "./results/ablation-2/webqsp/llm-results/"
WEBQSP_OUTPUT_DIR = "./results/ablation-2/webqsp/analysis/"

# --- Common ---
TOP_K = 100

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import networkx as nx
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.utils.metrics import (
    compute_answer_coverage, compute_path_coverage,
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
# ENRICH LLM RESULTS WITH Q_ENTITY FROM TEST DATA
# ============================================================================

def enrich_llm_results_with_q_entity(llm_results: List[Dict], test_data_path: str) -> int:
    """
    Load test data, match each LLM result by (question, ground_truth) and
    inject q_entity from the test dataset into the LLM result dict.

    Returns the number of successfully matched samples.
    """
    logger.info(f"  Loading test data to extract q_entity: {test_data_path}")
    test_data = torch.load(test_data_path, weights_only=False, map_location="cpu")
    logger.info(f"  Test data loaded: {len(test_data)} samples")

    # Build lookup: (question, frozenset(ground_truth)) -> q_entity
    # Use question + sorted ground_truth as key to handle order differences
    test_lookup = {}
    for sample in test_data:
        question = sample["question"]
        # ground_truth in test data is stored as "answer" field
        answers = sample.get("answer", [])
        # Normalize: sort answers for consistent matching
        key = (question, tuple(sorted(a.lower() for a in answers)))
        test_lookup[key] = sample["q_entity"]

    matched = 0
    unmatched = 0
    for llm_r in llm_results:
        # Skip if q_entity already present
        if "q_entity" in llm_r and llm_r["q_entity"]:
            matched += 1
            continue

        question = llm_r["question"]
        ground_truth = llm_r.get("ground_truth", [])
        key = (question, tuple(sorted(a.lower() for a in ground_truth)))

        if key in test_lookup:
            llm_r["q_entity"] = test_lookup[key]
            matched += 1
        else:
            # Fallback: try matching by question only (less precise but covers edge cases)
            fallback_key = question
            found = False
            for sample in test_data:
                if sample["question"] == fallback_key:
                    llm_r["q_entity"] = sample["q_entity"]
                    matched += 1
                    found = True
                    break
            if not found:
                llm_r["q_entity"] = []
                unmatched += 1

    logger.info(f"  Matched q_entity for {matched}/{len(llm_results)} samples ({unmatched} unmatched)")

    # Free memory
    del test_data
    return matched


# ============================================================================
# TRIPLET PARSING
# ============================================================================

def parse_triplet_string(triplet_str: str) -> Tuple[str, str, str]:
    """
    Parse a formatted triplet string back into (subject, relation, object).

    The format from run_reversed_attention.py is:
        "subject, relation words, object"

    Since relation can contain commas after format_relation() (unlikely but possible),
    we split on ", " and treat first element as subject, last as object,
    and everything in between as relation.
    """
    parts = triplet_str.split(", ")
    if len(parts) < 3:
        # Fallback: try splitting on just comma
        parts = triplet_str.split(",")
        parts = [p.strip() for p in parts]

    if len(parts) < 3:
        return (triplet_str, "", "")

    subject = parts[0]
    obj = parts[-1]
    relation = ", ".join(parts[1:-1])
    return (subject, relation, obj)


def parse_triplets_from_llm_result(llm_result: Dict) -> List[Tuple[str, str, str]]:
    """
    Extract structured triplets from an LLM result entry's selected_triplets field.
    """
    selected = llm_result.get("selected_triplets", [])
    triplets = []
    for t_str in selected:
        triplets.append(parse_triplet_string(t_str))
    return triplets


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
    logger.info("  (Using model-ranked triplets from llm_detailed_results.json)")
    logger.info("=" * 70)

    # --- Load LLM detailed results ---
    llm_detailed_path = os.path.join(args.llm_results_dir, "llm_detailed_results.json")
    if not os.path.exists(llm_detailed_path):
        logger.error(f"LLM results not found at: {llm_detailed_path}")
        logger.error("Cannot proceed without llm_detailed_results.json")
        sys.exit(1)

    with open(llm_detailed_path, "r") as f:
        llm_results = json.load(f)
    logger.info(f"  Loaded LLM results: {len(llm_results)} samples")

    # --- Check that selected_triplets key exists ---
    if llm_results and "selected_triplets" not in llm_results[0]:
        logger.error("llm_detailed_results.json does not contain 'selected_triplets' key.")
        logger.error("Re-run LLM evaluation with a version that saves selected_triplets.")
        sys.exit(1)

    # --- Enrich LLM results with q_entity from test data ---
    has_q_entity = llm_results and "q_entity" in llm_results[0] and llm_results[0]["q_entity"]
    if not has_q_entity:
        logger.info("\n--- Enriching LLM results with q_entity from test data ---")
        enrich_llm_results_with_q_entity(llm_results, args.test_data)
    else:
        logger.info("  q_entity already present in LLM results.")


    # --- Step 1: Compute hop distribution and per-sample metrics ---
    logger.info("\n--- Step 1: Hop Distribution Analysis (using model-ranked triplets from LLM results) ---")

    per_sample_analysis = []
    hop_counter = defaultdict(int)

    for idx, llm_r in enumerate(llm_results):
        question = llm_r["question"]
        ground_truth = llm_r.get("ground_truth", [])

        # Extract q_entity and a_entity
        # ground_truth serves as a_entity for coverage check
        a_entities = ground_truth

        # q_entity: try to get from llm_result if available, otherwise infer from triplets
        q_entities = llm_r.get("q_entity", [])

        # Parse the model-ranked triplets that were actually fed to the LLM
        triplets = parse_triplets_from_llm_result(llm_r)

        if len(triplets) == 0:
            hop_class = "no-entity"
            ans_cov = False
            path_cov = False
        else:
            # Compute answer coverage and path coverage on the SAME triplets fed to LLM
            ans_cov = compute_answer_coverage(triplets, a_entities)
            path_cov = compute_path_coverage(triplets, q_entities, a_entities) if q_entities else False

            # Classify hop (only meaningful if we have q_entities)
            if q_entities:
                hop_class = classify_question_hop(triplets, q_entities, a_entities)
            else:
                # Without q_entity, classify based on coverage only
                if not ans_cov:
                    hop_class = "no-entity"
                else:
                    hop_class = "unknown-hop"

        hop_counter[hop_class] += 1

        per_sample_analysis.append({
            "idx": idx,
            "question": question,
            "q_entity": q_entities,
            "a_entity": a_entities,
            "hop_class": hop_class,
            "answer_coverage": ans_cov,
            "path_coverage": path_cov,
            "num_triplets": len(triplets),
            "hit": llm_r.get("hit", 0),
            "hit_at_1": llm_r.get("hit_at_1", 0),
            "f1": llm_r.get("f1", 0),
            "precision": llm_r.get("precision", 0),
            "recall": llm_r.get("recall", 0),
            "prediction": llm_r.get("prediction", []),
            "selected_triplets": llm_r.get("selected_triplets", []),
        })

    # Print hop distribution
    total = len(per_sample_analysis)
    logger.info(f"\n  Hop Distribution (from LLM results, model-ranked top-{TOP_K} triplets):")
    logger.info(f"  {'Class':<12} {'Count':<8} {'Percent':<10}")
    logger.info(f"  {'-'*30}")
    all_classes = ["1-hop", "2-hop", "3-hop", "3+hop", "no-path", "no-entity", "unknown-hop"]
    for cls in all_classes:
        cnt = hop_counter.get(cls, 0)
        if cnt == 0:
            continue
        pct = cnt / total * 100 if total > 0 else 0
        logger.info(f"  {cls:<12} {cnt:<8} {pct:.1f}%")
    logger.info(f"  {'Total':<12} {total:<8}")

    # --- Step 2 & 3: Per-class Hit scores and coverage ---
    logger.info("\n--- Step 2 & 3: Per-Class Hit Score + Coverage ---")

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

        metrics["hit_list"].append(sample["hit"])
        metrics["hit1_list"].append(sample["hit_at_1"])
        metrics["f1_list"].append(sample["f1"])
        metrics["precision_list"].append(sample["precision"])
        metrics["recall_list"].append(sample["recall"])

    # Print per-class table
    logger.info(f"\n  {'Class':<12} {'N':<6} {'Hit%':<8} {'Hit@1%':<8} {'F1%':<8} {'AnsCov%':<9} {'PathCov%':<9}")
    logger.info(f"  {'-'*62}")

    summary_table = {}
    for cls in all_classes:
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

    # --- Overall metrics ---
    overall_ans_cov = sum(1 for s in per_sample_analysis if s["answer_coverage"]) / total * 100
    overall_path_cov = sum(1 for s in per_sample_analysis if s["path_coverage"]) / total * 100
    overall_hit = np.mean([s["hit"] for s in per_sample_analysis]) * 100
    overall_hit1 = np.mean([s["hit_at_1"] for s in per_sample_analysis]) * 100
    overall_f1 = np.mean([s["f1"] for s in per_sample_analysis]) * 100

    logger.info(f"  {'-'*62}")
    logger.info(f"  {'OVERALL':<12} {total:<6} {overall_hit:<8.1f} {overall_hit1:<8.1f} {overall_f1:<8.1f} {overall_ans_cov:<9.1f} {overall_path_cov:<9.1f}")

    # --- Step 4: Generate failure.json ---
    logger.info("\n--- Step 4: Generating failure.json ---")

    failures = []
    for sample in per_sample_analysis:
        # Failure = answer entity is NOT present in model-ranked top-k triplets
        if not sample["answer_coverage"]:
            failure_entry = {
                "question": sample["question"],
                "q_entity": sample["q_entity"],
                "a_entity": sample["a_entity"],
                "hop_class": sample["hop_class"],
                "answer_coverage": False,
                "path_coverage": sample["path_coverage"],
                "llm_prediction": sample["prediction"],
                "llm_hit": sample["hit"],
                "llm_f1": sample["f1"],
                "top_100_triplets": sample["selected_triplets"],
            }
            failures.append(failure_entry)

    failure_path = os.path.join(args.output_dir, "failure.json")
    with open(failure_path, "w") as f:
        json.dump(failures, f, indent=2)
    logger.info(f"  Saved {len(failures)} failure cases to: {failure_path}")
    logger.info(f"  (These are questions where answer entity is NOT in the model-ranked triplets fed to LLM)")

    # --- Also generate failures where entity IS present but Hit=0 ---
    llm_failures = []
    for sample in per_sample_analysis:
        if sample["answer_coverage"] and sample["hit"] == 0:
            llm_failures.append({
                "question": sample["question"],
                "q_entity": sample["q_entity"],
                "a_entity": sample["a_entity"],
                "hop_class": sample["hop_class"],
                "answer_coverage": True,
                "path_coverage": sample["path_coverage"],
                "llm_prediction": sample["prediction"],
                "ground_truth": sample["a_entity"],
                "llm_hit": 0,
                "llm_f1": sample["f1"],
                "top_100_triplets": sample["selected_triplets"],
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
            "total_samples": total,
            "source": "llm_detailed_results.json (model-ranked triplets)",
        },
        "overall_metrics": {
            "hit": round(overall_hit, 2),
            "hit_at_1": round(overall_hit1, 2),
            "f1": round(overall_f1, 2),
            "answer_coverage_pct": round(overall_ans_cov, 2),
            "path_coverage_pct": round(overall_path_cov, 2),
        },
        "hop_distribution": {cls: hop_counter.get(cls, 0) for cls in all_classes if hop_counter.get(cls, 0) > 0},
        "per_class_metrics": summary_table,
        "failure_summary": {
            "total_missing_entity": len(failures),
            "total_llm_wrong_with_entity": len(llm_failures),
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
            f"[RETRIEVAL/MODEL] {entity_miss_pct:.0f}% of test questions have answer entity MISSING "
            f"from the model-ranked top-{TOP_K} triplets. This is the model's retrieval ceiling — "
            f"no prompt can fix what the model doesn't supply. Consider: "
            f"(a) Increasing k beyond 100, (b) Improving the reward signal to favor answer-containing paths, "
            f"(c) Adding answer-entity-aware re-ranking during training."
        )

    # Suggestion based on LLM failures with entity present
    if llm_failures:
        llm_fail_pct = len(llm_failures) / total * 100
        suggestions.append(
            f"[PROMPT/LLM] {len(llm_failures)} questions ({llm_fail_pct:.1f}%) have the answer entity "
            f"in the model-ranked triplets but the LLM still gets Hit=0. This is a prompt/inference issue. "
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
    )
    parser.add_argument("--dataset", type=str, default="cwq", choices=["cwq", "webqsp"],
                        help="Dataset to analyze (default: cwq)")

    args = parser.parse_args()

    # Select paths based on dataset
    if args.dataset == "cwq":
        test_data_path = CWQ_TEST_DATA
        llm_results_dir = CWQ_LLM_RESULTS_DIR
        output_dir = CWQ_OUTPUT_DIR
    else:
        test_data_path = WEBQSP_TEST_DATA
        llm_results_dir = WEBQSP_LLM_RESULTS_DIR
        output_dir = WEBQSP_OUTPUT_DIR

    llm_detailed_path = os.path.join(llm_results_dir, "llm_detailed_results.json")
    if not os.path.exists(llm_detailed_path):
        print(f"ERROR: LLM results not found: {llm_detailed_path}")
        sys.exit(1)

    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data not found: {test_data_path}")
        sys.exit(1)

    # Create a namespace to pass to run_analysis
    class Args:
        pass

    run_args = Args()
    run_args.test_data = test_data_path
    run_args.llm_results_dir = llm_results_dir
    run_args.output_dir = output_dir

    run_analysis(run_args)


if __name__ == "__main__":
    main()
