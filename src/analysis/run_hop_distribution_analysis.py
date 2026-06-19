#!/usr/bin/env python3
"""
Hop Distribution Analysis for Training Data (Top-100 Reasoning Paths).

Analyzes the distribution of 1-hop, 2-hop, and 3-hop reasoning paths
in the top-100 triplets per question for the training dataset.

Context:
--------
In KGScout, each training sample contains a pool of candidate triplets
(stored in `topk_rel_data` as (cosine_score, (subject, relation, object)) tuples).
Each individual triplet is a single KG edge (1-hop). Multi-hop reasoning paths
emerge when triplets chain together: the object of one triplet matches the
subject of another.

This script determines, for each question's top-100 triplets:
  - How many 1-hop paths exist (direct q_entity → a_entity in one triplet)
  - How many 2-hop paths exist (q_entity → intermediate → a_entity via 2 triplets)
  - How many 3-hop paths exist (q_entity → ... → a_entity via 3 triplets)

Impact on Learning:
-------------------
The REINFORCE reward (compute_reward_v8) uses linear decay:
    conn = max(0, 1 - lambda_lin * (distance - 1))   [lambda_lin=0.2]

This means:
  - 1-hop path: reward_conn = 1.0
  - 2-hop path: reward_conn = 0.8
  - 3-hop path: reward_conn = 0.6
  - 4-hop path: reward_conn = 0.4
  - 5-hop path: reward_conn = 0.2

If the top-100 triplets are dominated by 1-hop connections, the model receives
strong reward signal for simple paths but weaker gradient signal for learning
multi-hop reasoning. For CWQ (which requires multi-hop reasoning), this could
mean the model under-learns complex path selection.

With only 3000 subsampled datapoints for CWQ training, a skewed hop distribution
further reduces the effective training signal for multi-hop questions.

Usage:
    python run_hop_distribution_analysis.py --data-path <path_to_train_data.pt> \
                                            --output-dir ./results/hop_analysis/ \
                                            [--top-k 100] \
                                            [--max-samples 3000]
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required for torch.load() to unpickle datasets saved from notebooks
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("hop_distribution_analysis")


def setup_logging(log_file: Optional[str] = None):
    """Configure logging to console and optionally to file."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


# ============================================================================
# HOP ANALYSIS FUNCTIONS
# ============================================================================

def count_hop_paths(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
    max_hops: int = 3
) -> Dict[str, int]:
    """
    Count the number of reasoning paths of different hop lengths
    between question entities and answer entities.

    A k-hop path means there exists a path of exactly k edges in the
    directed graph (built from selected triplets) from some q_entity
    to some a_entity.

    Args:
        triplets: List of (subject, relation, object) tuples
        q_entities: List of question entity strings
        a_entities: List of answer entity strings
        max_hops: Maximum hop count to check (default: 3)

    Returns:
        Dictionary with keys:
            - 'hop_1': number of q-a pairs connected by exactly 1 hop
            - 'hop_2': number of q-a pairs connected by exactly 2 hops
            - 'hop_3': number of q-a pairs connected by exactly 3 hops
            - 'no_path': number of q-a pairs with no path
            - 'total_qa_pairs': total number of (q, a) pairs checked
            - 'shortest_path_lengths': list of all shortest path lengths found
            - 'has_any_path': whether any q-a path exists
    """
    if not triplets or not q_entities or not a_entities:
        total_pairs = len(q_entities) * len(a_entities) if q_entities and a_entities else 0
        return {
            'hop_1': 0,
            'hop_2': 0,
            'hop_3': 0,
            'no_path': total_pairs,
            'total_qa_pairs': total_pairs,
            'shortest_path_lengths': [],
            'has_any_path': False
        }

    # Build directed graph from triplets
    G = nx.DiGraph()
    for s, r, o in triplets:
        s_l, o_l = s.lower(), o.lower()
        G.add_edge(s_l, o_l, relation=r.lower())

    # Also build undirected graph for more lenient path finding
    G_undirected = G.to_undirected()

    hop_counts = defaultdict(int)
    shortest_paths = []
    total_pairs = 0

    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            total_pairs += 1

            # Try directed graph first
            path_length = None
            try:
                path_length = nx.shortest_path_length(G, qn, an)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Try undirected graph as fallback
                try:
                    path_length = nx.shortest_path_length(G_undirected, qn, an)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            if path_length is not None:
                shortest_paths.append(path_length)
                if path_length <= max_hops:
                    hop_counts[path_length] += 1
                else:
                    hop_counts['longer'] = hop_counts.get('longer', 0) + 1
            else:
                hop_counts['no_path'] = hop_counts.get('no_path', 0) + 1

    return {
        'hop_1': hop_counts.get(1, 0),
        'hop_2': hop_counts.get(2, 0),
        'hop_3': hop_counts.get(3, 0),
        'longer_than_3': hop_counts.get('longer', 0),
        'no_path': hop_counts.get('no_path', 0),
        'total_qa_pairs': total_pairs,
        'shortest_path_lengths': shortest_paths,
        'has_any_path': len(shortest_paths) > 0
    }


def analyze_sample(
    sample: Dict,
    top_k: int = 100
) -> Dict:
    """
    Analyze hop distribution for a single training sample.

    Args:
        sample: Dataset sample dictionary
        top_k: Number of top triplets to consider (default: 100)

    Returns:
        Analysis results for this sample
    """
    # Extract triplets from topk_rel_data (already sorted by cosine similarity)
    topk_rel_data = sample.get("topk_rel_data", [])
    triplets = [t[1] for t in topk_rel_data[:top_k]]

    # Extract entities
    q_entities = sample.get("q_entity", [])
    a_entities = sample.get("a_entity", [])
    question = sample.get("question", "")

    # Count hops
    hop_info = count_hop_paths(triplets, q_entities, a_entities)

    return {
        'question': question,
        'q_entities': q_entities,
        'a_entities': a_entities,
        'num_triplets_available': len(topk_rel_data),
        'num_triplets_used': min(top_k, len(topk_rel_data)),
        **hop_info
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(
    data_path: str,
    output_dir: str,
    top_k: int = 100,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Run hop distribution analysis on training data.

    Args:
        data_path: Path to training data (.pt file)
        output_dir: Directory to save results
        top_k: Number of top triplets to analyze per question
        max_samples: Maximum number of samples to analyze (None = all)

    Returns:
        Aggregate statistics dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading training data from: {data_path}")
    data = torch.load(data_path, weights_only=False, map_location="cpu")

    # Handle both raw list and dataset object
    if isinstance(data, list):
        dataset = data
    elif hasattr(data, 'precomputed_data'):
        dataset = data.precomputed_data
    elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
        dataset = data
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    total_samples = len(dataset)
    logger.info(f"Total samples in dataset: {total_samples}")

    if max_samples and max_samples < total_samples:
        logger.info(f"Limiting analysis to first {max_samples} samples (simulating CWQ 3000 subsample)")
        num_to_analyze = max_samples
    else:
        num_to_analyze = total_samples

    # Run analysis
    logger.info(f"Analyzing top-{top_k} triplets per question...")
    per_sample_results = []
    aggregate = {
        'hop_1_total': 0,
        'hop_2_total': 0,
        'hop_3_total': 0,
        'longer_than_3_total': 0,
        'no_path_total': 0,
        'samples_with_1hop': 0,
        'samples_with_2hop': 0,
        'samples_with_3hop': 0,
        'samples_with_any_path': 0,
        'samples_with_no_path': 0,
        'all_shortest_paths': [],
    }

    for idx in tqdm(range(num_to_analyze), desc="Analyzing hop distribution"):
        try:
            sample = dataset[idx]
            result = analyze_sample(sample, top_k=top_k)
            per_sample_results.append(result)

            # Aggregate
            aggregate['hop_1_total'] += result['hop_1']
            aggregate['hop_2_total'] += result['hop_2']
            aggregate['hop_3_total'] += result['hop_3']
            aggregate['longer_than_3_total'] += result.get('longer_than_3', 0)
            aggregate['no_path_total'] += result['no_path']
            aggregate['all_shortest_paths'].extend(result['shortest_path_lengths'])

            if result['hop_1'] > 0:
                aggregate['samples_with_1hop'] += 1
            if result['hop_2'] > 0:
                aggregate['samples_with_2hop'] += 1
            if result['hop_3'] > 0:
                aggregate['samples_with_3hop'] += 1
            if result['has_any_path']:
                aggregate['samples_with_any_path'] += 1
            else:
                aggregate['samples_with_no_path'] += 1

        except Exception as e:
            logger.warning(f"Sample {idx}: Error during analysis - {e}")
            continue

    # Compute summary statistics
    num_analyzed = len(per_sample_results)
    all_paths = aggregate['all_shortest_paths']

    summary = {
        'config': {
            'data_path': data_path,
            'top_k': top_k,
            'max_samples': max_samples,
            'total_samples_in_dataset': total_samples,
            'samples_analyzed': num_analyzed,
        },
        'hop_distribution': {
            'total_1hop_paths': aggregate['hop_1_total'],
            'total_2hop_paths': aggregate['hop_2_total'],
            'total_3hop_paths': aggregate['hop_3_total'],
            'total_longer_than_3hop': aggregate['longer_than_3_total'],
            'total_no_path_pairs': aggregate['no_path_total'],
        },
        'per_sample_coverage': {
            'samples_with_at_least_one_1hop': aggregate['samples_with_1hop'],
            'samples_with_at_least_one_2hop': aggregate['samples_with_2hop'],
            'samples_with_at_least_one_3hop': aggregate['samples_with_3hop'],
            'samples_with_any_qa_path': aggregate['samples_with_any_path'],
            'samples_with_no_qa_path': aggregate['samples_with_no_path'],
        },
        'percentages': {
            'pct_samples_with_1hop': round(aggregate['samples_with_1hop'] / num_analyzed * 100, 2) if num_analyzed > 0 else 0,
            'pct_samples_with_2hop': round(aggregate['samples_with_2hop'] / num_analyzed * 100, 2) if num_analyzed > 0 else 0,
            'pct_samples_with_3hop': round(aggregate['samples_with_3hop'] / num_analyzed * 100, 2) if num_analyzed > 0 else 0,
            'pct_samples_with_any_path': round(aggregate['samples_with_any_path'] / num_analyzed * 100, 2) if num_analyzed > 0 else 0,
            'pct_samples_with_no_path': round(aggregate['samples_with_no_path'] / num_analyzed * 100, 2) if num_analyzed > 0 else 0,
        },
        'path_length_statistics': {
            'mean_shortest_path': round(float(np.mean(all_paths)), 3) if all_paths else None,
            'median_shortest_path': round(float(np.median(all_paths)), 3) if all_paths else None,
            'std_shortest_path': round(float(np.std(all_paths)), 3) if all_paths else None,
            'min_shortest_path': int(min(all_paths)) if all_paths else None,
            'max_shortest_path': int(max(all_paths)) if all_paths else None,
            'total_paths_found': len(all_paths),
        },
        'learning_impact_analysis': {
            'description': (
                "The REINFORCE reward uses linear decay: conn = max(0, 1 - 0.2*(d-1)). "
                "1-hop gets reward 1.0, 2-hop gets 0.8, 3-hop gets 0.6. "
                "If top-100 triplets are dominated by 1-hop connections, the model "
                "receives strong signal for simple paths but weaker gradient for multi-hop reasoning."
            ),
            'effective_reward_by_hop': {
                '1-hop': 1.0,
                '2-hop': 0.8,
                '3-hop': 0.6,
                '4-hop': 0.4,
                '5-hop': 0.2,
            },
            'hop_ratio_1_vs_2': (
                round(aggregate['hop_1_total'] / max(aggregate['hop_2_total'], 1), 2)
            ),
            'hop_ratio_1_vs_3': (
                round(aggregate['hop_1_total'] / max(aggregate['hop_3_total'], 1), 2)
            ),
            'recommendation': _generate_recommendation(aggregate, num_analyzed),
        }
    }

    # Save results
    summary_path = os.path.join(output_dir, "hop_distribution_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    # Save per-sample results (without the full shortest_path_lengths list for readability)
    per_sample_path = os.path.join(output_dir, "per_sample_hop_analysis.jsonl")
    with open(per_sample_path, 'w') as f:
        for result in per_sample_results:
            # Remove large list for JSONL output
            output_result = {k: v for k, v in result.items() if k != 'shortest_path_lengths'}
            output_result['num_paths_found'] = len(result['shortest_path_lengths'])
            if result['shortest_path_lengths']:
                output_result['min_path_length'] = min(result['shortest_path_lengths'])
                output_result['max_path_length'] = max(result['shortest_path_lengths'])
            f.write(json.dumps(output_result, ensure_ascii=False) + "\n")
    logger.info(f"Per-sample results saved to: {per_sample_path}")

    # Print summary to console
    _print_summary(summary)

    return summary


def _generate_recommendation(aggregate: Dict, num_analyzed: int) -> str:
    """Generate a recommendation based on the hop distribution."""
    if num_analyzed == 0:
        return "No samples analyzed."

    pct_1hop = aggregate['samples_with_1hop'] / num_analyzed * 100
    pct_2hop = aggregate['samples_with_2hop'] / num_analyzed * 100
    pct_3hop = aggregate['samples_with_3hop'] / num_analyzed * 100
    pct_no_path = aggregate['samples_with_no_path'] / num_analyzed * 100

    recommendations = []

    if pct_1hop > 80 and pct_2hop < 40:
        recommendations.append(
            "HIGH BIAS: Top-100 triplets are heavily dominated by 1-hop paths. "
            "The model may under-learn multi-hop reasoning. Consider: "
            "(1) Increasing the candidate pool size, "
            "(2) Using a retrieval strategy that prioritizes multi-hop paths, "
            "(3) Augmenting training data with more multi-hop questions."
        )

    if pct_no_path > 50:
        recommendations.append(
            "LOW COVERAGE: Over 50% of samples have no Q→A path in top-100 triplets. "
            "The model receives zero connectivity reward for these samples. "
            "Consider improving the initial retrieval to include more relevant triplets."
        )

    if aggregate['hop_1_total'] > 5 * aggregate['hop_2_total']:
        recommendations.append(
            "IMBALANCED: 1-hop paths outnumber 2-hop paths by >5x. "
            "Combined with the reward decay (1.0 vs 0.8), this creates strong bias "
            "toward simple path selection. For CWQ multi-hop questions, this may "
            "limit the model's ability to learn complex reasoning chains."
        )

    if not recommendations:
        recommendations.append(
            "Distribution appears reasonable. Monitor training reward curves "
            "to confirm the model is learning multi-hop path selection."
        )

    return " | ".join(recommendations)


def _print_summary(summary: Dict):
    """Print a formatted summary to console."""
    print("\n" + "=" * 70)
    print("HOP DISTRIBUTION ANALYSIS - SUMMARY")
    print("=" * 70)

    config = summary['config']
    print(f"\nDataset: {config['data_path']}")
    print(f"Samples analyzed: {config['samples_analyzed']} / {config['total_samples_in_dataset']}")
    print(f"Top-K triplets per question: {config['top_k']}")

    print("\n" + "-" * 70)
    print("HOP DISTRIBUTION (total across all samples)")
    print("-" * 70)
    hd = summary['hop_distribution']
    print(f"  1-hop paths:       {hd['total_1hop_paths']}")
    print(f"  2-hop paths:       {hd['total_2hop_paths']}")
    print(f"  3-hop paths:       {hd['total_3hop_paths']}")
    print(f"  >3-hop paths:      {hd['total_longer_than_3hop']}")
    print(f"  No path (Q→A):     {hd['total_no_path_pairs']}")

    print("\n" + "-" * 70)
    print("PER-SAMPLE COVERAGE")
    print("-" * 70)
    pct = summary['percentages']
    print(f"  Samples with ≥1 one-hop path:   {pct['pct_samples_with_1hop']:.1f}%")
    print(f"  Samples with ≥1 two-hop path:   {pct['pct_samples_with_2hop']:.1f}%")
    print(f"  Samples with ≥1 three-hop path: {pct['pct_samples_with_3hop']:.1f}%")
    print(f"  Samples with any Q→A path:      {pct['pct_samples_with_any_path']:.1f}%")
    print(f"  Samples with NO Q→A path:       {pct['pct_samples_with_no_path']:.1f}%")

    print("\n" + "-" * 70)
    print("PATH LENGTH STATISTICS")
    print("-" * 70)
    stats = summary['path_length_statistics']
    if stats['mean_shortest_path'] is not None:
        print(f"  Mean shortest path:   {stats['mean_shortest_path']:.3f}")
        print(f"  Median shortest path: {stats['median_shortest_path']:.3f}")
        print(f"  Std deviation:        {stats['std_shortest_path']:.3f}")
        print(f"  Range:                [{stats['min_shortest_path']}, {stats['max_shortest_path']}]")
        print(f"  Total paths found:    {stats['total_paths_found']}")
    else:
        print("  No paths found in any sample.")

    print("\n" + "-" * 70)
    print("LEARNING IMPACT")
    print("-" * 70)
    impact = summary['learning_impact_analysis']
    print(f"  1-hop to 2-hop ratio: {impact['hop_ratio_1_vs_2']}:1")
    print(f"  1-hop to 3-hop ratio: {impact['hop_ratio_1_vs_3']}:1")
    print(f"\n  Recommendation:")
    print(f"    {impact['recommendation']}")

    print("\n" + "=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze hop distribution in top-K training triplets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze full training dataset with top-100 triplets
    python run_hop_distribution_analysis.py \\
        --data-path /path/to/train_jointrainer_path_dataset_v3_ppr.pt \\
        --output-dir ./results/hop_analysis/

    # Analyze CWQ 3000 subsample
    python run_hop_distribution_analysis.py \\
        --data-path /path/to/cwq_train_data.pt \\
        --output-dir ./results/hop_analysis/cwq_3000/ \\
        --max-samples 3000

    # Analyze with different top-k values
    python run_hop_distribution_analysis.py \\
        --data-path /path/to/train_data.pt \\
        --output-dir ./results/hop_analysis/top50/ \\
        --top-k 50
        """
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data file (.pt or .pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/hop_analysis/",
        help="Directory to save analysis results (default: ./results/hop_analysis/)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top triplets to analyze per question (default: 100)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all). "
             "Use 3000 to simulate CWQ training subsample."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file or os.path.join(args.output_dir, "hop_analysis.log")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_file)

    # Validate input
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    # Run analysis
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("HOP DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Output dir: {args.output_dir}")

    summary = run_analysis(
        data_path=args.data_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_samples=args.max_samples
    )

    elapsed = time.time() - start_time
    logger.info(f"\nAnalysis complete. Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
