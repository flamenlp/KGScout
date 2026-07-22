#!/usr/bin/env python3
"""
Cosine Reachability Analysis Script.

Computes answer entity presence and path coverage for the cosine-sorted
candidate pool at Top-N = 1000 and Top-N = 1500, on both WebQSP and CWQ test sets.

This quantifies the ceiling imposed by the initial cosine similarity filtering stage:
if a relevant triplet is not in the Top-N pool, the downstream PathRankingModel
cannot recover it.

Reads dataset paths from config.yml (consistent with justfile full-pipeline).
Uses compute_answer_coverage and compute_path_coverage from src.utils.metrics.

Usage:
    python scripts/check_cosine_reachability.py
    python scripts/check_cosine_reachability.py --output-dir results/rebuttal/cosine-ceiling
    python scripts/check_cosine_reachability.py --datasets cwq
    python scripts/check_cosine_reachability.py --top-n 500 1000 1500 2000
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

import yaml
import torch
from tqdm import tqdm

# Add project root to path for imports (script lives in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import compute_answer_coverage, compute_path_coverage
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR

# Allow loading datasets saved from notebooks where JointTrainingDatasetv3PPR
# was defined in __main__. torch.load/unpickle looks up __main__ for the class.
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# ─── Configuration ────────────────────────────────────────────────────────────

# Default Top-N values to evaluate
DEFAULT_TOP_N_VALUES = [1000, 1500]

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "cosine_reachability.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Config Reader ────────────────────────────────────────────────────────────


def load_config() -> Dict:
    """Load config.yml from project root (one level up from scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yml not found at: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_test_dataset_path(cfg: Dict, dataset_name: str) -> str:
    """
    Get the test dataset path from config.yml.

    Follows the same resolution logic as scripts/read_config.py used by the justfile.

    Args:
        cfg: Parsed config.yml dict.
        dataset_name: 'webqsp' or 'cwq'.

    Returns:
        Absolute path to the test .pt dataset file.
    """
    datasets = cfg.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[dataset_name]["test"]


# ─── Core Analysis ────────────────────────────────────────────────────────────


def analyze_dataset(
    dataset_path: str,
    dataset_name: str,
    top_n_values: List[int],
) -> Dict[str, Any]:
    """
    Analyze cosine reachability for a single dataset at multiple Top-N thresholds.

    The .pt dataset stores triplets in topk_rel_data as:
        [(score, (subject, relation, object)), ...]
    already sorted by cosine similarity (descending).

    We slice the first N triplets and compute:
      - Answer entity presence (src.utils.metrics.compute_answer_coverage)
      - Path coverage (src.utils.metrics.compute_path_coverage)

    Args:
        dataset_path: Path to the .pt dataset file.
        dataset_name: Human-readable dataset name (for logging).
        top_n_values: List of Top-N values to evaluate.

    Returns:
        Dict with per-N metrics and overall statistics.
    """
    logger.info(f"Loading dataset: {dataset_name} from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please ensure the preprocessed test data exists at the expected location."
        )

    data = torch.load(dataset_path, weights_only=False, map_location="cpu")
    logger.info(f"  Loaded {len(data)} samples")

    results_by_n = {}

    for top_n in top_n_values:
        logger.info(f"  Evaluating Top-N = {top_n}...")

        ans_present_count = 0
        path_exists_count = 0
        total_valid = 0
        skipped = 0

        for idx, sample in enumerate(tqdm(data, desc=f"  {dataset_name} Top-{top_n}")):
            try:
                # Extract metadata
                q_entities = sample.get("q_entity", [])
                a_entities = sample.get("a_entity", [])
                is_empty = sample.get("is_empty", False)

                # Skip empty/invalid samples
                if is_empty:
                    skipped += 1
                    continue
                if not a_entities or not q_entities:
                    skipped += 1
                    continue

                # Extract triplets from topk_rel_data
                # Format: [(score, (subject, relation, object)), ...]
                topk_rel_data = sample.get("topk_rel_data", [])
                if not topk_rel_data:
                    skipped += 1
                    continue

                # Take first top_n triplets (pre-sorted by cosine similarity)
                n = min(top_n, len(topk_rel_data))
                selected_triplets = []
                for i in range(n):
                    item = topk_rel_data[i]
                    # item is (score, (subject, relation, object))
                    triplet = item[1]
                    if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                        selected_triplets.append(tuple(triplet))

                if not selected_triplets:
                    skipped += 1
                    continue

                # Compute metrics using src.utils.metrics functions
                # These functions handle lowercasing internally
                ans_present = compute_answer_coverage(selected_triplets, a_entities)
                path_exists = compute_path_coverage(selected_triplets, q_entities, a_entities)

                if ans_present:
                    ans_present_count += 1
                if path_exists:
                    path_exists_count += 1
                total_valid += 1

            except Exception as e:
                logger.warning(f"  Sample {idx}: Error - {e}")
                skipped += 1
                continue

        # Compute percentages
        ans_coverage_pct = (ans_present_count / total_valid * 100) if total_valid > 0 else 0.0
        path_coverage_pct = (path_exists_count / total_valid * 100) if total_valid > 0 else 0.0

        results_by_n[top_n] = {
            "top_n": top_n,
            "total_samples": len(data),
            "valid_samples": total_valid,
            "skipped_samples": skipped,
            "answer_entity_present": ans_present_count,
            "answer_entity_present_pct": round(ans_coverage_pct, 2),
            "reasoning_path_exists": path_exists_count,
            "reasoning_path_exists_pct": round(path_coverage_pct, 2),
            "not_reachable_count": total_valid - path_exists_count,
            "not_reachable_pct": round(
                (total_valid - path_exists_count) / total_valid * 100, 2
            ) if total_valid > 0 else 0.0,
        }

        logger.info(f"    Valid samples: {total_valid}")
        logger.info(
            f"    Answer entity present: {ans_present_count}/{total_valid} "
            f"({ans_coverage_pct:.2f}%)"
        )
        logger.info(
            f"    Reasoning path exists: {path_exists_count}/{total_valid} "
            f"({path_coverage_pct:.2f}%)"
        )
        logger.info(
            f"    NOT reachable: {total_valid - path_exists_count}/{total_valid} "
            f"({(total_valid - path_exists_count) / total_valid * 100:.2f}%)"
        )

    # Compute delta between top-1000 and top-1500
    delta = {}
    if 1000 in results_by_n and 1500 in results_by_n:
        r1000 = results_by_n[1000]
        r1500 = results_by_n[1500]
        delta = {
            "answer_entity_gain": r1500["answer_entity_present"] - r1000["answer_entity_present"],
            "answer_entity_gain_pct_points": round(
                r1500["answer_entity_present_pct"] - r1000["answer_entity_present_pct"], 2
            ),
            "path_coverage_gain": r1500["reasoning_path_exists"] - r1000["reasoning_path_exists"],
            "path_coverage_gain_pct_points": round(
                r1500["reasoning_path_exists_pct"] - r1000["reasoning_path_exists_pct"], 2
            ),
            "description": "Gain from expanding pool from Top-1000 to Top-1500",
        }

    return {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "total_samples": len(data),
        "results_by_top_n": results_by_n,
        "delta_1000_to_1500": delta,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check cosine reachability (answer presence + path coverage) "
            "at Top-N for WebQSP and CWQ test sets. "
            "Reads dataset paths from config.yml."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/check_cosine_reachability.py
    python scripts/check_cosine_reachability.py --output-dir results/rebuttal/cosine-ceiling
    python scripts/check_cosine_reachability.py --datasets cwq
    python scripts/check_cosine_reachability.py --datasets webqsp cwq --top-n 500 1000 1500 2000
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cosine-reachability",
        help="Directory to save results JSON (default: results/cosine-reachability)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["webqsp", "cwq"],
        default=["webqsp", "cwq"],
        help="Datasets to analyze (default: both webqsp and cwq)",
    )
    parser.add_argument(
        "--top-n",
        nargs="+",
        type=int,
        default=DEFAULT_TOP_N_VALUES,
        help="Top-N values to evaluate (default: 1000 1500)",
    )

    args = parser.parse_args()

    # ─── Load config.yml ──────────────────────────────────────────────────────
    cfg = load_config()

    logger.info("=" * 70)
    logger.info("COSINE REACHABILITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"  Datasets: {args.datasets}")
    logger.info(f"  Top-N values: {args.top_n}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 70)

    all_results = {}

    for dataset_name in args.datasets:
        try:
            dataset_path = get_test_dataset_path(cfg, dataset_name)
        except ValueError as e:
            logger.error(str(e))
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name.upper()}")
        logger.info(f"  Path: {dataset_path}")
        logger.info(f"{'='*60}")

        try:
            result = analyze_dataset(dataset_path, dataset_name, args.top_n)
            all_results[dataset_name] = result
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No datasets were successfully analyzed. Exiting.")
        sys.exit(1)

    # ─── Summary Table ────────────────────────────────────────────────────────

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*80}")
    header = f"{'Dataset':<10} {'Top-N':<8} {'Answer Presence':<20} {'Path Coverage':<20} {'Not Reachable':<15}"
    logger.info(header)
    logger.info("-" * 73)

    for ds_name, ds_result in all_results.items():
        for top_n, metrics in ds_result["results_by_top_n"].items():
            logger.info(
                f"{ds_name:<10} {top_n:<8} "
                f"{metrics['answer_entity_present']}/{metrics['valid_samples']} "
                f"({metrics['answer_entity_present_pct']:.1f}%)   "
                f"{metrics['reasoning_path_exists']}/{metrics['valid_samples']} "
                f"({metrics['reasoning_path_exists_pct']:.1f}%)   "
                f"{metrics['not_reachable_count']} ({metrics['not_reachable_pct']:.1f}%)"
            )
        # Print delta
        delta = ds_result.get("delta_1000_to_1500", {})
        if delta:
            logger.info(
                f"{'':10} {'Δ':<8} "
                f"+{delta['answer_entity_gain']} (+{delta['answer_entity_gain_pct_points']:.1f}pp)       "
                f"+{delta['path_coverage_gain']} (+{delta['path_coverage_gain_pct_points']:.1f}pp)"
            )
        logger.info("-" * 73)

    logger.info(f"{'='*80}")

    # ─── Save Results ─────────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, "cosine_reachability_results.json")

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "description": (
                "Cosine reachability analysis: measures the ceiling imposed by "
                "the initial cosine similarity filtering. Shows what fraction of "
                "test questions have answer entities present and reasoning paths "
                "reachable within the Top-N cosine-ranked candidate pool."
            ),
            "top_n_values": args.top_n,
            "datasets_analyzed": list(all_results.keys()),
        },
        "results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")

    # Also save a compact summary for quick reference
    summary_file = os.path.join(args.output_dir, "summary.json")
    summary = {"metadata": output_data["metadata"], "summary": {}}

    for ds_name, ds_result in all_results.items():
        first_n = args.top_n[0] if args.top_n else None
        summary["summary"][ds_name] = {
            "total_valid_samples": (
                ds_result["results_by_top_n"][first_n]["valid_samples"]
                if first_n and first_n in ds_result["results_by_top_n"]
                else 0
            ),
            "metrics_by_top_n": {},
            "delta_1000_to_1500": ds_result.get("delta_1000_to_1500", {}),
        }
        for top_n, metrics in ds_result["results_by_top_n"].items():
            summary["summary"][ds_name]["metrics_by_top_n"][str(top_n)] = {
                "answer_entity_present_pct": metrics["answer_entity_present_pct"],
                "reasoning_path_exists_pct": metrics["reasoning_path_exists_pct"],
                "not_reachable_pct": metrics["not_reachable_pct"],
            }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary saved to: {summary_file}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
