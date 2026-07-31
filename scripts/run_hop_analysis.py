#!/usr/bin/env python3
"""
Run hop-stratified retriever analysis for KGScout vs cosine baseline.

Classifies test questions by reasoning hop count (1-hop, 2-hop, ≥3-hop, no-path)
using the cosine top-1000 pool graph, then computes ans_present and has_path for
both retrievers at each k value.

Checkpoint resolution (same as statistical-analysis):
  Primary:  results/k-ablation/{dataset}/k{K}/model/main_training_k{K}/
  Fallback: results/full-pipeline/{dataset}/k{K}-N1000/model/main_training_k{K}/

Results saved to: results/hop-analysis/{dataset}/

Usage:
    python scripts/run_hop_analysis.py --dataset cwq
    python scripts/run_hop_analysis.py --dataset webqsp
    python scripts/run_hop_analysis.py --dataset cwq --k-values 30 50 100 150
"""

import argparse
import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.hop_analysis_service import HopAnalysisService


def load_config() -> dict:
    """Load config.yml from project root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yml")

    if not os.path.exists(config_path):
        print(f"Error: config.yml not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Hop-stratified retriever analysis: KGScout vs cosine"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cwq", "webqsp"],
        help="Dataset to analyse (cwq or webqsp)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[30, 50, 100, 150],
        help="K values to analyse (default: 30 50 100 150)",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="./results",
        help="Base results directory (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/hop-analysis/{dataset})",
    )
    parser.add_argument(
        "--sample-k",
        type=int,
        default=1000,
        help="Pool size N for cosine top-N candidates (default: 1000)",
    )

    args = parser.parse_args()

    # Resolve test data path from config.yml
    config = load_config()
    dataset_config = config.get("datasets", {}).get(args.dataset)
    if not dataset_config:
        print(
            f"Error: Dataset '{args.dataset}' not found in config.yml",
            file=sys.stderr,
        )
        sys.exit(1)

    test_data_path = dataset_config.get("test")
    if not test_data_path:
        print(
            f"Error: No 'test' path defined for dataset '{args.dataset}' in config.yml",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found: {test_data_path}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    service = HopAnalysisService()

    try:
        results = service.run_hop_analysis(
            dataset=args.dataset,
            test_data_path=test_data_path,
            k_values=args.k_values,
            results_base=args.results_base,
            output_dir=args.output_dir,
            sample_k=args.sample_k,
        )
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Exit non-zero if any k had an error
    has_errors = any(
        "error" in v for v in results["results"].values()
    )
    if has_errors:
        print(
            "\nWARNING: Some k values had errors. Check output above.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
