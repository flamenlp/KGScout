#!/usr/bin/env python3
"""
Compute coverage metrics (answer_present, path_coverage) and save to JSON.

Usage:
    python scripts/run_coverage.py <model_path> <test_data> <top_k> <output_file>
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.evaluate_service import EvaluateService


def main():
    if len(sys.argv) < 5:
        print("Usage: python scripts/run_coverage.py <model_path> <test_data> <top_k> <output_file>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    test_data = sys.argv[2]
    top_k = int(sys.argv[3])
    output_file = sys.argv[4]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    svc = EvaluateService()
    metrics = svc.evaluate(
        model_path=model_path,
        test_data_path=test_data,
        top_k=top_k,
    )

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Coverage metrics saved to: {output_file}")
    print(f"  Answer Coverage: {metrics.get('answer_coverage', 'N/A')}")
    print(f"  Path Coverage:   {metrics.get('path_coverage', 'N/A')}")


if __name__ == "__main__":
    main()
