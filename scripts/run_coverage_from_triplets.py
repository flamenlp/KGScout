#!/usr/bin/env python3
"""
Compute coverage metrics (answer_coverage, path_coverage) from selected_triplets.json.

No model or dataset loading required — reads directly from the triplet selection output.

Usage:
    python scripts/run_coverage_from_triplets.py <selected_triplets.json> <output_file>
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import compute_answer_coverage, compute_path_coverage


def parse_linearized_triplet(linearized: str):
    """
    Parse a linearized triplet string back into (subject, relation, object).

    Format: "subject, relation, object"
    Uses rsplit to handle relations that may contain commas.
    """
    parts = linearized.split(", ", 1)
    if len(parts) < 2:
        return None
    subject = parts[0]
    rest = parts[1]
    # Split from right to separate relation from object
    r_parts = rest.rsplit(", ", 1)
    if len(r_parts) < 2:
        return None
    relation = r_parts[0]
    obj = r_parts[1]
    return (subject, relation, obj)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/run_coverage_from_triplets.py "
            "<selected_triplets.json> <output_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    answer_cov_count = 0
    path_cov_count = 0

    for entry in data:
        a_entities = entry.get("a_entity", [])
        q_entities = entry.get("q_entity", [])
        reranker = entry.get("reranker", [])

        if not a_entities or not reranker:
            continue

        # Parse linearized triplets back into tuples
        triplets = []
        for lin in reranker:
            parsed = parse_linearized_triplet(lin)
            if parsed:
                triplets.append(parsed)

        if not triplets:
            continue

        total += 1

        if compute_answer_coverage(triplets, a_entities):
            answer_cov_count += 1

        if compute_path_coverage(triplets, q_entities, a_entities):
            path_cov_count += 1

    metrics = {
        "total_samples": total,
        "answer_coverage": answer_cov_count / total if total > 0 else 0.0,
        "path_coverage": path_cov_count / total if total > 0 else 0.0,
        "answer_coverage_count": answer_cov_count,
        "path_coverage_count": path_cov_count,
    }

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Coverage metrics saved to: {output_file}")
    print(f"  Total samples:    {total}")
    print(f"  Answer Coverage:  {metrics['answer_coverage']:.4f} ({answer_cov_count}/{total})")
    print(f"  Path Coverage:    {metrics['path_coverage']:.4f} ({path_cov_count}/{total})")


if __name__ == "__main__":
    main()
