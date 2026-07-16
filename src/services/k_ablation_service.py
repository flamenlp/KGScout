"""
K-Value Ablation Service for evaluating different k values with fixed LLM.

This service orchestrates k-value ablation studies:
1. Iterate through specified k-values
2. Run LLM comparison for each k-value using Llama-3.1-8b
3. Generate comparison table across k-values
4. Save individual and summary results

Results saved to: results/llm-inference/{llm}/{kval}/
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime

from src.services.llm_comparison_service import LLMComparisonService


class KAblationService:
    """
    Service for k-value ablation study.

    Runs evaluation across multiple k-values using a fixed LLM (Llama-3.1-8b)
    to analyze how the number of selected triplets affects answer quality.
    """

    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.

        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device
        print("KAblationService initialized")

    def run_ablation(
        self,
        dataset_path: str,
        retriever_type: str,
        k_values: List[int] = None,
        k: int = None,
        model_path: str = None,
        output_dir: str = "results/llm-inference",
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run k-value ablation study with Llama-3.1-8b.

        Args:
            dataset_path: Path to test .pt dataset file
            retriever_type: Retriever type ('kgscout' or 'cosine')
            k_values: List of k values to test (default: [30, 50, 100, 150])
            k: Optional single k value (overrides k_values)
            model_path: Path to model checkpoint (required for kgscout)
            output_dir: Base output directory
            sample_k: Number of triplets to feed to model (default: 1000)

        Returns:
            Dictionary with results for each k value and comparison table
        """
        print("=" * 60)
        print("K-VALUE ABLATION STUDY")
        print("=" * 60)
        print(f"  Dataset: {dataset_path}")
        print(f"  LLM: llama (fixed)")
        print(f"  Retriever: {retriever_type}")

        # Determine k-values
        if k is not None:
            k_values_to_test = [k]
        elif k_values is not None and len(k_values) > 0:
            k_values_to_test = k_values
        else:
            k_values_to_test = [30, 50, 100, 150]

        print(f"  K values: {k_values_to_test}")
        print("=" * 60)

        results_by_k = {}

        for idx, k_val in enumerate(k_values_to_test, 1):
            print(f"\n{'='*60}")
            print(f"  k={k_val} ({idx}/{len(k_values_to_test)})")
            print(f"{'='*60}")

            service = LLMComparisonService(device=self.device)

            try:
                results = service.run_comparison(
                    llm_model="llama",
                    k=k_val,
                    dataset_path=dataset_path,
                    model_path=model_path,
                    retriever_type=retriever_type,
                    output_dir=output_dir,
                    sample_k=sample_k,
                )
                results_by_k[k_val] = results
            except Exception as e:
                print(f"\n  Error for k={k_val}: {e}")
                results_by_k[k_val] = {"error": str(e)}

        # Print comparison table
        comparison_table = self._generate_comparison_table(results_by_k)
        print(comparison_table)

        # Save summary
        summary_dir = os.path.join(output_dir, "llama", "k-ablation-summary")
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, "k_ablation_summary.json")

        summary_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "retriever_type": retriever_type,
                "llm_model": "llama",
                "k_values": k_values_to_test,
            },
            "results_by_k": {str(k): r for k, r in results_by_k.items()},
            "comparison_table": comparison_table,
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

        return {"results_by_k": results_by_k, "comparison_table": comparison_table}

    def _generate_comparison_table(self, results_by_k: Dict[int, Dict]) -> str:
        """Generate formatted comparison table."""
        k_values = sorted(results_by_k.keys())
        table = f"\n{'='*90}\nK-VALUE ABLATION COMPARISON\n{'='*90}\n"
        table += f"{'k':<8} {'Hit':<10} {'Hit@1':<10} {'F1':<10} {'Prec':<10} {'Recall':<10} {'EM':<10}\n"
        table += "-" * 90 + "\n"

        for k in k_values:
            r = results_by_k[k]
            if "error" in r:
                table += f"{k:<8} ERROR\n"
            else:
                table += (
                    f"{k:<8} {r.get('hit', 0):<10.2f} {r.get('hit_at_1', 0):<10.2f} "
                    f"{r.get('macro_f1', 0):<10.2f} {r.get('macro_precision', 0):<10.2f} "
                    f"{r.get('macro_recall', 0):<10.2f} {r.get('exact_match', 0):<10.2f}\n"
                )

        table += "=" * 90 + "\n"
        return table
