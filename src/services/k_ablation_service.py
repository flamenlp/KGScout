"""
K-Value Ablation Service for evaluating different k values with fixed LLM.

This service orchestrates k-value ablation studies:
1. Iterate through specified k-values
2. Run LLM comparison for each k-value using Llama-3.1-8b
3. Generate comparison table across k-values
4. Save individual and summary results
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime

from src.services.llm_comparison_service import LLMComparisonService


class KAblationService:
    """
    Service for k-value ablation study.
    
    This service runs evaluation across multiple k-values using a fixed LLM (Llama-3.1-8b)
    to analyze how the number of selected triplets affects answer quality.
    
    Requirements:
        - 2.1: Iterate through all specified k-values and run evaluation for each
        - 2.2: Use Llama-3.1-8b as the fixed LLM for all k-value experiments
        - 2.3: Generate new Selected_JSON files with the corresponding number of triplets for each k-value
        - 2.4: Generate a comparison table showing metrics across all k-values
        - 2.5: Save individual results for each k-value and a summary comparison file
        - 2.6: Use default values [30, 50, 100, 150] when k-values list is empty
    """
    
    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.
        
        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device
        print(f"KAblationService initialized")
    
    def run_ablation(
        self,
        dataset: str,
        retriever_type: str,
        k_values: List[int] = None,
        k: int = None,
        model_path: str = None,
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Run k-value ablation study with Llama-3.1-8b.
        
        Pipeline steps:
        1. Determine k-values to test (use provided list, single k, or defaults)
        2. For each k-value:
           a. Run LLMComparisonService with Llama-3.1-8b
           b. Store results
        3. Generate comparison table across all k-values
        4. Save individual and summary results
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
            retriever_type: Retriever type ('kgscout' or 'cosine')
            k_values: List of k values to test (default: [30, 50, 100, 150])
            k: Optional specific k value for single run (overrides k_values)
            model_path: Path to KGscout model (required if retriever_type='kgscout')
            output_dir: Directory to save results (default: 'results')
        
        Returns:
            Dictionary with results for each k value and comparison table
        
        Requirements:
            - 2.1: Iterate through all specified k-values and run evaluation for each
            - 2.2: Use Llama-3.1-8b as the fixed LLM for all k-value experiments
            - 2.3: Generate new Selected_JSON files with the corresponding number of triplets
            - 2.4: Generate a comparison table showing metrics across all k-values
            - 2.5: Save individual results for each k-value and a summary comparison file
            - 2.6: Use default values [30, 50, 100, 150] when k-values list is empty
        """
        print("=" * 60)
        print("K-VALUE ABLATION STUDY")
        print("=" * 60)
        print(f"Dataset: {dataset}")
        print(f"LLM Model: llama (Llama-3.1-8b) [FIXED]")
        print(f"Retriever: {retriever_type}")
        print(f"Output directory: {output_dir}")
        
        # Determine k-values to test
        if k is not None:
            # Single k value provided
            k_values_to_test = [k]
            print(f"Testing single k-value: {k}")
        elif k_values is not None and len(k_values) > 0:
            # List of k values provided
            k_values_to_test = k_values
            print(f"Testing k-values: {k_values_to_test}")
        else:
            # Use default k-values
            k_values_to_test = [30, 50, 100, 150]
            print(f"Using default k-values: {k_values_to_test}")
        
        print("=" * 60)
        
        # Store results for each k-value
        results_by_k = {}
        
        # Iterate through k-values
        for idx, k_val in enumerate(k_values_to_test, 1):
            print(f"\n{'=' * 60}")
            print(f"Evaluating k={k_val} ({idx}/{len(k_values_to_test)})...")
            print(f"{'=' * 60}")
            
            # Reuse LLMComparisonService for each k value with Llama-3.1-8b fixed
            comparison_service = LLMComparisonService(device=self.device)
            
            try:
                results = comparison_service.run_comparison(
                    dataset=dataset,
                    llm_model="llama",  # Fixed to Llama-3.1-8b
                    retriever_type=retriever_type,
                    k=k_val,
                    model_path=model_path,
                    output_dir=output_dir
                )
                
                # Store results
                results_by_k[k_val] = results
                
                print(f"\nResults for k={k_val}:")
                print(f"  Hit:       {results['hit']:.4f}")
                print(f"  Hit@1:     {results['hit_at_1']:.4f}")
                print(f"  Macro F1:  {results['macro_f1']:.4f}")
                print(f"  Precision: {results['macro_precision']:.4f}")
                print(f"  Recall:    {results['macro_recall']:.4f}")
                print(f"  Exact Match: {results['exact_match']:.4f}")
                
            except Exception as e:
                print(f"\nError: Failed to evaluate k={k_val}. Error: {str(e)}")
                print(f"Suggestion: Check model path and dataset availability")
                # Store error result
                results_by_k[k_val] = {
                    "error": str(e),
                    "hit": 0.0,
                    "hit_at_1": 0.0,
                    "macro_f1": 0.0,
                    "macro_precision": 0.0,
                    "macro_recall": 0.0,
                    "exact_match": 0.0
                }
        
        # Generate comparison table
        comparison_table = self._generate_comparison_table(results_by_k)
        
        # Save ablation results
        output_file = self._save_ablation_results(
            results_by_k,
            comparison_table,
            dataset,
            retriever_type,
            output_dir
        )
        
        print("\n" + "=" * 60)
        print("K-VALUE ABLATION STUDY COMPLETE")
        print("=" * 60)
        print(f"Summary saved to: {output_file}")
        print("=" * 60)
        
        return {
            'results_by_k': results_by_k,
            'comparison_table': comparison_table,
            'output_file': output_file
        }

    
    def _generate_comparison_table(self, results_by_k: Dict[int, Dict[str, Any]]) -> str:
        """
        Generate comparison table for metrics across k-values.
        
        Creates a formatted table showing how metrics change with different k-values.
        
        Args:
            results_by_k: Dictionary mapping k-values to their results
        
        Returns:
            Formatted comparison table as string
        
        Requirements:
            - 2.4: Generate a comparison table showing metrics across all k-values
        """
        # Sort k-values
        k_values = sorted(results_by_k.keys())
        
        # Build table header
        table = "\n" + "=" * 100 + "\n"
        table += "K-VALUE ABLATION COMPARISON TABLE\n"
        table += "=" * 100 + "\n"
        table += f"{'k-value':<10} {'Hit':<10} {'Hit@1':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Exact Match':<12}\n"
        table += "-" * 100 + "\n"
        
        # Add rows for each k-value
        for k in k_values:
            results = results_by_k[k]
            
            # Check if there was an error
            if "error" in results:
                table += f"{k:<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<12}\n"
            else:
                table += (
                    f"{k:<10} "
                    f"{results['hit']:<10.4f} "
                    f"{results['hit_at_1']:<10.4f} "
                    f"{results['macro_f1']:<10.4f} "
                    f"{results['macro_precision']:<12.4f} "
                    f"{results['macro_recall']:<10.4f} "
                    f"{results['exact_match']:<12.4f}\n"
                )
        
        table += "=" * 100 + "\n"
        
        return table
    
    def _save_ablation_results(
        self,
        results_by_k: Dict[int, Dict[str, Any]],
        comparison_table: str,
        dataset: str,
        retriever_type: str,
        output_dir: str
    ) -> str:
        """
        Save ablation results with individual and summary files.
        
        Saves:
        1. Summary JSON file with all k-value results and comparison table
        2. Individual result files are already saved by LLMComparisonService
        
        Args:
            results_by_k: Dictionary mapping k-values to their results
            comparison_table: Formatted comparison table string
            dataset: Dataset name
            retriever_type: Retriever type
            output_dir: Base output directory
        
        Returns:
            Path to saved summary file
        
        Requirements:
            - 2.5: Save individual results for each k-value and a summary comparison file
            - 8.1: Save results to the specified output directory
            - 8.2: Include timestamps in all output filenames
            - 8.5: Create output directory if it doesn't exist
            - 8.7: Validate all output files were written successfully
        """
        # Create ablation summary directory
        summary_dir = os.path.join(output_dir, dataset, f"{retriever_type}-ablation")
        
        try:
            os.makedirs(summary_dir, exist_ok=True)
        except Exception as e:
            raise IOError(
                f"Failed to create output directory: {summary_dir}\n"
                f"Error: {str(e)}"
            )
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(summary_dir, f"k_ablation_summary_{timestamp}.json")
        
        # Prepare summary data
        summary_data = {
            "metadata": {
                "timestamp": timestamp,
                "dataset": dataset,
                "retriever_type": retriever_type,
                "llm_model": "llama",
                "k_values": sorted(results_by_k.keys())
            },
            "results_by_k": {},
            "comparison_table": comparison_table
        }
        
        # Add metrics for each k-value (exclude file paths and detailed results)
        for k, results in results_by_k.items():
            if "error" in results:
                summary_data["results_by_k"][str(k)] = {
                    "error": results["error"]
                }
            else:
                summary_data["results_by_k"][str(k)] = {
                    "hit": results["hit"],
                    "hit_at_1": results["hit_at_1"],
                    "macro_f1": results["macro_f1"],
                    "macro_precision": results["macro_precision"],
                    "macro_recall": results["macro_recall"],
                    "exact_match": results["exact_match"],
                    "total_questions": results.get("total_questions", 0),
                    "predictions_file": results.get("predictions_file", ""),
                    "results_file": results.get("results_file", "")
                }
        
        # Save summary to JSON
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(
                f"Failed to write summary file: {summary_file}\n"
                f"Error: {str(e)}"
            )
        
        # Validate summary file was written successfully
        if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
            raise IOError(
                f"Summary file was not created successfully: {summary_file}"
            )
        
        # Print comparison table
        print(comparison_table)
        
        print(f"\nAblation summary saved to: {summary_file}")
        print(f"Individual results saved in: {os.path.join(output_dir, dataset)}/")
        
        return summary_file
