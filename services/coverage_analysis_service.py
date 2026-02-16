"""
Coverage Analysis Service for evaluating answer and path coverage metrics.

This service analyzes the quality of triplet selection by measuring:
- Answer coverage: Whether answer entities exist in selected triplets
- Path coverage: Whether complete reasoning paths exist in selected triplets

The service compares KGscout retriever against cosine retriever across different k values.
"""

import os
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from utils.evaluation_utils import load_dataset, load_model_checkpoint
from utils.triplet_selector import select_triplets_kgscout, select_triplets_cosine
from utils.metrics import compute_answer_coverage, compute_path_coverage


class CoverageAnalysisService:
    """
    Service for path and answer coverage analysis.
    
    This service evaluates retriever quality independent of LLM performance by
    measuring whether selected triplets contain answer entities and complete
    reasoning paths.
    
    Requirements:
        - 3.1: Compute Answer_Coverage and Path_Coverage for each specified k-value
        - 3.2: Compare KGscout retriever against Cosine_Retriever for each k-value
        - 3.3: Check if answer entities exist in the selected triplets
        - 3.4: Check if a complete Reasoning_Path exists in the selected triplets
        - 3.5: Generate a comparison table showing coverage metrics for both retrievers
        - 3.6: Save detailed per-question results and summary statistics to JSON files
        - 3.7: Display error message and exit when model-path does not exist
    """
    
    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.
        
        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_dataset(self, dataset: str) -> List[Dict]:
        """
        Load dataset using evaluation_utils.
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
        
        Returns:
            List of dataset samples
        """
        return load_dataset(dataset)
    
    def _select_with_kgscout(
        self,
        data: List[Dict],
        model,
        k: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Select triplets using KGscout model for all questions.
        
        Args:
            data: List of dataset samples
            model: Trained PathRankingModel
            k: Number of top triplets to select
        
        Returns:
            List of triplet lists (one per question)
        """
        triplets_per_question = []
        failed_count = 0
        
        for sample in tqdm(data, desc=f"Selecting with KGscout (k={k})", unit="question"):
            try:
                triplets = select_triplets_kgscout(model, sample, k, self.device)
                triplets_per_question.append(triplets)
            except Exception as e:
                failed_count += 1
                print(f"\nWarning: Failed to select triplets for question. Error: {str(e)}")
                triplets_per_question.append([])
        
        if failed_count > 0:
            print(f"\nWarning: {failed_count} out of {len(data)} questions failed during KGscout selection")
        
        return triplets_per_question
    
    def _select_with_cosine(
        self,
        data: List[Dict],
        k: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Select triplets using cosine similarity for all questions.
        
        Args:
            data: List of dataset samples
            k: Number of top triplets to select
        
        Returns:
            List of triplet lists (one per question)
        """
        triplets_per_question = []
        failed_count = 0
        
        for sample in tqdm(data, desc=f"Selecting with Cosine (k={k})", unit="question"):
            try:
                triplets = select_triplets_cosine(sample, k)
                triplets_per_question.append(triplets)
            except Exception as e:
                failed_count += 1
                print(f"\nWarning: Failed to select triplets for question. Error: {str(e)}")
                triplets_per_question.append([])
        
        if failed_count > 0:
            print(f"\nWarning: {failed_count} out of {len(data)} questions failed during Cosine selection")
        
        return triplets_per_question
    
    def _compute_coverage_metrics(
        self,
        data: List[Dict],
        triplets_per_question: List[List[Tuple[str, str, str]]]
    ) -> Dict[str, float]:
        """
        Compute answer and path coverage metrics.
        
        Args:
            data: List of dataset samples
            triplets_per_question: List of triplet lists (one per question)
        
        Returns:
            Dictionary with answer_coverage and path_coverage percentages
        """
        answer_coverage_count = 0
        path_coverage_count = 0
        
        for question_data, triplets in zip(data, triplets_per_question):
            # Check answer coverage
            if compute_answer_coverage(triplets, question_data['a_entity']):
                answer_coverage_count += 1
            
            # Check path coverage
            if compute_path_coverage(triplets, question_data['q_entity'], question_data['a_entity']):
                path_coverage_count += 1
        
        total_questions = len(data)
        return {
            'answer_coverage': answer_coverage_count / total_questions if total_questions > 0 else 0.0,
            'path_coverage': path_coverage_count / total_questions if total_questions > 0 else 0.0,
            'answer_coverage_count': answer_coverage_count,
            'path_coverage_count': path_coverage_count,
            'total_questions': total_questions
        }
    
    def _generate_coverage_comparison(self, results: Dict[str, Dict[int, Dict]]) -> str:
        """
        Generate comparison table for coverage metrics.
        
        Args:
            results: Dictionary with structure {retriever: {k: metrics}}
        
        Returns:
            Formatted comparison table string
        """
        # Get all k values (sorted)
        k_values = sorted(list(results['kgscout'].keys()))
        
        # Build table header
        table = "\n" + "=" * 80 + "\n"
        table += "COVERAGE ANALYSIS COMPARISON\n"
        table += "=" * 80 + "\n\n"
        
        # Build table rows
        table += f"{'K Value':<10} {'Retriever':<12} {'Answer Cov':<15} {'Path Cov':<15}\n"
        table += "-" * 80 + "\n"
        
        for k in k_values:
            # KGscout row
            kgscout_metrics = results['kgscout'][k]
            table += f"{k:<10} {'KGscout':<12} "
            table += f"{kgscout_metrics['answer_coverage']:.2%}".ljust(15)
            table += f"{kgscout_metrics['path_coverage']:.2%}".ljust(15)
            table += "\n"
            
            # Cosine row
            cosine_metrics = results['cosine'][k]
            table += f"{'':<10} {'Cosine':<12} "
            table += f"{cosine_metrics['answer_coverage']:.2%}".ljust(15)
            table += f"{cosine_metrics['path_coverage']:.2%}".ljust(15)
            table += "\n"
            
            # Difference row
            answer_diff = kgscout_metrics['answer_coverage'] - cosine_metrics['answer_coverage']
            path_diff = kgscout_metrics['path_coverage'] - cosine_metrics['path_coverage']
            table += f"{'':<10} {'Difference':<12} "
            table += f"{answer_diff:+.2%}".ljust(15)
            table += f"{path_diff:+.2%}".ljust(15)
            table += "\n"
            table += "-" * 80 + "\n"
        
        table += "=" * 80 + "\n"
        
        return table
    
    def _save_coverage_results(
        self,
        results: Dict[str, Dict[int, Dict]],
        comparison_table: str,
        output_dir: str,
        dataset: str,
        k_values: List[int]
    ) -> str:
        """
        Save coverage results with summary JSON only (no per-question results).
        
        Args:
            results: Dictionary with structure {retriever: {k: metrics}}
            comparison_table: Formatted comparison table string
            output_dir: Directory to save results
            dataset: Dataset name
            k_values: List of k values tested
        
        Returns:
            Path to saved results file
        
        Requirements:
            - 8.1: Save results to the specified output directory
            - 8.2: Include timestamps in all output filenames
            - 8.5: Create output directory if it doesn't exist
            - 8.7: Validate all output files were written successfully
        """
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise IOError(
                f"Failed to create output directory: {output_dir}\n"
                f"Error: {str(e)}"
            )
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coverage_analysis_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare results dictionary
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "dataset": dataset,
                "k_values": k_values
            },
            "results": results,
            "comparison_table": comparison_table
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(
                f"Failed to write results file: {filepath}\n"
                f"Error: {str(e)}"
            )
        
        # Validate file was written successfully
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            raise IOError(
                f"Results file was not created successfully: {filepath}"
            )
        
        return filepath
    
    def run_coverage_analysis(
        self,
        dataset: str,
        model_path: str,
        k_values: List[int],
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Analyze answer and path coverage for different k values.
        
        This method orchestrates the full coverage analysis pipeline:
        1. Load dataset and model
        2. For each k value:
           - Select triplets with KGscout
           - Select triplets with Cosine
           - Compute coverage metrics for both
        3. Generate comparison table
        4. Save results
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
            model_path: Path to KGscout model
            k_values: List of k values to test
            output_dir: Directory to save results (default: 'results/coverage')
        
        Returns:
            Dictionary with coverage metrics for both retrievers and output file path
        
        Requirements:
            - 3.1: Compute Answer_Coverage and Path_Coverage for each specified k-value
            - 3.2: Compare KGscout retriever against Cosine_Retriever for each k-value
            - 3.5: Generate a comparison table showing coverage metrics
            - 3.6: Save summary statistics to JSON files
            - 3.7: Display error message when model-path does not exist
        """
        # Set default output directory
        if output_dir is None:
            output_dir = "results/coverage"
        
        # Load dataset
        print(f"\nLoading dataset: {dataset}")
        data = self._load_dataset(dataset)
        print(f"Loaded {len(data)} questions")
        
        # Load model (this will raise FileNotFoundError if model doesn't exist)
        print(f"\nLoading model from: {model_path}")
        model = load_model_checkpoint(model_path, self.device)
        
        # Initialize results structure
        results = {
            'kgscout': {},
            'cosine': {}
        }
        
        # Analyze coverage for each k value
        for idx, k in enumerate(k_values, 1):
            print(f"\n{'='*60}")
            print(f"Analyzing coverage for k={k} ({idx}/{len(k_values)})")
            print(f"{'='*60}")
            
            # KGscout coverage
            print(f"\nKGscout retriever:")
            kgscout_triplets = self._select_with_kgscout(data, model, k)
            results['kgscout'][k] = self._compute_coverage_metrics(data, kgscout_triplets)
            print(f"  Answer Coverage: {results['kgscout'][k]['answer_coverage']:.2%} ({results['kgscout'][k]['answer_coverage_count']}/{results['kgscout'][k]['total_questions']})")
            print(f"  Path Coverage:   {results['kgscout'][k]['path_coverage']:.2%} ({results['kgscout'][k]['path_coverage_count']}/{results['kgscout'][k]['total_questions']})")
            
            # Cosine coverage
            print(f"\nCosine retriever:")
            cosine_triplets = self._select_with_cosine(data, k)
            results['cosine'][k] = self._compute_coverage_metrics(data, cosine_triplets)
            print(f"  Answer Coverage: {results['cosine'][k]['answer_coverage']:.2%} ({results['cosine'][k]['answer_coverage_count']}/{results['cosine'][k]['total_questions']})")
            print(f"  Path Coverage:   {results['cosine'][k]['path_coverage']:.2%} ({results['cosine'][k]['path_coverage_count']}/{results['cosine'][k]['total_questions']})")
        
        # Generate comparison table
        comparison_table = self._generate_coverage_comparison(results)
        print(comparison_table)
        
        # Save results
        output_file = self._save_coverage_results(
            results,
            comparison_table,
            output_dir,
            dataset,
            k_values
        )
        
        print(f"\nResults saved to: {output_file}")
        
        return {
            'results': results,
            'comparison_table': comparison_table,
            'output_file': output_file
        }
