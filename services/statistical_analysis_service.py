"""
Statistical Analysis Service for comparing retrievers with case categorization.

This service performs comprehensive statistical comparison between cosine and KGscout
retrievers by categorizing each question into one of six predefined cases based on
answer coverage and path coverage metrics.
"""

import os
import json
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from utils.evaluation_utils import load_dataset, load_model_checkpoint
from utils.triplet_selector import select_triplets_kgscout, select_triplets_cosine
from utils.metrics import compute_answer_coverage, compute_path_coverage


class StatisticalAnalysisService:
    """
    Service for statistical comparison between retrievers.
    
    This service categorizes each question into one of six cases:
    - Case 1: Cosine no relevant, KGscout some relevant
    - Case 2: Cosine relevant no path, KGscout has path
    - Case 3: Both have relevant triplets (overlapping paths)
    - Case 4: Both have relevant triplets (non-overlapping paths)
    - Case 5: Cosine better than KGscout
    - Case 6: Both fail
    
    Requirements:
        - 4.1: Categorize each question into one of six predefined cases
        - 4.2: Identify Case 1 where Cosine has no relevant triplets but KGscout has some
        - 4.3: Identify Case 2 where Cosine has relevant triplets without paths but KGscout has complete path
        - 4.4: Identify Case 3 where both retrievers have relevant triplets with overlapping paths
        - 4.5: Identify Case 4 where both retrievers have relevant triplets with non-overlapping paths
        - 4.6: Identify Case 5 where Cosine performs better than KGscout
        - 4.7: Identify Case 6 where both retrievers fail to find relevant triplets
        - 4.8: Generate detailed statistics showing question count and percentage for each case
        - 4.9: Save example questions for each case category
        - 4.10: Save all results to JSON files with case breakdowns and example questions
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
    
    def _categorize_question(
        self,
        question_data: Dict,
        cosine_triplets: List[Tuple[str, str, str]],
        kgscout_triplets: List[Tuple[str, str, str]]
    ) -> str:
        """
        Categorize question into one of six cases based on coverage metrics.
        
        Case categorization logic:
        - Case 1: Cosine no relevant, KGscout some relevant
        - Case 2: Cosine relevant no path, KGscout has path
        - Case 3: Both have relevant triplets (overlapping paths)
        - Case 4: Both have relevant triplets (non-overlapping paths)
        - Case 5: Cosine better than KGscout
        - Case 6: Both fail
        
        Args:
            question_data: Dataset sample with question, entities, and answers
            cosine_triplets: Triplets selected by cosine retriever
            kgscout_triplets: Triplets selected by KGscout retriever
        
        Returns:
            Case identifier string ('case1' through 'case6')
        
        Requirements:
            - 4.2: Identify Case 1 where Cosine has no relevant triplets but KGscout has some
            - 4.3: Identify Case 2 where Cosine has relevant triplets without paths but KGscout has complete path
            - 4.4: Identify Case 3 where both retrievers have relevant triplets with overlapping paths
            - 4.5: Identify Case 4 where both retrievers have relevant triplets with non-overlapping paths
            - 4.6: Identify Case 5 where Cosine performs better than KGscout
            - 4.7: Identify Case 6 where both retrievers fail to find relevant triplets
        """
        # Compute coverage for both retrievers
        cosine_answer_cov = compute_answer_coverage(cosine_triplets, question_data['a_entity'])
        cosine_path_cov = compute_path_coverage(
            cosine_triplets,
            question_data['q_entity'],
            question_data['a_entity']
        )
        
        kgscout_answer_cov = compute_answer_coverage(kgscout_triplets, question_data['a_entity'])
        kgscout_path_cov = compute_path_coverage(
            kgscout_triplets,
            question_data['q_entity'],
            question_data['a_entity']
        )
        
        # Case 6: Both fail (no answer coverage for either)
        if not cosine_answer_cov and not kgscout_answer_cov:
            return 'case6'
        
        # Case 1: Cosine no relevant, KGscout some relevant
        if not cosine_answer_cov and kgscout_answer_cov:
            return 'case1'
        
        # Case 5: Cosine better (has path, KGscout doesn't have path)
        if cosine_path_cov and not kgscout_path_cov:
            return 'case5'
        
        # Case 2: Cosine relevant no path, KGscout has path
        if cosine_answer_cov and not cosine_path_cov and kgscout_path_cov:
            return 'case2'
        
        # Case 3 & 4: Both have paths (check for overlap)
        if cosine_path_cov and kgscout_path_cov:
            # Check if paths overlap by comparing edges
            cosine_edges = set()
            for s, r, o in cosine_triplets:
                cosine_edges.add((s.lower(), o.lower()))
            
            kgscout_edges = set()
            for s, r, o in kgscout_triplets:
                kgscout_edges.add((s.lower(), o.lower()))
            
            if cosine_edges & kgscout_edges:  # Overlapping paths
                return 'case3'
            else:  # Non-overlapping paths
                return 'case4'
        
        # Default: Both have relevant triplets but unclear categorization
        # This handles cases where both have answer coverage but neither has path coverage
        return 'case3'
    
    def _compute_case_statistics(
        self,
        case_results: Dict[str, List[Dict]],
        total_questions: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for each case category.
        
        Args:
            case_results: Dictionary mapping case names to lists of question data
            total_questions: Total number of questions in dataset
        
        Returns:
            Dictionary with count and percentage for each case
        
        Requirements:
            - 4.8: Generate detailed statistics showing question count and percentage for each case
        """
        statistics = {}
        
        case_descriptions = {
            'case1': 'Cosine no relevant, KGscout some relevant',
            'case2': 'Cosine relevant no path, KGscout has path',
            'case3': 'Both have relevant triplets (overlapping paths)',
            'case4': 'Both have relevant triplets (non-overlapping paths)',
            'case5': 'Cosine better than KGscout',
            'case6': 'Both fail'
        }
        
        for case_name, questions in case_results.items():
            count = len(questions)
            percentage = (count / total_questions * 100) if total_questions > 0 else 0.0
            
            statistics[case_name] = {
                'count': count,
                'percentage': percentage,
                'description': case_descriptions.get(case_name, 'Unknown case')
            }
        
        return statistics
    
    def _save_statistical_results(
        self,
        case_results: Dict[str, List[Dict]],
        statistics: Dict[str, Dict[str, Any]],
        output_dir: str,
        dataset: str,
        k: int
    ) -> str:
        """
        Save statistical results following Analysis-Copy1.ipynb format.
        
        Args:
            case_results: Dictionary mapping case names to lists of question data
            statistics: Dictionary with count and percentage for each case
            output_dir: Directory to save results
            dataset: Dataset name
            k: Number of top triplets used
        
        Returns:
            Path to saved results file
        
        Requirements:
            - 4.9: Save example questions for each case category
            - 4.10: Save all results to JSON files with case breakdowns and example questions
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
        filename = f"statistical_analysis_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare examples for each case (up to 5 examples per case)
        examples_per_case = {}
        for case_name, questions in case_results.items():
            examples = []
            for q_data in questions[:5]:  # Limit to 5 examples
                example = {
                    'question_id': q_data['question_id'],
                    'question': q_data['question'],
                    'cosine_triplet_count': len(q_data['cosine_triplets']),
                    'kgscout_triplet_count': len(q_data['kgscout_triplets'])
                }
                examples.append(example)
            examples_per_case[case_name] = examples
        
        # Prepare output data
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "dataset": dataset,
                "k": k,
                "total_questions": sum(stats['count'] for stats in statistics.values())
            },
            "case_statistics": statistics,
            "examples_per_case": examples_per_case
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
    
    def run_statistical_analysis(
        self,
        dataset: str,
        model_path: str,
        k: int,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison analysis with case categorization.
        
        This method orchestrates the full statistical analysis pipeline:
        1. Load dataset and model
        2. Select triplets with both retrievers
        3. Categorize each question into one of six cases
        4. Compute statistics for each case
        5. Save results with examples
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
            model_path: Path to KGscout model
            k: Number of top triplets
            output_dir: Directory to save results (default: 'results/statistical')
        
        Returns:
            Dictionary with case results, statistics, and output file path
        
        Requirements:
            - 4.1: Categorize each question into one of six predefined cases
            - 4.8: Generate detailed statistics showing question count and percentage for each case
            - 4.9: Save example questions for each case category
            - 4.10: Save all results to JSON files with case breakdowns and example questions
        """
        # Set default output directory
        if output_dir is None:
            output_dir = "results/statistical"
        
        # Load dataset
        print(f"\nLoading dataset: {dataset}")
        data = self._load_dataset(dataset)
        print(f"Loaded {len(data)} questions")
        
        # Load model (this will raise FileNotFoundError if model doesn't exist)
        print(f"\nLoading model from: {model_path}")
        model = load_model_checkpoint(model_path, self.device)
        
        # Select triplets with both retrievers
        print(f"\n{'='*60}")
        print(f"Selecting triplets with k={k}")
        print(f"{'='*60}")
        
        kgscout_triplets = self._select_with_kgscout(data, model, k)
        cosine_triplets = self._select_with_cosine(data, k)
        
        # Categorize each question
        print(f"\nCategorizing questions into cases...")
        case_results = {
            'case1': [],  # Cosine no relevant, KGscout some relevant
            'case2': [],  # Cosine relevant no path, KGscout has path
            'case3': [],  # Both have relevant triplets (overlapping paths)
            'case4': [],  # Both have relevant triplets (non-overlapping paths)
            'case5': [],  # Cosine better than KGscout
            'case6': [],  # Both fail
        }
        
        for i, question_data in enumerate(tqdm(data, desc="Categorizing")):
            case = self._categorize_question(
                question_data,
                cosine_triplets[i],
                kgscout_triplets[i]
            )
            case_results[case].append({
                'question_id': i,
                'question': question_data['question'],
                'cosine_triplets': cosine_triplets[i],
                'kgscout_triplets': kgscout_triplets[i]
            })
        
        # Compute statistics
        statistics = self._compute_case_statistics(case_results, len(data))
        
        # Display statistics
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"\nTotal Questions: {len(data)}")
        print(f"\nCase Breakdown:")
        print(f"{'Case':<10} {'Description':<50} {'Count':<10} {'Percentage':<10}")
        print("-" * 80)
        
        for case_name in ['case1', 'case2', 'case3', 'case4', 'case5', 'case6']:
            stats = statistics[case_name]
            print(f"{case_name:<10} {stats['description']:<50} {stats['count']:<10} {stats['percentage']:.2f}%")
        
        print(f"{'='*60}\n")
        
        # Save results
        output_file = self._save_statistical_results(
            case_results,
            statistics,
            output_dir,
            dataset,
            k
        )
        
        print(f"Results saved to: {output_file}")
        
        return {
            'case_results': case_results,
            'statistics': statistics,
            'output_file': output_file
        }
