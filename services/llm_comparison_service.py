"""
LLM Comparison Service for evaluating different LLM models with various retrievers.

This service orchestrates the full evaluation pipeline:
1. Load dataset
2. Select triplets using KGscout or cosine retriever
3. Run LLM inference to generate answers
4. Compute evaluation metrics
5. Save results in lama-inference.py format
"""

import os
import json
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from utils.evaluation_utils import load_dataset, load_model_checkpoint
from utils.triplet_selector import select_triplets_kgscout, select_triplets_cosine
from utils.llm_inference import load_llm_model, format_prompt, run_llm_inference
from utils.metrics import (
    extract_predictions_from_response,
    compute_hit_score,
    compute_hit_at_1,
    compute_precision,
    compute_recall,
    compute_f1_score,
    should_use_double_check,
    preprocess_date_answers
)


class LLMComparisonService:
    """
    Service for comparing LLM models with different retrievers.
    
    This service implements the full evaluation pipeline for LLM comparison analysis,
    supporting both KGscout and cosine retriever methods.
    
    Requirements:
        - 1.1: Load dataset and run inference with specified LLM and retriever
        - 1.2: Generate Selected_JSON using trained model when retriever-type is "kgscout"
        - 1.3: Use topk_linearized_triplets field when retriever-type is "cosine"
        - 1.4: Compute Hit, Hit@1, Macro F1, Precision, Recall, and Exact Match metrics
        - 1.5: Save results to JSON file in output directory with timestamp
        - 9.1-9.6: LLM inference integration
        - 10.1-10.5: Triplet selection methods
    """
    
    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.
        
        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LLMComparisonService initialized on device: {self.device}")
    
    def _load_dataset(self, dataset: str) -> List[Dict]:
        """
        Load dataset using evaluation_utils.
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
        
        Returns:
            List of dataset samples
        """
        print(f"\nLoading dataset: {dataset}")
        data = load_dataset(dataset)
        print(f"Dataset loaded: {len(data)} samples")
        return data
    
    def _select_with_kgscout(
        self,
        data: List[Dict],
        model,
        k: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Select triplets using KGscout model.
        
        Args:
            data: List of dataset samples
            model: Trained PathRankingModel
            k: Number of top triplets to select
        
        Returns:
            List of selected triplets for each question
        """
        print(f"\nSelecting top-{k} triplets using KGscout model...")
        triplets_per_question = []
        failed_count = 0
        
        for sample in tqdm(data, desc="Selecting triplets", unit="question"):
            try:
                selected = select_triplets_kgscout(model, sample, k, self.device)
                triplets_per_question.append(selected)
            except Exception as e:
                failed_count += 1
                print(f"\nWarning: Failed to select triplets for question. Error: {str(e)}")
                triplets_per_question.append([])
        
        if failed_count > 0:
            print(f"\nWarning: {failed_count} out of {len(data)} questions failed during triplet selection")
        
        print(f"Triplet selection complete: {len(data) - failed_count}/{len(data)} successful")
        return triplets_per_question
    
    def _select_with_cosine(
        self,
        data: List[Dict],
        k: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Select triplets using cosine similarity.
        
        Args:
            data: List of dataset samples
            k: Number of top triplets to select
        
        Returns:
            List of selected triplets for each question
        """
        print(f"\nSelecting top-{k} triplets using cosine similarity...")
        triplets_per_question = []
        failed_count = 0
        
        for sample in tqdm(data, desc="Selecting triplets", unit="question"):
            try:
                selected = select_triplets_cosine(sample, k)
                triplets_per_question.append(selected)
            except Exception as e:
                failed_count += 1
                print(f"\nWarning: Failed to select triplets for question. Error: {str(e)}")
                triplets_per_question.append([])
        
        if failed_count > 0:
            print(f"\nWarning: {failed_count} out of {len(data)} questions failed during triplet selection")
        
        print(f"Triplet selection complete: {len(data) - failed_count}/{len(data)} successful")
        return triplets_per_question
    
    def _run_llm_inference(
        self,
        data: List[Dict],
        triplets_per_question: List[List[Tuple[str, str, str]]],
        llm_model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Run LLM inference using llm_inference utils.
        
        Args:
            data: List of dataset samples
            triplets_per_question: Selected triplets for each question
            llm_model_name: LLM model name ('llama', 'qwen', 'deepseek')
        
        Returns:
            List of prediction dictionaries with question, predicted, ground_truth
        """
        print(f"\nLoading LLM model: {llm_model_name}")
        try:
            llm_model, tokenizer = load_llm_model(llm_model_name, self.device)
            print(f"LLM model loaded successfully")
        except Exception as e:
            print(f"\nError: Failed to load LLM model '{llm_model_name}'")
            print(f"Reason: {str(e)}")
            print(f"Suggestion: Verify the model name is correct and the model is available")
            raise
        
        print(f"\nRunning LLM inference on {len(data)} questions...")
        predictions = []
        failed_count = 0
        
        for idx, (sample, triplets) in enumerate(tqdm(
            zip(data, triplets_per_question),
            total=len(data),
            desc="LLM inference",
            unit="question"
        )):
            try:
                # Format triplets as linearized strings
                linearized_triplets = [f"{s}, {r}, {o}" for s, r, o in triplets]
                
                # Format prompt
                prompt = format_prompt(sample["question"], linearized_triplets)
                
                # Run inference
                response = run_llm_inference(llm_model, tokenizer, prompt)
                
                # Extract predictions from response
                predicted_answers = extract_predictions_from_response(response)
                
                # Store prediction
                predictions.append({
                    "question": sample["question"],
                    "predicted": predicted_answers,
                    "ground_truth": sample["answer"],
                    "response": response
                })
                
            except Exception as e:
                failed_count += 1
                print(f"\nWarning: LLM inference failed for question {idx}. Error: {str(e)}")
                predictions.append({
                    "question": sample.get("question", ""),
                    "predicted": [],
                    "ground_truth": sample.get("answer", []),
                    "response": ""
                })
        
        if failed_count > 0:
            print(f"\nWarning: {failed_count} out of {len(data)} questions failed during LLM inference")
        
        return predictions
    
    def _compute_metrics(
        self,
        data: List[Dict],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute metrics using metrics utils.
        
        Args:
            data: List of dataset samples
            predictions: List of prediction dictionaries
        
        Returns:
            Dictionary with aggregated metrics
        """
        print("\nComputing evaluation metrics...")
        
        total_hit = 0.0
        total_hit_at_1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_exact_match = 0.0
        
        detailed_results = []
        
        for idx, (sample, pred) in enumerate(zip(data, predictions)):
            question = sample["question"]
            predicted = pred["predicted"]
            ground_truth = sample["answer"]
            
            # Preprocess date answers if needed
            ground_truth = preprocess_date_answers(question, ground_truth)
            
            # Determine if double-check should be used
            double_check = should_use_double_check(question)
            
            # Compute metrics
            hit = compute_hit_score(predicted, ground_truth, double_check)
            hit_at_1 = compute_hit_at_1(predicted, ground_truth, double_check)
            precision, matched_p, total_p = compute_precision(predicted, ground_truth, double_check)
            recall, matched_r, total_r = compute_recall(predicted, ground_truth, double_check)
            f1 = compute_f1_score(precision, recall)
            
            # Exact match: all predictions match all ground truths
            exact_match = 1.0 if (precision == 1.0 and recall == 1.0) else 0.0
            
            # Accumulate metrics
            total_hit += hit
            total_hit_at_1 += hit_at_1
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_exact_match += exact_match
            
            # Store detailed result
            detailed_results.append({
                "question": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "hit": hit,
                "hit_at_1": hit_at_1,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_match": exact_match
            })
        
        # Compute averages
        n = len(data)
        metrics = {
            "hit": total_hit / n,
            "hit_at_1": total_hit_at_1 / n,
            "macro_precision": total_precision / n,
            "macro_recall": total_recall / n,
            "macro_f1": total_f1 / n,
            "exact_match": total_exact_match / n,
            "total_questions": n,
            "detailed_results": detailed_results
        }
        
        print(f"Metrics computed:")
        print(f"  Hit: {metrics['hit']:.4f}")
        print(f"  Hit@1: {metrics['hit_at_1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        
        return metrics

    
    def _save_results(
        self,
        metrics: Dict[str, Any],
        dataset: str,
        retriever_type: str,
        k: int,
        output_dir: str
    ) -> Tuple[str, str]:
        """
        Save results following lama-inference.py format (text + jsonl files).
        
        Saves results in directory structure: {output_dir}/{dataset}/{retriever}-k{k}/
        - predictions.txt: One prediction per line
        - results.jsonl: One JSON object per line with detailed results
        
        Args:
            metrics: Dictionary with metrics and detailed_results
            dataset: Dataset name ('webqsp' or 'cwq')
            retriever_type: Retriever type ('kgscout' or 'cosine')
            k: Number of top triplets
            output_dir: Base output directory
        
        Returns:
            Tuple of (predictions_file_path, results_file_path)
        
        Requirements:
            - 8.1: Save results to the specified output directory
            - 8.3: Implement directory structure for commands 1&2: results/{dataset}/{retriever}-k{k}/
            - 8.5: Create output directory if it doesn't exist
            - 8.7: Validate all output files were written successfully
        """
        # Create directory structure: {output_dir}/{dataset}/{retriever}-k{k}/
        result_dir = os.path.join(output_dir, dataset, f"{retriever_type}-k{k}")
        
        try:
            os.makedirs(result_dir, exist_ok=True)
        except Exception as e:
            raise IOError(
                f"Failed to create output directory: {result_dir}\n"
                f"Error: {str(e)}"
            )
        
        print(f"\nSaving results to: {result_dir}")
        
        # Save predictions.txt (one prediction per line)
        predictions_file = os.path.join(result_dir, "predictions.txt")
        try:
            with open(predictions_file, 'w', encoding='utf-8') as f:
                for result in metrics["detailed_results"]:
                    # Write first prediction (or empty line if no predictions)
                    if result["predicted"]:
                        f.write(result["predicted"][0] + "\n")
                    else:
                        f.write("\n")
        except Exception as e:
            raise IOError(
                f"Failed to write predictions file: {predictions_file}\n"
                f"Error: {str(e)}"
            )
        
        # Validate predictions file was written successfully
        if not os.path.exists(predictions_file) or os.path.getsize(predictions_file) == 0:
            raise IOError(
                f"Predictions file was not created successfully: {predictions_file}"
            )
        
        # Save results.jsonl (one JSON object per line)
        results_file = os.path.join(result_dir, "results.jsonl")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                for result in metrics["detailed_results"]:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        except Exception as e:
            raise IOError(
                f"Failed to write results file: {results_file}\n"
                f"Error: {str(e)}"
            )
        
        # Validate results file was written successfully
        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
            raise IOError(
                f"Results file was not created successfully: {results_file}"
            )
        
        # Save summary metrics
        summary_file = os.path.join(result_dir, "summary.json")
        try:
            summary = {
                "dataset": dataset,
                "retriever_type": retriever_type,
                "k": k,
                "metrics": {
                    "hit": metrics["hit"],
                    "hit_at_1": metrics["hit_at_1"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "exact_match": metrics["exact_match"]
                },
                "total_questions": metrics["total_questions"]
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
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
        
        print(f"Results saved:")
        print(f"  - Predictions: {predictions_file}")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Summary: {summary_file}")
        
        return predictions_file, results_file
    
    def run_comparison(
        self,
        dataset: str,
        llm_model: str,
        retriever_type: str,
        k: int,
        model_path: str = None,
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Run LLM comparison analysis orchestrating the full pipeline.
        
        Pipeline steps:
        1. Load dataset
        2. Load model (if using KGscout)
        3. Select triplets based on retriever type
        4. Run LLM inference
        5. Compute metrics
        6. Save results
        
        Args:
            dataset: Dataset name ('webqsp' or 'cwq')
            llm_model: LLM model name ('llama', 'qwen', 'deepseek')
            retriever_type: Retriever type ('kgscout' or 'cosine')
            k: Number of top triplets
            model_path: Path to KGscout model (required if retriever_type='kgscout')
            output_dir: Directory to save results (default: 'results')
        
        Returns:
            Dictionary with metrics and output file paths
        
        Requirements:
            - 1.1: Load dataset and run inference with specified LLM and retriever
            - 1.2: Generate Selected_JSON using trained model when retriever-type is "kgscout"
            - 1.3: Use topk_linearized_triplets field when retriever-type is "cosine"
            - 1.4: Compute Hit, Hit@1, Macro F1, Precision, Recall, and Exact Match metrics
            - 1.5: Save results to JSON file in output directory with timestamp
        """
        print("=" * 60)
        print("LLM COMPARISON ANALYSIS")
        print("=" * 60)
        print(f"Dataset: {dataset}")
        print(f"LLM Model: {llm_model}")
        print(f"Retriever: {retriever_type}")
        print(f"Top-k: {k}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        
        # Step 1: Load dataset
        data = self._load_dataset(dataset)
        
        # Step 2: Select triplets based on retriever type
        if retriever_type == "kgscout":
            if model_path is None:
                raise ValueError(
                    "model_path is required when retriever_type='kgscout'. "
                    "Please provide the path to the trained model checkpoint."
                )
            model = load_model_checkpoint(model_path, self.device)
            triplets_per_question = self._select_with_kgscout(data, model, k)
        elif retriever_type == "cosine":
            triplets_per_question = self._select_with_cosine(data, k)
        else:
            raise ValueError(
                f"Invalid retriever_type: '{retriever_type}'. "
                f"Expected 'kgscout' or 'cosine'."
            )
        
        # Step 3: Run LLM inference
        predictions = self._run_llm_inference(data, triplets_per_question, llm_model)
        
        # Step 4: Compute metrics
        metrics = self._compute_metrics(data, predictions)
        
        # Step 5: Save results
        predictions_file, results_file = self._save_results(
            metrics,
            dataset,
            retriever_type,
            k,
            output_dir
        )
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            "hit": metrics["hit"],
            "hit_at_1": metrics["hit_at_1"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "exact_match": metrics["exact_match"],
            "total_questions": metrics["total_questions"],
            "predictions_file": predictions_file,
            "results_file": results_file
        }
