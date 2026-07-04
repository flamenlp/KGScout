"""
Coverage Analysis Service for evaluating answer and path coverage metrics.

This service analyzes the quality of triplet selection by measuring:
- Answer coverage: Whether answer entities exist in selected triplets
- Path coverage: Whether complete reasoning paths exist in selected triplets

The service compares KGscout retriever against cosine retriever across different k values.
Uses DataLoader(batch_size=1) pattern consistent with the rest of the project.
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.utils.triplet_selector import (
    select_triplets_kgscout, select_triplets_cosine, _extract_metadata_from_batch
)
from src.utils.metrics import compute_answer_coverage, compute_path_coverage


class CoverageAnalysisService:
    """
    Service for path and answer coverage analysis.

    Evaluates retriever quality independent of LLM performance by measuring whether
    selected triplets contain answer entities and complete reasoning paths.

    Uses DataLoader(batch_size=1, default collate) for data iteration.

    Results saved to: results/coverage-analysis/{dataset}/{kval}/
    """

    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.

        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _create_dataloader(self, dataset_path: str, sample_k: int = 1000) -> DataLoader:
        """
        Load dataset and create DataLoader(batch_size=1).

        Args:
            dataset_path: Path to .pt dataset file
            sample_k: Number of triplets to sample per question

        Returns:
            DataLoader iterating one sample at a time
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}\n"
                f"Please ensure the dataset has been preprocessed."
            )

        data = torch.load(dataset_path, weights_only=False, map_location="cpu")
        print(f"  Loaded {len(data)} samples from {dataset_path}")

        dataset = SampledJointTrainingDataset(data, k=sample_k)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load model from checkpoint directory (save_pretrained format).

        Args:
            model_path: Path to model checkpoint directory containing path_ranker.pt

        Returns:
            Loaded model
        """
        from src.model.path_ranker import PathRankingModel

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path not found: {model_path}\n"
                f"Please ensure the model was trained and saved correctly."
            )

        model = PathRankingModel.from_pretrained(model_path, device=self.device)
        model.to(self.device)
        model.eval()
        print(f"  Model loaded from {model_path}")
        return model

    def _evaluate_coverage(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        k: int,
        retriever_type: str = "kgscout"
    ) -> Dict[str, Any]:
        """
        Evaluate answer and path coverage on the given DataLoader.

        Args:
            dataloader: DataLoader(batch_size=1) over test dataset
            model: Trained model (used if retriever_type='kgscout')
            k: Number of top triplets to select
            retriever_type: 'kgscout' or 'cosine'

        Returns:
            Dict with coverage metrics and per-sample results
        """
        ans_present_count = 0
        path_exists_count = 0
        total_count = 0
        per_sample_results = []

        for i, batch in enumerate(tqdm(dataloader, desc=f"  Coverage ({retriever_type}, k={k})")):
            try:
                # Select triplets
                if retriever_type == "kgscout":
                    selected_triplets = select_triplets_kgscout(model, batch, k, self.device)
                else:
                    selected_triplets = select_triplets_cosine(batch, k)

                if len(selected_triplets) == 0:
                    continue

                # Extract metadata
                meta = _extract_metadata_from_batch(batch)
                q_ents = [e.lower() for e in meta["q_entity"]] if meta["q_entity"] else []
                a_ents = [e.lower() for e in meta["a_entity"]] if meta["a_entity"] else []

                if not a_ents:
                    continue

                # Compute coverage metrics
                ans_present = compute_answer_coverage(selected_triplets, a_ents)
                path_exists = compute_path_coverage(selected_triplets, q_ents, a_ents)

                if ans_present:
                    ans_present_count += 1
                if path_exists:
                    path_exists_count += 1

                total_count += 1

                per_sample_results.append({
                    "id": i,
                    "question": meta["question"],
                    "answer_entity_present": ans_present,
                    "reasoning_path_exists": path_exists,
                    "num_selected_triplets": len(selected_triplets),
                })

            except Exception as e:
                continue

        if total_count == 0:
            return {"total_samples": 0, "per_sample_results": []}

        metrics = {
            "total_samples": total_count,
            "answer_coverage": ans_present_count / total_count,
            "path_coverage": path_exists_count / total_count,
            "answer_coverage_count": ans_present_count,
            "path_coverage_count": path_exists_count,
            "per_sample_results": per_sample_results,
        }
        return metrics

    def run_coverage_analysis(
        self,
        dataset_path: str,
        model_path: str,
        k_values: List[int],
        dataset_name: str = "cwq",
        output_dir: str = None,
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Analyze answer and path coverage for different k values.

        Pipeline:
        1. Load test DataLoader
        2. Load model
        3. For each k value, compute coverage for both KGscout and Cosine
        4. Generate comparison table
        5. Save results to results/coverage-analysis/{dataset}/{kval}/

        Args:
            dataset_path: Path to test .pt dataset file
            model_path: Path to model checkpoint directory
            k_values: List of k values to test
            dataset_name: Dataset name for output directory (default: 'cwq')
            output_dir: Base output directory (default: 'results/coverage-analysis')
            sample_k: Number of triplets to feed to model (default: 1000)

        Returns:
            Dictionary with coverage metrics for both retrievers
        """
        if output_dir is None:
            output_dir = "results/coverage-analysis"

        print(f"\n{'='*60}")
        print("COVERAGE ANALYSIS")
        print(f"{'='*60}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Model:   {model_path}")
        print(f"  K values: {k_values}")

        # Load DataLoader and model
        print("\nLoading data...")
        dataloader = self._create_dataloader(dataset_path, sample_k=sample_k)

        print("Loading model...")
        model = self._load_model(model_path)

        results = {"kgscout": {}, "cosine": {}}

        for k in k_values:
            print(f"\n{'='*60}")
            print(f"  k = {k}")
            print(f"{'='*60}")

            # KGscout coverage
            kgscout_metrics = self._evaluate_coverage(dataloader, model, k, "kgscout")
            results["kgscout"][k] = kgscout_metrics
            print(f"  KGscout — Answer: {kgscout_metrics['answer_coverage']:.2%}, Path: {kgscout_metrics['path_coverage']:.2%}")

            # Cosine coverage
            cosine_metrics = self._evaluate_coverage(dataloader, None, k, "cosine")
            results["cosine"][k] = cosine_metrics
            print(f"  Cosine  — Answer: {cosine_metrics['answer_coverage']:.2%}, Path: {cosine_metrics['path_coverage']:.2%}")

            # Save per-k results
            k_output_dir = os.path.join(output_dir, dataset_name, str(k))
            os.makedirs(k_output_dir, exist_ok=True)

            summary = {
                "k": k,
                "kgscout": {key: val for key, val in kgscout_metrics.items() if key != "per_sample_results"},
                "cosine": {key: val for key, val in cosine_metrics.items() if key != "per_sample_results"},
            }
            with open(os.path.join(k_output_dir, "coverage_metrics.json"), "w") as f:
                json.dump(summary, f, indent=2)

        # Print comparison table
        self._print_comparison_table(results, k_values)

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    def _print_comparison_table(self, results: Dict, k_values: List[int]):
        """Print formatted comparison table."""
        print(f"\n{'='*80}")
        print("COVERAGE COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'K':<8} {'Retriever':<12} {'Answer Cov':<15} {'Path Cov':<15}")
        print("-" * 50)
        for k in k_values:
            kg = results["kgscout"][k]
            cos = results["cosine"][k]
            print(f"{k:<8} {'KGscout':<12} {kg['answer_coverage']:.2%}{'':>7} {kg['path_coverage']:.2%}")
            print(f"{'':8} {'Cosine':<12} {cos['answer_coverage']:.2%}{'':>7} {cos['path_coverage']:.2%}")
            print("-" * 50)
        print(f"{'='*80}\n")
