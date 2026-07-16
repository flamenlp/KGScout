"""
Statistical Analysis Service for comparing retrievers with case categorization.

This service performs comprehensive statistical comparison between cosine and KGscout
retrievers by categorizing each question into one of six predefined cases based on
answer coverage and path coverage metrics.

Uses DataLoader(batch_size=1) pattern consistent with the rest of the project.
Results saved to: results/statistical-analysis/{dataset}/{kval}/
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


class StatisticalAnalysisService:
    """
    Service for statistical comparison between retrievers.

    Categorizes each question into one of six cases:
    - Case 1: Cosine no relevant, KGscout some relevant
    - Case 2: Cosine relevant no path, KGscout has path
    - Case 3: Both have relevant triplets (overlapping paths)
    - Case 4: Both have relevant triplets (non-overlapping paths)
    - Case 5: Cosine better than KGscout
    - Case 6: Both fail

    Uses DataLoader(batch_size=1) for iteration.
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
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        data = torch.load(dataset_path, weights_only=False, map_location="cpu")
        print(f"  Loaded {len(data)} samples from {dataset_path}")

        dataset = SampledJointTrainingDataset(data, k=sample_k)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint directory."""
        from src.model.path_ranker import PathRankingModel

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        model = PathRankingModel.from_pretrained(model_path, device=self.device)
        model.to(self.device)
        model.eval()
        print(f"  Model loaded from {model_path}")
        return model

    def _categorize_question(
        self,
        cosine_triplets: List[Tuple[str, str, str]],
        kgscout_triplets: List[Tuple[str, str, str]],
        q_entity: List[str],
        a_entity: List[str],
    ) -> str:
        """
        Categorize question into one of six cases based on coverage metrics.

        Returns:
            Case identifier string ('case1' through 'case6')
        """
        cosine_answer_cov = compute_answer_coverage(cosine_triplets, a_entity)
        cosine_path_cov = compute_path_coverage(cosine_triplets, q_entity, a_entity)
        kgscout_answer_cov = compute_answer_coverage(kgscout_triplets, a_entity)
        kgscout_path_cov = compute_path_coverage(kgscout_triplets, q_entity, a_entity)

        # Case 6: Both fail
        if not cosine_answer_cov and not kgscout_answer_cov:
            return 'case6'

        # Case 1: Cosine no relevant, KGscout some relevant
        if not cosine_answer_cov and kgscout_answer_cov:
            return 'case1'

        # Case 5: Cosine better (has path, KGscout doesn't)
        if cosine_path_cov and not kgscout_path_cov:
            return 'case5'

        # Case 2: Cosine relevant no path, KGscout has path
        if cosine_answer_cov and not cosine_path_cov and kgscout_path_cov:
            return 'case2'

        # Case 3 & 4: Both have paths
        if cosine_path_cov and kgscout_path_cov:
            cosine_edges = {(s.lower(), o.lower()) for s, _, o in cosine_triplets}
            kgscout_edges = {(s.lower(), o.lower()) for s, _, o in kgscout_triplets}
            if cosine_edges & kgscout_edges:
                return 'case3'
            else:
                return 'case4'

        return 'case3'

    def run_statistical_analysis(
        self,
        dataset_path: str,
        model_path: str,
        k: int,
        dataset_name: str = "cwq",
        output_dir: str = None,
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison analysis with case categorization.

        Pipeline:
        1. Load test DataLoader and model
        2. For each sample, select triplets with both retrievers
        3. Categorize each question
        4. Compute statistics and save results

        Args:
            dataset_path: Path to test .pt dataset file
            model_path: Path to model checkpoint directory
            k: Number of top triplets to select
            dataset_name: Dataset name for output directory
            output_dir: Base output directory (default: 'results/statistical-analysis')
            sample_k: Number of triplets to feed to model (default: 1000)

        Returns:
            Dictionary with case results and statistics
        """
        if output_dir is None:
            output_dir = "results/statistical-analysis"

        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*60}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Model:   {model_path}")
        print(f"  K: {k}")

        # Load
        dataloader = self._create_dataloader(dataset_path, sample_k=sample_k)
        model = self._load_model(model_path)

        # Categorize each question
        case_results = {f'case{i}': [] for i in range(1, 7)}
        case_descriptions = {
            'case1': 'Cosine no relevant, KGscout some relevant',
            'case2': 'Cosine relevant no path, KGscout has path',
            'case3': 'Both have relevant triplets (overlapping paths)',
            'case4': 'Both have relevant triplets (non-overlapping paths)',
            'case5': 'Cosine better than KGscout',
            'case6': 'Both fail',
        }

        for i, batch in enumerate(tqdm(dataloader, desc="Categorizing")):
            try:
                kgscout_triplets = select_triplets_kgscout(model, batch, k, self.device)
                cosine_triplets = select_triplets_cosine(batch, k)
                meta = _extract_metadata_from_batch(batch)

                q_ents = meta["q_entity"] if meta["q_entity"] else []
                a_ents = meta["a_entity"] if meta["a_entity"] else []

                if not a_ents:
                    continue

                case = self._categorize_question(cosine_triplets, kgscout_triplets, q_ents, a_ents)
                case_results[case].append({
                    "question_id": i,
                    "question": meta["question"],
                })
            except Exception:
                continue

        # Compute statistics
        total = sum(len(v) for v in case_results.values())
        statistics = {}
        for case_name, questions in case_results.items():
            count = len(questions)
            statistics[case_name] = {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0.0,
                "description": case_descriptions[case_name],
            }

        # Print table
        print(f"\n{'='*80}")
        print(f"STATISTICAL ANALYSIS RESULTS (k={k})")
        print(f"{'='*80}")
        print(f"{'Case':<10} {'Description':<50} {'Count':<8} {'%':<8}")
        print("-" * 80)
        for case_name in [f'case{i}' for i in range(1, 7)]:
            s = statistics[case_name]
            print(f"{case_name:<10} {s['description']:<50} {s['count']:<8} {s['percentage']:.2f}%")
        print(f"{'='*80}\n")

        # Save results
        k_output_dir = os.path.join(output_dir, dataset_name, str(k))
        os.makedirs(k_output_dir, exist_ok=True)

        output_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "dataset": dataset_name,
                "k": k,
                "total_questions": total,
            },
            "case_statistics": statistics,
            "examples_per_case": {
                case: questions[:5] for case, questions in case_results.items()
            },
        }

        output_file = os.path.join(k_output_dir, "statistical_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_file}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {"statistics": statistics, "output_file": output_file}
