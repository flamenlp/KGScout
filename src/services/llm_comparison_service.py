"""
LLM Comparison Service for evaluating LLM models with various retrievers.

This service orchestrates the full evaluation pipeline:
1. Load dataset as DataLoader OR load pre-computed selected_triplets.json
2. Select triplets using KGscout or cosine retriever (if no pre-computed file)
3. Run LLM inference to generate answers
4. Compute evaluation metrics
5. Save results to results/llm-inference/{llm}/{kval}/

Uses DataLoader(batch_size=1) pattern consistent with the rest of the project.
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.utils.triplet_selector import (
    select_triplets_kgscout, select_triplets_cosine,
    generate_selected_json, _extract_metadata_from_batch, format_relation,
)
from src.utils.llm_inference import load_llm_model, format_prompt, run_llm_inference
from src.utils.metrics import (
    extract_predictions_from_response,
    compute_hit_score,
    compute_hit_at_1,
    compute_precision,
    compute_recall,
    compute_f1_score,
    should_use_double_check,
    preprocess_date_answers,
)


class LLMComparisonService:
    """
    Service for LLM inference evaluation with retriever-selected triplets.

    Supports two modes:
    1. Provide selected_triplets_path: Load pre-computed triplets, skip model inference.
    2. Provide model_path + dataset_path: Generate selected_triplets.json first, then run LLM.

    Results saved to: results/llm-inference/{llm}/{kval}/
    """

    def __init__(self, device: str = None):
        """
        Initialize service with device configuration.

        Args:
            device: Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LLMComparisonService initialized on device: {self.device}")

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

    def _load_selected_triplets(self, path: str) -> List[Dict]:
        """Load pre-computed selected_triplets.json."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Selected triplets file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} samples from {path}")
        return data

    def _run_llm_on_selected(
        self,
        data: List[Dict],
        llm_model_name: str,
        top_k: int,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Run LLM inference on pre-computed selected triplets and compute metrics.

        Args:
            data: List of dicts with 'question', 'answer', 'a_entity', 'q_entity', 'reranker'
            llm_model_name: LLM model name ('llama', 'qwen', 'deepseek')
            top_k: Number of triplets to include in prompt

        Returns:
            Tuple of (aggregate_metrics, detailed_results)
        """
        print(f"\nLoading LLM: {llm_model_name}")
        llm_model, tokenizer = load_llm_model(llm_model_name, self.device)
        print("  LLM loaded.")

        hit_list, hit1_list, f1_list = [], [], []
        precision_list, recall_list = [], []
        detailed_results = []

        for i, sample in enumerate(tqdm(data, desc="  LLM Inference")):
            try:
                question = sample["question"]
                ground_truth = sample.get("answer", sample.get("a_entity", []))
                q_entity = sample.get("q_entity", [])
                triplets = sample.get("reranker", [])

                if not ground_truth or not triplets:
                    continue

                # Ensure ground_truth is a flat list of strings
                if isinstance(ground_truth, str):
                    ground_truth = [ground_truth]
                ground_truth = [str(a) for a in ground_truth if a]
                if not ground_truth:
                    continue

                # Limit to top-k and format prompt
                triplets_used = triplets[:top_k]
                prompt = format_prompt(question, triplets_used, topk=top_k, q_entity=q_entity)

                # Run LLM
                raw_prediction = run_llm_inference(llm_model, tokenizer, prompt)
                prediction = extract_predictions_from_response(raw_prediction)
                prediction = [s for s in prediction if s != "" and s is not None]

                # Preprocess answers
                answer = preprocess_date_answers(question, ground_truth)
                double_check = should_use_double_check(question)

                # Compute metrics
                prec, _, _ = compute_precision(prediction, answer, double_check)
                rec, _, _ = compute_recall(prediction, answer, double_check)
                f1 = compute_f1_score(prec, rec)
                hit = compute_hit_score(prediction, answer, double_check)
                hit1 = compute_hit_at_1(prediction, answer, double_check)

                hit_list.append(hit)
                hit1_list.append(hit1)
                f1_list.append(f1)
                precision_list.append(prec)
                recall_list.append(rec)

                detailed_results.append({
                    "question": question,
                    "prediction": prediction,
                    "ground_truth": answer,
                    "hit": hit,
                    "hit_at_1": hit1,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                })

            except Exception as e:
                print(f"  Warning: sample {i} failed: {e}")
                continue

        # Aggregate
        import numpy as np
        n = len(hit_list)
        if n == 0:
            return {"total_samples": 0}, []

        metrics = {
            "hit": round(np.mean(hit_list) * 100, 2),
            "hit_at_1": round(np.mean(hit1_list) * 100, 2),
            "macro_f1": round(np.mean(f1_list) * 100, 2),
            "macro_precision": round(np.mean(precision_list) * 100, 2),
            "macro_recall": round(np.mean(recall_list) * 100, 2),
            "exact_match": round((np.array(f1_list) == 1).sum() / n * 100, 2),
            "total_samples": n,
        }

        print(f"\n  Results: Hit={metrics['hit']:.2f}%, Hit@1={metrics['hit_at_1']:.2f}%, "
              f"F1={metrics['macro_f1']:.2f}%, P={metrics['macro_precision']:.2f}%, "
              f"R={metrics['macro_recall']:.2f}%, EM={metrics['exact_match']:.2f}%")

        # Cleanup LLM
        del llm_model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return metrics, detailed_results

    def run_comparison(
        self,
        llm_model: str,
        k: int,
        selected_triplets_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        model_path: Optional[str] = None,
        retriever_type: str = "kgscout",
        output_dir: str = "results/llm-inference",
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run LLM comparison analysis.

        Two modes:
        1. If selected_triplets_path is provided: Load triplets from JSON, skip model.
        2. If dataset_path + model_path provided: Generate selected_triplets.json first.

        Args:
            llm_model: LLM model name ('llama', 'qwen', 'deepseek')
            k: Number of top triplets
            selected_triplets_path: Path to pre-computed selected_triplets.json (optional)
            dataset_path: Path to test .pt dataset (required if no selected_triplets_path)
            model_path: Path to model checkpoint (required if retriever_type='kgscout' and no selected_triplets_path)
            retriever_type: 'kgscout' or 'cosine'
            output_dir: Base output directory (default: 'results/llm-inference')
            sample_k: Number of triplets to sample per question (default: 1000)

        Returns:
            Dictionary with metrics and output file paths
        """
        print("=" * 60)
        print("LLM COMPARISON SERVICE")
        print("=" * 60)
        print(f"  LLM: {llm_model}")
        print(f"  K: {k}")
        print(f"  Retriever: {retriever_type}")

        # Result directory: results/llm-inference/{llm}/{kval}/
        result_dir = os.path.join(output_dir, llm_model, str(k))
        os.makedirs(result_dir, exist_ok=True)

        # Step 1: Get selected_triplets.json
        if selected_triplets_path and os.path.exists(selected_triplets_path):
            print(f"\n  Using pre-computed triplets: {selected_triplets_path}")
            data = self._load_selected_triplets(selected_triplets_path)
        else:
            # Generate selected_triplets.json from model + dataset
            if dataset_path is None:
                raise ValueError("Either selected_triplets_path or dataset_path must be provided")

            print(f"\n  Generating selected_triplets.json...")
            dataloader = self._create_dataloader(dataset_path, sample_k=sample_k)

            model = None
            if retriever_type == "kgscout":
                if model_path is None:
                    raise ValueError("model_path is required for kgscout retriever")
                model = self._load_model(model_path)

            triplet_output_dir = os.path.join(result_dir, "triplet-result")
            selected_triplets_path = generate_selected_json(
                dataloader=dataloader,
                model=model,
                output_dir=triplet_output_dir,
                k=k,
                retriever_type=retriever_type,
                device=self.device,
            )
            print(f"  Saved: {selected_triplets_path}")

            # Cleanup model
            if model is not None:
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            data = self._load_selected_triplets(selected_triplets_path)

        # Step 2: Run LLM inference
        metrics, detailed_results = self._run_llm_on_selected(data, llm_model, k)

        # Step 3: Save results
        metrics_path = os.path.join(result_dir, "llm_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        detailed_path = os.path.join(result_dir, "llm_detailed_results.json")
        with open(detailed_path, "w") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"\n  Metrics saved to: {metrics_path}")
        print(f"  Detailed results: {detailed_path}")
        print("=" * 60)

        return {
            **metrics,
            "metrics_file": metrics_path,
            "detailed_file": detailed_path,
        }
