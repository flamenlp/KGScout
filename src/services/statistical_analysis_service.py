"""
Statistical Analysis Service for comparing retrievers with case categorization.

Performs pairwise comparison between cosine-selected and KGScout-selected triplets
across multiple k values. Loads the test dataset and model checkpoint for each k,
running inference to get (s, r, o) triplets directly — no JSON parsing needed.

Results saved to: results/statistical-analysis/{dataset}/
"""

import os
import json
import torch
import torch.nn as nn
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.utils.triplet_selector import (
    select_triplets_kgscout, select_triplets_cosine, _extract_metadata_from_batch
)


class StatisticalAnalysisService:
    """
    Service for statistical comparison between retrievers.

    Categorizes each question into one of six cases:
    - Case 1: Cosine no relevant, KGscout some relevant
    - Case 2: Cosine relevant no path, KGscout has path
    - Case 3: Both have relevant triplets (overlapping paths, Jaccard >= 0.7)
    - Case 4: Both have relevant triplets (non-overlapping paths, Jaccard <= 0.3)
    - Case 5: Cosine better than KGscout
    - Case 6: Both fail

    Loads test dataset and model checkpoints directly (no JSON parsing ambiguity).
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------

    def _create_dataloader(self, dataset_path: str, sample_k: int = 1000) -> DataLoader:
        """Load test dataset and create DataLoader(batch_size=1)."""
        import __main__
        from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
        __main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        data = torch.load(dataset_path, weights_only=False, map_location="cpu")
        print(f"  Loaded {len(data)} samples from {dataset_path}")

        dataset = SampledJointTrainingDataset(data, k=sample_k)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint file (path_ranker.pt)."""
        from src.model.path_ranker import PathRankingModel

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # model_path points to the directory containing path_ranker.pt
        model_dir = os.path.dirname(model_path)
        model = PathRankingModel.from_pretrained(model_dir, device=self.device)
        model.to(self.device)
        model.eval()
        print(f"  Model loaded from {model_path}")
        return model

    # -------------------------------------------------------------------
    # Checkpoint resolution
    # -------------------------------------------------------------------

    @staticmethod
    def _resolve_checkpoint(
        dataset: str,
        k: int,
        results_base: str = "./results",
    ) -> str:
        """
        Resolve model checkpoint path for a given dataset and k value.

        Search order:
        1. k-ablation/{dataset}/k{K}/model/main_training_k{K}/
        2. full-pipeline/{dataset}/k{K}-N1000/model/main_training_k{K}/

        Uses find_checkpoint logic (best checkpoint, counting down from epoch 30).

        Returns:
            Path to path_ranker.pt

        Raises:
            FileNotFoundError if no checkpoint found.
        """
        import importlib.util
        # Load find_checkpoint.py as a module without importing via package path
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "find_checkpoint.py"
        )
        spec = importlib.util.spec_from_file_location("find_checkpoint_module", script_path)
        fc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fc_module)
        find_checkpoint = fc_module.find_checkpoint

        # Primary: k-ablation
        primary_train_dir = os.path.join(
            results_base, "k-ablation", dataset, f"k{k}", "model", f"main_training_k{k}"
        )
        if os.path.isdir(primary_train_dir):
            try:
                return find_checkpoint(primary_train_dir, pick_last=False)
            except FileNotFoundError:
                pass

        # Fallback: full-pipeline (especially for k=100)
        fallback_train_dir = os.path.join(
            results_base, "full-pipeline", dataset, f"k{k}-N1000", "model", f"main_training_k{k}"
        )
        if os.path.isdir(fallback_train_dir):
            try:
                return find_checkpoint(fallback_train_dir, pick_last=False)
            except FileNotFoundError:
                pass

        raise FileNotFoundError(
            f"No model checkpoint found for {dataset} k={k}.\n"
            f"  Checked: {primary_train_dir}\n"
            f"  Checked: {fallback_train_dir}\n"
            f"  Please train a model first (just full-pipeline {dataset} {k})."
        )

    # -------------------------------------------------------------------
    # Graph & path utilities (matches notebook logic)
    # -------------------------------------------------------------------

    @staticmethod
    def _build_graph(triplets: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """Build a directed graph from triplets (matches notebook's create_graph)."""
        G = nx.DiGraph()
        for s, r, o in triplets:
            G.add_edge(s.lower(), o.lower(), relation=r.lower())
        return G

    @staticmethod
    def _get_all_paths(
        G: nx.DiGraph,
        q_entities: List[str],
        a_entities: List[str],
    ) -> List[List[str]]:
        """
        Find reasoning paths between question and answer entities.
        Matches notebook's check_reasoning_path: bidirectional shortest path
        search on the directed graph (q→a and a→q).
        """
        paths = []
        # Forward: question → answer
        for q_ent in q_entities:
            qn = q_ent.lower()
            if qn not in G.nodes:
                continue
            for a_ent in a_entities:
                an = a_ent.lower()
                if an not in G.nodes:
                    continue
                try:
                    if nx.has_path(G, qn, an):
                        path = nx.shortest_path(G, qn, an)
                        paths.append(path)
                except (nx.NetworkXError, nx.NodeNotFound):
                    continue
        # Backward: answer → question
        for a_ent in a_entities:
            an = a_ent.lower()
            if an not in G.nodes:
                continue
            for q_ent in q_entities:
                qn = q_ent.lower()
                if qn not in G.nodes:
                    continue
                try:
                    if nx.has_path(G, an, qn):
                        path = nx.shortest_path(G, an, qn)
                        paths.append(path)
                except (nx.NetworkXError, nx.NodeNotFound):
                    continue
        return paths

    @staticmethod
    def _get_path_triplets(
        G: nx.DiGraph,
        path: List[str],
    ) -> List[Tuple[str, str, str]]:
        """
        Extract triplets along a path from the directed graph.
        Matches notebook's get_path_triplets: checks directed edge existence.
        """
        if not path or len(path) < 2:
            return []
        triplets = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            if G.has_edge(source, target):
                relation = G[source][target].get('relation', 'unknown')
                triplets.append((source, relation, target))
        return triplets

    @staticmethod
    def _compute_path_jaccard(
        G_a: nx.DiGraph,
        paths_a: List[List[str]],
        G_b: nx.DiGraph,
        paths_b: List[List[str]],
    ) -> float:
        """
        Compute Jaccard similarity over path triplets between two sets of paths.
        Matches notebook logic: collects all unique (s, r, o) triplets from all paths,
        then computes |intersection| / |union|.
        """
        triplets_a = set()
        for path in paths_a:
            for t in StatisticalAnalysisService._get_path_triplets(G_a, path):
                triplets_a.add((t[0].lower(), t[1].lower(), t[2].lower()))

        triplets_b = set()
        for path in paths_b:
            for t in StatisticalAnalysisService._get_path_triplets(G_b, path):
                triplets_b.add((t[0].lower(), t[1].lower(), t[2].lower()))

        union = triplets_a | triplets_b
        if not union:
            return 0.0
        intersection = triplets_a & triplets_b
        return len(intersection) / len(union)

    # -------------------------------------------------------------------
    # Answer entity check (matches notebook's check_answer_entity_presence)
    # -------------------------------------------------------------------

    @staticmethod
    def _check_answer_entity_presence(
        triplets: List[Tuple[str, str, str]],
        answer_entities: List[str],
    ) -> bool:
        """Check if any answer entity is present in the triplets (as subject or object)."""
        if not answer_entities:
            return False
        return any(
            ent.lower() in {s.lower(), o.lower()}
            for ent in answer_entities
            for s, _, o in triplets
        )

    # -------------------------------------------------------------------
    # Case categorization (matches notebook logic)
    # -------------------------------------------------------------------

    def _categorize_question(
        self,
        cosine_triplets: List[Tuple[str, str, str]],
        kgscout_triplets: List[Tuple[str, str, str]],
        q_entity: List[str],
        a_entity: List[str],
    ) -> Optional[str]:
        """
        Categorize question into one of six cases based on coverage metrics.

        Uses Jaccard similarity on path triplets with thresholds (matching notebook):
        - overlap_ratio <= 0.3 → case4 (non-overlapping)
        - overlap_ratio >= 0.7 → case3 (overlapping)
        - 0.3 < overlap_ratio < 0.7 → None (grey zone, skipped)

        Returns:
            Case identifier string ('case1' through 'case6'), or None if in the
            Jaccard grey zone (0.3–0.7).
        """
        cosine_answer_cov = self._check_answer_entity_presence(cosine_triplets, a_entity)
        kgscout_answer_cov = self._check_answer_entity_presence(kgscout_triplets, a_entity)

        # Build directed graphs for path analysis
        G_cosine = self._build_graph(cosine_triplets)
        G_kgscout = self._build_graph(kgscout_triplets)

        # Find reasoning paths
        cosine_paths = self._get_all_paths(G_cosine, q_entity, a_entity)
        kgscout_paths = self._get_all_paths(G_kgscout, q_entity, a_entity)

        cosine_has_path = len(cosine_paths) > 0
        kgscout_has_path = len(kgscout_paths) > 0

        # Case 6: Both fail (no answer entity in either)
        if not cosine_answer_cov and not kgscout_answer_cov:
            return 'case6'

        # Case 1: Cosine no relevant, KGscout some relevant
        if not cosine_answer_cov and kgscout_answer_cov:
            return 'case1'

        # Case 5: Cosine better (cosine has answer, KGscout doesn't)
        if cosine_answer_cov and not kgscout_answer_cov:
            return 'case5'

        # Case 2: Cosine relevant no path, KGscout has path
        if cosine_answer_cov and not cosine_has_path and kgscout_has_path:
            return 'case2'

        # Case 5b: Both have answer, cosine has path but KGscout doesn't
        if cosine_has_path and not kgscout_has_path:
            return 'case5'

        # Case 3 & 4: Both have paths — use Jaccard on path triplets
        if cosine_has_path and kgscout_has_path:
            overlap_ratio = self._compute_path_jaccard(
                G_cosine, cosine_paths, G_kgscout, kgscout_paths
            )
            if overlap_ratio <= 0.3:
                return 'case4'  # non-overlapping
            elif overlap_ratio >= 0.7:
                return 'case3'  # overlapping
            else:
                return None  # grey zone — uncategorized

        # Fallback (e.g., both have answer but neither has path)
        return 'case3'

    # -------------------------------------------------------------------
    # Main analysis entry point
    # -------------------------------------------------------------------

    def run_statistical_analysis(
        self,
        dataset: str,
        test_data_path: str,
        k_values: List[int],
        results_base: str = "./results",
        output_dir: str = None,
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run statistical comparison analysis across multiple k values.

        For each k value:
        1. Load model checkpoint (from k-ablation or full-pipeline)
        2. Iterate test DataLoader
        3. For each question, select triplets with both retrievers
        4. Categorize into cases and compute statistics

        Args:
            dataset: Dataset name ('cwq' or 'webqsp')
            test_data_path: Path to the test .pt file
            k_values: List of k values to analyze (e.g., [30, 50, 100, 150])
            results_base: Base results directory (default: './results')
            output_dir: Output directory (default: results/statistical-analysis/{dataset})
            sample_k: Pool size for model input (default: 1000)

        Returns:
            Dictionary with per-k statistics and output paths
        """
        if output_dir is None:
            output_dir = os.path.join(results_base, "statistical-analysis", dataset)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"STATISTICAL ANALYSIS: {dataset}")
        print(f"{'='*70}")
        print(f"  K values: {k_values}")
        print(f"  Test data: {test_data_path}")
        print(f"  Sample K (pool): {sample_k}")
        print(f"  Device: {self.device}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}")

        # Load test dataloader once (shared across all k values)
        print("\nLoading test dataset...")
        dataloader = self._create_dataloader(test_data_path, sample_k=sample_k)

        all_results = {}

        for k in k_values:
            print(f"\n{'─'*70}")
            print(f"  k = {k}")
            print(f"{'─'*70}")

            # Resolve and load model checkpoint
            try:
                ckpt_path = self._resolve_checkpoint(dataset, k, results_base)
                print(f"  Checkpoint: {ckpt_path}")
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                all_results[f"k{k}"] = {"error": str(e)}
                continue

            model = self._load_model(ckpt_path)

            # Categorize each question
            case_results = {f'case{i}': [] for i in range(1, 7)}
            skipped_grey_zone = 0
            skipped_no_entities = 0
            errors = 0

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader, desc=f"  k={k}")):
                    try:
                        meta = _extract_metadata_from_batch(batch)

                        # Lowercase entities upfront (matching notebook)
                        q_ents = [e.lower() for e in meta["q_entity"]] if meta["q_entity"] else []
                        a_ents = [e.lower() for e in meta["a_entity"]] if meta["a_entity"] else []

                        if not a_ents or not q_ents:
                            skipped_no_entities += 1
                            continue

                        # Select triplets — returns List[(s, r, o)] tuples directly
                        kgscout_triplets = select_triplets_kgscout(model, batch, k, self.device)
                        cosine_triplets = select_triplets_cosine(batch, k)

                        # Categorize
                        case = self._categorize_question(
                            cosine_triplets, kgscout_triplets, q_ents, a_ents
                        )

                        if case is None:
                            skipped_grey_zone += 1
                            continue

                        case_results[case].append({
                            "question_id": idx,
                            "question": meta["question"],
                        })
                    except Exception as e:
                        errors += 1
                        continue

            # Free model memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute statistics
            total_categorized = sum(len(v) for v in case_results.values())
            case_descriptions = {
                'case1': 'Cosine no relevant, KGscout some relevant',
                'case2': 'Cosine relevant no path, KGscout has path',
                'case3': 'Both paths overlapping (Jaccard >= 0.7)',
                'case4': 'Both paths non-overlapping (Jaccard <= 0.3)',
                'case5': 'Cosine better than KGscout',
                'case6': 'Both fail',
            }

            statistics = {}
            for case_name, questions in case_results.items():
                count = len(questions)
                statistics[case_name] = {
                    "count": count,
                    "percentage": round((count / total_categorized * 100), 2) if total_categorized > 0 else 0.0,
                    "description": case_descriptions[case_name],
                }

            # Print table
            print(f"\n  {'Case':<10} {'Description':<45} {'Count':<8} {'%':<8}")
            print(f"  {'-'*75}")
            for case_name in [f'case{i}' for i in range(1, 7)]:
                s = statistics[case_name]
                print(f"  {case_name:<10} {s['description']:<45} {s['count']:<8} {s['percentage']:.2f}%")
            print(f"  {'-'*75}")
            print(f"  Total categorized: {total_categorized}")
            print(f"  Skipped (grey zone 0.3–0.7): {skipped_grey_zone}")
            print(f"  Skipped (no entities): {skipped_no_entities}")
            print(f"  Errors: {errors}")

            # Save per-k results
            k_output = {
                "metadata": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": dataset,
                    "k": k,
                    "checkpoint": ckpt_path,
                    "test_data": test_data_path,
                    "sample_k": sample_k,
                    "total_questions": len(dataloader),
                    "total_categorized": total_categorized,
                    "skipped_grey_zone": skipped_grey_zone,
                    "skipped_no_entities": skipped_no_entities,
                    "errors": errors,
                },
                "case_statistics": statistics,
                "case_results": {
                    case: questions for case, questions in case_results.items()
                },
            }

            k_file = os.path.join(output_dir, f"k{k}_statistical_analysis.json")
            with open(k_file, 'w', encoding='utf-8') as f:
                json.dump(k_output, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {k_file}")

            all_results[f"k{k}"] = {
                "statistics": statistics,
                "output_file": k_file,
            }

        # Save summary across all k values
        summary = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset,
                "k_values": k_values,
                "test_data": test_data_path,
            },
            "per_k_summary": {},
        }
        for k in k_values:
            key = f"k{k}"
            if key in all_results and "statistics" in all_results[key]:
                summary["per_k_summary"][key] = all_results[key]["statistics"]
            elif key in all_results and "error" in all_results[key]:
                summary["per_k_summary"][key] = {"error": all_results[key]["error"]}

        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"STATISTICAL ANALYSIS COMPLETE")
        print(f"  Summary: {summary_file}")
        print(f"{'='*70}")

        return {"summary_file": summary_file, "results": all_results}
