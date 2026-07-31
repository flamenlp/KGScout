"""
Hop-Stratified Retriever Analysis Service.

Classifies test questions by reasoning hop count (1-hop, 2-hop, ≥3-hop, no-path)
using the cosine top-1000 pool graph, then computes ans_present and has_path for
both KGScout and cosine retrievers at each k value (30, 50, 100, 150).

Hop classification is done once (retriever-agnostic, using cosine top-1000 pool).
Per-k metrics are computed for each model checkpoint independently.

Results saved to: results/hop-analysis/{dataset}/
  k{K}_hop_analysis.json  — per-k breakdown by hop bucket
  summary.json            — all k values in one file
"""

import os
import json
import torch
import torch.nn as nn
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.services.statistical_analysis_service import StatisticalAnalysisService
from src.utils.triplet_selector import (
    select_triplets_kgscout,
    select_triplets_cosine,
    _extract_metadata_from_batch,
)
from src.utils.failure_analysis import compute_min_hop


# ---------------------------------------------------------------------------
# Hop bucket labels
# ---------------------------------------------------------------------------

HOP_1 = "hop_1"
HOP_2 = "hop_2"
HOP_3P = "hop_3p"
NO_PATH = "no_path"


def _classify_hop(min_hop: Optional[int]) -> str:
    """Map a min_hop value to a bucket label."""
    if min_hop is None:
        return NO_PATH
    if min_hop == 1:
        return HOP_1
    if min_hop == 2:
        return HOP_2
    return HOP_3P


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class HopAnalysisService(StatisticalAnalysisService):
    """
    Hop-stratified retriever analysis.

    Inherits data loading, checkpoint resolution, graph utilities, and
    answer-entity checks from StatisticalAnalysisService. Adds:
      - Hop classification via cosine top-1000 pool graph (once, shared)
      - Per-k ans_present / has_path metrics broken down by hop bucket
    """

    # -------------------------------------------------------------------
    # Phase 1: Classify all questions by hop count (cosine pool, k=1000)
    # -------------------------------------------------------------------

    def classify_hops(
        self,
        dataloader: DataLoader,
        sample_k: int = 1000,
    ) -> Dict[int, str]:
        """
        Classify every question in the dataloader by hop count.

        Uses the cosine top-sample_k pool graph (retriever-agnostic).
        Bidirectional shortest-path search between q_entity and a_entity.

        Args:
            dataloader: DataLoader(batch_size=1, shuffle=False) over
                        SampledJointTrainingDataset(full_dataset, k=sample_k).
            sample_k:   Pool size used when building the dataloader (default 1000).

        Returns:
            Dict mapping question index → hop bucket label
            ("hop_1", "hop_2", "hop_3p", "no_path").
        """
        print("\n  [Phase 1] Classifying questions by hop count (cosine top-1000 pool)...")
        hop_labels: Dict[int, str] = {}

        for idx, batch in enumerate(tqdm(dataloader, desc="  hop-classify")):
            try:
                meta = _extract_metadata_from_batch(batch)
                q_ents = [e.lower() for e in meta["q_entity"]] if meta["q_entity"] else []
                a_ents = [e.lower() for e in meta["a_entity"]] if meta["a_entity"] else []

                if not q_ents or not a_ents:
                    hop_labels[idx] = NO_PATH
                    continue

                # Full cosine pool — use all sample_k candidates
                pool_triplets = select_triplets_cosine(batch, k=sample_k)
                G = self._build_graph(pool_triplets)
                paths = self._get_all_paths(G, q_ents, a_ents)
                min_hop = compute_min_hop(paths)
                hop_labels[idx] = _classify_hop(min_hop)

            except Exception:
                hop_labels[idx] = NO_PATH

        # Summary
        counts = Counter(hop_labels.values())
        print(f"  Hop distribution: 1-hop={counts[HOP_1]}, "
              f"2-hop={counts[HOP_2]}, "
              f"≥3-hop={counts[HOP_3P]}, "
              f"no-path={counts[NO_PATH]}")
        return hop_labels

    # -------------------------------------------------------------------
    # Phase 2: Compute per-k metrics broken down by hop bucket
    # -------------------------------------------------------------------

    def _compute_k_metrics(
        self,
        dataloader: DataLoader,
        hop_labels: Dict[int, str],
        model: nn.Module,
        k: int,
        ckpt_path: str,
    ) -> Dict[str, Any]:
        """
        For a single k value, iterate the dataloader and compute ans_present
        and has_path for KGScout and cosine retrievers, grouped by hop bucket.

        Args:
            dataloader: DataLoader(batch_size=1, shuffle=False).
            hop_labels: Pre-computed hop bucket per question index.
            model:      Loaded PathRankingModel for KGScout.
            k:          Top-k used for both retrievers.
            ckpt_path:  Checkpoint path (stored in metadata).

        Returns:
            Dict with per-hop-bucket stats for both retrievers plus metadata.
        """
        # Accumulators: bucket → retriever → list of booleans
        buckets = [HOP_1, HOP_2, HOP_3P, NO_PATH]
        retrievers = ["kgscout", "cosine"]

        # {bucket: {retriever: {"ans_present": [], "has_path": []}}}
        records: Dict[str, Dict[str, Dict[str, List[bool]]]] = {
            b: {r: {"ans_present": [], "has_path": []} for r in retrievers}
            for b in buckets
        }
        errors = 0
        skipped_no_entities = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc=f"  k={k}")):
                try:
                    meta = _extract_metadata_from_batch(batch)
                    q_ents = [e.lower() for e in meta["q_entity"]] if meta["q_entity"] else []
                    a_ents = [e.lower() for e in meta["a_entity"]] if meta["a_entity"] else []

                    if not a_ents or not q_ents:
                        skipped_no_entities += 1
                        continue

                    bucket = hop_labels.get(idx, NO_PATH)

                    # --- KGScout top-k ---
                    kg_triplets = select_triplets_kgscout(model, batch, k, self.device)
                    kg_ans = self._check_answer_entity_presence(kg_triplets, a_ents)
                    G_kg = self._build_graph(kg_triplets)
                    kg_paths = self._get_all_paths(G_kg, q_ents, a_ents)
                    kg_has_path = len(kg_paths) > 0

                    records[bucket]["kgscout"]["ans_present"].append(kg_ans)
                    records[bucket]["kgscout"]["has_path"].append(kg_has_path)

                    # --- Cosine top-k ---
                    cos_triplets = select_triplets_cosine(batch, k)
                    cos_ans = self._check_answer_entity_presence(cos_triplets, a_ents)
                    G_cos = self._build_graph(cos_triplets)
                    cos_paths = self._get_all_paths(G_cos, q_ents, a_ents)
                    cos_has_path = len(cos_paths) > 0

                    records[bucket]["cosine"]["ans_present"].append(cos_ans)
                    records[bucket]["cosine"]["has_path"].append(cos_has_path)

                except Exception:
                    errors += 1
                    continue

        # --- Aggregate ---
        def _agg(bools: List[bool]) -> Dict[str, Any]:
            n = len(bools)
            if n == 0:
                return {"count": 0, "rate": None}
            return {
                "count": n,
                "rate": round(sum(bools) / n, 4),
            }

        stats: Dict[str, Any] = {}
        for b in buckets:
            stats[b] = {}
            for r in retrievers:
                ap = records[b][r]["ans_present"]
                hp = records[b][r]["has_path"]
                stats[b][r] = {
                    "ans_present": _agg(ap),
                    "has_path": _agg(hp),
                }

        return {
            "metadata": {
                "k": k,
                "checkpoint": ckpt_path,
                "skipped_no_entities": skipped_no_entities,
                "errors": errors,
            },
            "hop_stats": stats,
        }

    # -------------------------------------------------------------------
    # Pretty-print helper
    # -------------------------------------------------------------------

    @staticmethod
    def _print_k_results(k: int, result: Dict[str, Any]) -> None:
        stats = result["hop_stats"]
        bucket_labels = {
            HOP_1: "1-hop",
            HOP_2: "2-hop",
            HOP_3P: "≥3-hop",
            NO_PATH: "no-path",
        }
        print(f"\n  {'Bucket':<10} {'Retriever':<10} "
              f"{'ans_present (rate)':<22} {'has_path (rate)':<18} {'N'}")
        print(f"  {'-'*75}")
        for b, label in bucket_labels.items():
            for r in ["kgscout", "cosine"]:
                ap = stats[b][r]["ans_present"]
                hp = stats[b][r]["has_path"]
                ap_str = f"{ap['rate']:.4f}" if ap["rate"] is not None else "—"
                hp_str = f"{hp['rate']:.4f}" if hp["rate"] is not None else "—"
                print(f"  {label:<10} {r:<10} {ap_str:<22} {hp_str:<18} {ap['count']}")

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def run_hop_analysis(
        self,
        dataset: str,
        test_data_path: str,
        k_values: List[int],
        results_base: str = "./results",
        output_dir: Optional[str] = None,
        sample_k: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run hop-stratified retriever analysis across multiple k values.

        Pipeline:
          Phase 1 (once): Classify all questions by hop count using cosine
                          top-1000 pool graph. Results cached per question index.
          Phase 2 (per k): For each k, load the best-val checkpoint, iterate
                           the dataloader, and compute ans_present / has_path
                           for KGScout and cosine at that k, grouped by hop bucket.

        Args:
            dataset:        Dataset name ("cwq" or "webqsp").
            test_data_path: Path to the test .pt file.
            k_values:       List of k values (e.g., [30, 50, 100, 150]).
            results_base:   Base results directory (default: "./results").
            output_dir:     Override output dir (default: results/hop-analysis/{dataset}).
            sample_k:       Pool size for SampledJointTrainingDataset (default: 1000).

        Returns:
            Dict with per-k result files and a summary file path.
        """
        if output_dir is None:
            output_dir = os.path.join(results_base, "hop-analysis", dataset)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"HOP ANALYSIS: {dataset}")
        print(f"{'='*70}")
        print(f"  K values:   {k_values}")
        print(f"  Test data:  {test_data_path}")
        print(f"  Sample K:   {sample_k}")
        print(f"  Device:     {self.device}")
        print(f"  Output:     {output_dir}")
        print(f"{'='*70}")

        # ------------------------------------------------------------------
        # Load dataset once — shared across all phases
        # ------------------------------------------------------------------
        print("\nLoading test dataset...")
        dataloader, _ = self._create_dataloader(test_data_path, sample_k=sample_k)

        # ------------------------------------------------------------------
        # Phase 1: Hop classification (cosine pool, runs once)
        # ------------------------------------------------------------------
        hop_labels = self.classify_hops(dataloader, sample_k=sample_k)

        # Persist hop labels to output dir for traceability
        hop_label_file = os.path.join(output_dir, "hop_labels.json")
        with open(hop_label_file, "w", encoding="utf-8") as f:
            # Convert int keys to str for JSON serialisation
            json.dump({str(i): v for i, v in hop_labels.items()}, f, indent=2)
        print(f"  Hop labels saved: {hop_label_file}")

        # ------------------------------------------------------------------
        # Phase 2: Per-k metrics
        # ------------------------------------------------------------------
        all_results: Dict[str, Any] = {}

        for k in k_values:
            print(f"\n{'─'*70}")
            print(f"  k = {k}")
            print(f"{'─'*70}")

            # Resolve checkpoint
            try:
                ckpt_path = self._resolve_checkpoint(dataset, k, results_base)
                print(f"  Checkpoint: {ckpt_path}")
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                all_results[f"k{k}"] = {"error": str(e)}
                continue

            # Load model
            model = self._load_model(ckpt_path)

            # Compute hop-stratified metrics
            result = self._compute_k_metrics(
                dataloader, hop_labels, model, k, ckpt_path
            )

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Print summary table
            self._print_k_results(k, result)

            # Augment with timestamp and dataset info
            result["metadata"].update({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset,
                "test_data": test_data_path,
                "sample_k": sample_k,
                "total_questions": len(dataloader),
            })

            # Save per-k file
            k_file = os.path.join(output_dir, f"k{k}_hop_analysis.json")
            with open(k_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {k_file}")

            all_results[f"k{k}"] = {
                "hop_stats": result["hop_stats"],
                "metadata": result["metadata"],
                "output_file": k_file,
            }

        # ------------------------------------------------------------------
        # Summary across all k values
        # ------------------------------------------------------------------
        summary = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset,
                "k_values": k_values,
                "test_data": test_data_path,
                "sample_k": sample_k,
                "hop_label_file": hop_label_file,
            },
            "per_k": {},
        }

        for k in k_values:
            key = f"k{k}"
            if key in all_results and "hop_stats" in all_results[key]:
                summary["per_k"][key] = all_results[key]["hop_stats"]
            elif key in all_results and "error" in all_results[key]:
                summary["per_k"][key] = {"error": all_results[key]["error"]}

        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"HOP ANALYSIS COMPLETE")
        print(f"  Summary: {summary_file}")
        print(f"{'='*70}")

        return {"summary_file": summary_file, "results": all_results}
