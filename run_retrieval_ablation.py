#!/usr/bin/env python3
"""
Triplet Coverage Analysis for Ablation Results.

For each ablation variant, loads the trained model checkpoint + test data,
runs the model to select top-100 triplets, builds an undirected graph,
and computes:
  1. Answer entity presence (any answer entity is a node in the graph)
  2. Reasoning path existence (path from any q_entity to any a_entity)

Results saved to: ./results/<ablation-type>/<variant>/triplet-metrics/

Usage:
    python run_triplet_analysis.py                          # all variants
    python run_triplet_analysis.py --mode model             # model ablations only
    python run_triplet_analysis.py --mode reward            # reward ablations only
    python run_triplet_analysis.py --mode model --experiments no-ppr no-gate
"""

import os
import sys
import json
import re
import argparse
import logging
import time
from typing import Optional
from tqdm import tqdm

import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Required for torch.load() to unpickle datasets saved from notebooks
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# Import model architectures from ablation modules
from ablation.model_ablation import (
    PathRankingModelOriginal,
    PathRankingModelNoPPR,
    PathRankingModelNoRT,
    PathRankingModelNoTT,
    PathRankingModelNoGate,
    PathRankingModelNoRA,
    PathRankingModelNoTA,
)

# ============================================================================
# HARDCODED CONFIGURATION
# ============================================================================
MODEL_ABLATION_DIR = "./results/model-ablation"
REWARD_ABLATION_DIR = "./results/reward-ablation"
TEST_DATASET_PATH = "/mnt/LS226/LS25/sourav23099/cwq/cwq-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"
TOP_K = 100
SAMPLE_K = 1000  # number of triplets to feed to model
LOG_FILE = os.path.join("logs", "triplet_analysis.log")

MODEL_VARIANTS = ["no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"]
REWARD_VARIANTS = ["no_pres", "no_conn", "no_path", "only_pres", "only_conn", "only_cov"]

# Variant → model class mapping
MODEL_CLASS_MAP = {
    "no-ppr": PathRankingModelNoPPR,
    "no-rt": PathRankingModelNoRT,
    "no-tt": PathRankingModelNoTT,
    "no-gate": PathRankingModelNoGate,
    "no-ra": PathRankingModelNoRA,
    "no-ta": PathRankingModelNoTA,
}
REWARD_MODEL_CLASS = PathRankingModelOriginal

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("ablation.triplet_analysis")


def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


# ============================================================================
# DATASET
# ============================================================================

class SampledJointTrainingDataset(Dataset):
    def __init__(self, dataset, k=1000):
        self.dataset = dataset
        self.k = k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_available = data["topk_linearized_triplet_embeddings"].shape[0]
        use_nums = min(self.k, num_available)
        return {
            "question": data["question"],
            "is_empty": data["is_empty"],
            "q_entity": data["q_entity"],
            "a_entity": data["a_entity"],
            "answer": data["answer"],
            "question_embedding": data["question_embedding"],
            "topk_linearized_triplets": data["topk_linearized_triplets"][:use_nums],
            "topk_linearized_triplet_embeddings": data["topk_linearized_triplet_embeddings"][:use_nums],
            "topk_rel_data": data["topk_rel_data"][:use_nums],
            "topK_rel_embeddings": data["topK_rel_embeddings"][:use_nums],
            "graph_features": data["graph_features"][:use_nums]
        }


# ============================================================================
# CHECKPOINT FINDER
# ============================================================================

def _find_last_checkpoint(train_dir: str) -> Optional[str]:
    """Find the last epoch checkpoint directory (highest epoch number)."""
    if not os.path.exists(train_dir):
        return None
    checkpoints = []
    for d in os.listdir(train_dir):
        full_path = os.path.join(train_dir, d)
        if os.path.isdir(full_path) and "checkpoint" in d:
            checkpoints.append(full_path)
    if not checkpoints:
        return None

    def get_epoch(path):
        name = os.path.basename(path)
        match = re.search(r'epoch_(\d+)', name)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=get_epoch, reverse=True)
    return checkpoints[0]


# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

@torch.no_grad()
def evaluate_coverage(test_dataloader, model, device, top_k):
    """
    Run model on test data, select top_k triplets, compute coverage metrics.
    Returns (aggregate_metrics_dict, per_sample_list).
    """
    model.eval()
    ans_present_count = 0
    reasoning_path_count = 0
    total_count = 0
    per_sample_results = []

    for i, batch in enumerate(tqdm(test_dataloader, desc="    Coverage analysis")):
        try:
            triplets = [(data[1][0][0], data[1][1][0], data[1][2][0]) for data in batch["topk_rel_data"]]
            if len(triplets) == 0:
                continue

            q_ents = [p[0].lower() for p in batch["q_entity"]]
            a_ents = [p[0].lower() for p in batch["a_entity"]]

            if not a_ents:
                continue

            # Forward pass
            ques_embed = batch["question_embedding"].to(device)
            triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
            relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
            graph_features = batch["graph_features"].squeeze(0).to(device)

            ranking_scores, path_probs = model(ques_embed, triplet_embeds, relation_embeds, graph_features)
            # Deterministic top-k selection (no sampling at inference)
            k = min(top_k, len(triplets))
            top_k_scores, top_k_indices = torch.topk(ranking_scores, k)
            selected_triplets = [triplets[i] for i in top_k_indices.tolist()]

            # Build undirected graph
            G = nx.Graph()
            for s, p, o in selected_triplets:
                G.add_edge(s.lower(), o.lower())

            # Check answer entity presence
            ans_present = any(ent in G.nodes for ent in a_ents)
            if ans_present:
                ans_present_count += 1

            # Check reasoning path existence
            path_found = False
            for q_ent in q_ents:
                for a_ent in a_ents:
                    if q_ent in G.nodes and a_ent in G.nodes:
                        if nx.has_path(G, q_ent, a_ent):
                            path_found = True
                            break
                if path_found:
                    break
            if path_found:
                reasoning_path_count += 1

            total_count += 1

            # Per-sample record
            found_entities = [ent for ent in a_ents if ent in G.nodes]
            per_sample_results.append({
                "id": i,
                "question": batch["question"][0],
                "answer_entity_present": ans_present,
                "reasoning_path_exists": path_found,
                "answer_entities": a_ents,
                "answer_entities_found": found_entities,
                "question_entities": q_ents,
                "num_selected_triplets": len(selected_triplets)
            })

        except Exception:
            continue

    # Aggregate
    if total_count == 0:
        return {"total_samples": 0}, []

    metrics = {
        "total_samples": total_count,
        "answer_entity_present": ans_present_count,
        "answer_entity_present_pct": round(ans_present_count / total_count * 100, 2),
        "reasoning_path_exists": reasoning_path_count,
        "reasoning_path_exists_pct": round(reasoning_path_count / total_count * 100, 2)
    }
    return metrics, per_sample_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Triplet coverage analysis for ablation results")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"])
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("TRIPLET COVERAGE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Top-K: {TOP_K}")
    logger.info(f"Test dataset: {TEST_DATASET_PATH}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test dataset
    logger.info("Loading test dataset...")
    test_data = torch.load(TEST_DATASET_PATH, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)} samples")

    test_sampled = SampledJointTrainingDataset(test_data, k=SAMPLE_K)
    test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)

    # Build task list
    tasks = []
    if args.mode in ("all", "model"):
        variants = args.experiments if (args.experiments and args.mode == "model") else MODEL_VARIANTS
        for v in variants:
            tasks.append(("model-ablation", MODEL_ABLATION_DIR, v, MODEL_CLASS_MAP[v]))
    if args.mode in ("all", "reward"):
        variants = args.experiments if (args.experiments and args.mode == "reward") else REWARD_VARIANTS
        for v in variants:
            tasks.append(("reward-ablation", REWARD_ABLATION_DIR, v, REWARD_MODEL_CLASS))

    for ablation_type, base_dir, variant, model_class in tasks:
        logger.info(f"{'='*60}")
        logger.info(f"  Variant: {variant} ({ablation_type})")

        train_dir = os.path.join(base_dir, variant, "model", "trained")
        result_dir = os.path.join(base_dir, variant, "triplet-metrics")

        # Find checkpoint
        checkpoint_dir = _find_last_checkpoint(train_dir)
        if checkpoint_dir is None:
            logger.warning(f"  No checkpoint found in {train_dir}, skipping.")
            continue

        ckpt_path = os.path.join(checkpoint_dir, "path_ranker.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  path_ranker.pt not found in {checkpoint_dir}, skipping.")
            continue

        logger.info(f"  Checkpoint: {checkpoint_dir}")

        # Load model
        model = model_class(device=device_str)
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        # The checkpoint is a dict of sub-module state dicts (not a flat model state_dict)
        # Load each component individually
        for key, value in state.items():
            if key in ('temperature', 'baseline'):
                getattr(model, key).data = value
            elif hasattr(model, key):
                getattr(model, key).load_state_dict(value)
            else:
                logger.warning(f"  Unexpected key in checkpoint: '{key}' (model {model_class.__name__} has no such attribute)")
        model.to(device_str)
        model.eval()

        # Run coverage analysis
        metrics, per_sample = evaluate_coverage(test_loader, model, device_str, TOP_K)

        # Save results
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "coverage_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(result_dir, "per_sample_metrics.jsonl"), "w") as f:
            for record in per_sample:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  Results: {metrics}")
        logger.info(f"  Saved to: {result_dir}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL COVERAGE ANALYSIS COMPLETE. Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
