#!/usr/bin/env python3
"""
Generate selected_triplets.json from a trained Reversed Attention model.

Loads a model checkpoint and the test dataset, runs deterministic top-k
selection (no sampling at inference time), and writes the ranked triplets
to a JSON file compatible with run_inference.py.

Usage:
    python ablation-2/generate_triplets_file.py \
        --model-path results/ablation-2/cwq/model/trained/complete_10_best \
        --test-data /path/to/test_jointrainer_path_dataset_v3_ppr.pt \
        --output results/ablation-2/cwq/triplet-result/selected_triplets.json \
        --top-k 100

    # Or use dataset shorthand:
    python ablation-2/generate_triplets_file.py \
        --dataset cwq \
        --model-path results/ablation-2/cwq/model/trained/complete_10_best
"""

import os
import sys
import json
import logging
import argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR

# Allow loading datasets saved from notebooks
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

torch.manual_seed(100)
np.random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("generate_triplets")

# ============================================================================
# HARDCODED PATHS (for --dataset shorthand)
# ============================================================================

DATASET_CONFIGS = {
    "cwq": {
        "test_data": "/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/test/test_jointrainer_path_dataset_v3_ppr.pt",
        "output": "./results/ablation-2/cwq/triplet-result/selected_triplets.json",
    },
    "webqsp": {
        "test_data": "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/test/test_jointrainer_path_dataset_v3_ppr.pt",
        "output": "./results/ablation-2/webqsp/triplet-result/selected_triplets.json",
    },
}


# ============================================================================
# MODEL: Reversed Attention PathRankingModel (from run_reversed_attention.py)
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F


class PathRankingModelReversedAttention(nn.Module):
    """
    Model with REVERSED attention: Query=Question, Key=Value=Triplets/Relations.

    Difference from original:
      - Original:  MHA(query=T, key=q, value=q) → N×d (all rows identical)
      - Reversed:  MHA(query=q, key=T, value=T) → 1×d (question attending over triplets)
                   then expanded to N×d for downstream concatenation.
    """
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1), nn.Sigmoid())
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        num_triplets = triplet_embeds.size(0)
        question_embed = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed

        # REVERSED: Query=question(1,d), Key=Value=triplets(N,d)
        triplet_attended, triplet_weights = self.question_triplet_attention(
            question_embed, triplet_embeds, triplet_embeds)
        # REVERSED: Query=question(1,d), Key=Value=relations(N,d)
        relation_attended, relation_weights = self.question_relation_attention(
            question_embed, relation_embeds, relation_embeds)

        # Extract attention weights as per-triplet relevance scores
        triplet_weights = triplet_weights.squeeze(0).squeeze(0)  # (N,)
        relation_weights = relation_weights.squeeze(0).squeeze(0)  # (N,)
        if triplet_weights.dim() == 0:
            triplet_weights = triplet_weights.unsqueeze(0)
        if relation_weights.dim() == 0:
            relation_weights = relation_weights.unsqueeze(0)

        # Expand attended outputs to (N, d) for per-triplet scoring
        triplet_attended = triplet_attended.expand(num_triplets, -1)
        relation_attended = relation_attended.expand(num_triplets, -1)

        question_expanded = question_embed.expand(num_triplets, -1)
        gate_input = torch.cat([question_expanded, triplet_embeds, relation_embeds], dim=-1)
        path_gates = self.gate_network(gate_input).squeeze(-1)

        # Gated combination of attention weights
        gated_attention_weights = path_gates * triplet_weights + (1 - path_gates) * relation_weights

        triplet_centric_input = torch.cat([
            triplet_embeds, triplet_attended, question_expanded, graph_scores], dim=-1)
        tower_A_scores = self.triplet_mlp(triplet_centric_input).squeeze(-1)

        relation_centric_input = torch.cat([
            relation_embeds, relation_attended, question_expanded, graph_scores], dim=-1)
        tower_B_scores = self.relation_mlp(relation_centric_input).squeeze(-1)

        combiner_input = torch.stack([
            tower_A_scores, tower_B_scores, gated_attention_weights], dim=-1)
        combined_scores = self.combiner_mlp(combiner_input).squeeze(-1)

        temp = self.temperature.clamp(min=0.1, max=5.0)
        path_probs = F.softmax(combined_scores / temp, dim=0)
        return combined_scores, path_probs


# ============================================================================
# DATASET WRAPPER
# ============================================================================

class SampledDataset(Dataset):
    """Wraps a dataset and limits triplets to k per sample."""
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
            "graph_features": data["graph_features"][:use_nums],
        }


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "generate_triplets.log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path):
    """Load a trained PathRankingModelReversedAttention from checkpoint directory."""
    model = PathRankingModelReversedAttention(device=str(device))
    ckpt_path = os.path.join(model_path, "path_ranker.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"path_ranker.pt not found in: {model_path}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model.question_triplet_attention.load_state_dict(ckpt['question_triplet_attention'])
    model.question_relation_attention.load_state_dict(ckpt['question_relation_attention'])
    model.gate_network.load_state_dict(ckpt['gate_network'])
    model.triplet_mlp.load_state_dict(ckpt['triplet_mlp'])
    model.relation_mlp.load_state_dict(ckpt['relation_mlp'])
    model.combiner_mlp.load_state_dict(ckpt['combiner_mlp'])
    model.temperature.data = ckpt['temperature'].to(device)
    model.baseline.data = ckpt['baseline'].to(device)
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# TRIPLET GENERATION (DETERMINISTIC TOP-K, NO SAMPLING)
# ============================================================================

def format_relation(rel):
    """Convert 'award.award_nomination.award_nominee' to 'award award nomination award nominee'."""
    return rel.replace('.', ' ').replace('_', ' ')


@torch.no_grad()
def generate_triplets(model, test_data, top_k, output_path):
    """
    Run the model on test data using deterministic top-k selection (no sampling)
    and write selected_triplets.json.
    """
    logger.info("Generating triplets (deterministic top-k, no sampling)...")

    test_sampled = SampledDataset(test_data, k=1000)
    test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)

    results = []
    skipped = 0

    for batch in tqdm(test_loader, desc="  Generating triplets"):
        question = batch['question'][0]
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        q_entity = [p[0] for p in batch['q_entity']]
        a_entity = [p[0] for p in batch['a_entity']]
        ground_truth = [p[0] for p in batch["answer"]]
        # Extract structured triplets: (subject, relation, object)
        structured_triplets = [(d[1][0][0], d[1][1][0], d[1][2][0]) for d in batch["topk_rel_data"]]

        if len(paths) == 0 or len(q_entity) == 0:
            skipped += 1
            continue

        ques_embed = batch["question_embedding"].to(device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
        graph_features = batch["graph_features"].squeeze(0).to(device)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)

        # Forward pass
        ranking_scores, _ = model(ques_embed, triplet_embeds, relation_embeds, graph_features)

        # Deterministic top-k selection (greedy, NO sampling at inference)
        k = min(top_k, len(structured_triplets))
        top_k_scores, top_k_indices = torch.topk(ranking_scores, k)
        selected_triplets = [structured_triplets[i] for i in top_k_indices.tolist()]

        # Format as comma-separated strings: "subject, relation (spaces), object"
        formatted_triplets = [
            f"{s}, {format_relation(r)}, {o}" for s, r, o in selected_triplets
        ]

        results.append({
            "question": question,
            "q_entity": q_entity,
            "a_entity": a_entity,
            "answer": ground_truth,
            "reranker": formatted_triplets,
        })

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"  Generated triplets for {len(results)} samples (skipped {skipped})")
    logger.info(f"  Output saved to: {output_path}")
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate selected_triplets.json from a trained Reversed Attention model",
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint directory (contains path_ranker.pt)")
    parser.add_argument("--dataset", type=str, choices=["cwq", "webqsp"],
                        help="Use preset paths for dataset (alternative to --test-data/--output)")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test dataset .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for selected_triplets.json")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of triplets to select per sample (default: 100)")

    args = parser.parse_args()

    # Resolve paths from dataset shorthand
    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        if args.test_data is None:
            args.test_data = config["test_data"]
        if args.output is None:
            args.output = config["output"]
    else:
        if args.test_data is None or args.output is None:
            parser.error("Either --dataset or both --test-data and --output are required")

    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}")
        sys.exit(1)

    setup_logging(args.output)

    logger.info("=" * 70)
    logger.info("GENERATE SELECTED TRIPLETS (Reversed Attention Model)")
    logger.info(f"  Model:     {args.model_path}")
    logger.info(f"  Test data: {args.test_data}")
    logger.info(f"  Output:    {args.output}")
    logger.info(f"  Top-k:     {args.top_k}")
    logger.info(f"  Device:    {device}")
    logger.info("=" * 70)

    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path)
    logger.info("  Model loaded.")

    # Load test data
    logger.info("Loading test data...")
    test_data = torch.load(args.test_data, weights_only=False, map_location="cpu")
    logger.info(f"  Test samples: {len(test_data)}")

    # Generate triplets
    generate_triplets(model, test_data, top_k=args.top_k, output_path=args.output)

    logger.info("=" * 70)
    logger.info("DONE.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
