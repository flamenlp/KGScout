"""
Cosine pretraining dataset for pretraining phase.

Aligned with the working implementation in ablation-2/model_ablation.py.
Returns samples with cosine_targets (exponential positional decay)
and field names compatible with the Pretrainer.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CosinePretrainingDataset(Dataset):
    """
    Dataset for pretraining phase using cosine similarity targets.

    Wraps a base dataset (typically JointTrainingDatasetv3PPR loaded from .pt)
    and samples min(k, available_triplets) triplets per sample.

    Returns dict with:
        - question_embedding: [embed_dim]
        - path_embeddings: [n, embed_dim] (triplet embeddings)
        - rel_embeddings: [n, embed_dim] (relation embeddings)
        - cosine_targets: [n] (exponential positional decay targets)
        - graph_features: [n, 2] (PPR features)

    Returns None for empty samples (filtered by collate_fn).
    """

    def __init__(self, dataset: Dataset, k: int = 500):
        """
        Args:
            dataset: Base dataset (JointTrainingDatasetv3PPR or similar)
            k: Number of triplets to sample per question (default 500)
        """
        self.dataset = dataset
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Return sampled pretraining sample or None if empty.
        """
        try:
            data = self.dataset[idx]
        except Exception as e:
            logger.warning(f"Sample {idx}: Failed to load from base dataset: {e}")
            return None

        # Check for empty samples (matching ablation-2 conditions)
        if data.get("is_empty", False):
            return None

        if len(data.get("topk_linearized_triplets", [])) == 0 or len(data.get("q_entity", [])) == 0:
            return None

        # Get available triplet count
        if "topk_linearized_triplet_embeddings" not in data:
            return None

        num_triplets = data["topk_linearized_triplet_embeddings"].shape[0]
        if num_triplets == 0:
            return None

        # Determine how many to use
        n = min(self.k, num_triplets)

        # Always take first k triplets (ordered by cosine similarity, matching ablation-2)
        path_embeds = data["topk_linearized_triplet_embeddings"][:n]
        rel_embeds = data["topK_rel_embeddings"][:n]
        graph_feats = data["graph_features"][:n]

        return {
            "question_embedding": data["question_embedding"],
            "path_embeddings": path_embeds,
            "rel_embeddings": rel_embeds,
            "cosine_targets": torch.exp(-0.01 * torch.arange(n, dtype=torch.float32)),
            "graph_features": graph_feats,
        }
