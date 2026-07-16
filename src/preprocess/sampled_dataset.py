"""
Sampled joint training dataset for main training phase.

This module implements SampledJointTrainingDataset which wraps a base dataset
and samples k triplets from each sample for training with configurable k parameter.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict


class SampledJointTrainingDataset(Dataset):
    """
    Wrapper dataset that samples k triplets from each sample.
    Used for main training with configurable k parameter.
    
    This dataset wraps a base dataset (typically JointTrainingDatasetv3PPR)
    and samples exactly min(k, available_triplets) triplets per sample.
    """
    
    def __init__(self, dataset: Dataset, k: int):
        """
        Initialize sampled dataset with configurable k parameter.
        
        Args:
            dataset: Base dataset (e.g., JointTrainingDatasetv3PPR)
            k: Number of triplets to sample per question
        
        Requirements:
            - Requirement 2.2: Organize in preprocess/ directory
            - Requirement 2.5: Support configurable k parameter
        """
        self.dataset = dataset
        self.k = k
    
    def __len__(self) -> int:
        """Return number of samples in base dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Sample k triplets from the base dataset sample.

        Uses the minimum available count across all triplet-related fields to handle
        inconsistent data gracefully (same approach as ablation-2/model_ablation.py).

        Args:
            idx: Sample index

        Returns:
            Dict containing sampled data with exactly min(k, available_triplets) triplets
        """
        sample = self.dataset[idx]

        # Determine available triplets as the minimum across all fields
        # This handles dirty data where fields have inconsistent lengths
        num_triplets = min(
            len(sample["topk_rel_data"]),
            sample["topk_linearized_triplet_embeddings"].shape[0],
            sample["topK_rel_embeddings"].shape[0],
            sample["graph_features"].shape[0],
            len(sample["topk_linearized_triplets"]),
        )

        # Slice to consistent length, then take min(k, num_triplets)
        n = min(self.k, num_triplets)

        return {
            "question": sample["question"],
            "is_empty": sample["is_empty"],
            "q_entity": sample["q_entity"],
            "a_entity": sample["a_entity"],
            "answer": sample["answer"],
            "question_embedding": sample["question_embedding"],
            "topk_linearized_triplets": sample["topk_linearized_triplets"][:n],
            "topk_linearized_triplet_embeddings": sample["topk_linearized_triplet_embeddings"][:n],
            "topk_rel_data": sample["topk_rel_data"][:n],
            "topK_rel_embeddings": sample["topK_rel_embeddings"][:n],
            "graph_features": sample["graph_features"][:n],
        }
