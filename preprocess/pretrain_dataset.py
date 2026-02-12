"""
Cosine pretraining dataset for pretraining phase.

This module implements CosinePretrainingDataset which is used during the
pretraining phase with fixed k=500 parameter.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import random
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CosinePretrainingDataset(Dataset):
    """
    Dataset for pretraining phase using cosine similarity targets.
    Fixed k=500 for all samples.
    
    This dataset wraps a base dataset (typically JointTrainingDatasetv3PPR)
    and samples exactly min(500, available_triplets) triplets per sample.
    Returns None for samples with empty paths to skip them during pretraining.
    """
    
    def __init__(self, original_dataset: Dataset, k: int = 500):
        """
        Initialize pretraining dataset with fixed k=500.
        
        Args:
            original_dataset: Base dataset (e.g., JointTrainingDatasetv3PPR)
            k: Fixed at 500 for pretraining (default=500)
        
        Requirements:
            - Requirement 2.3: Organize in preprocess/ directory
            - Requirement 2.6: Use fixed k=500
        """
        self.original_dataset = original_dataset
        self.k = k
    
    def __len__(self) -> int:
        """Return number of samples in base dataset."""
        return len(self.original_dataset)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Return sample or None if empty paths.
        Skips empty samples during pretraining.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing sampled data with exactly min(k, available_triplets) triplets,
            or None if the sample has empty paths
            
        Raises:
            KeyError: If required fields are missing from base sample
            ValueError: If sample data is invalid
            
        Requirements:
            - Requirement 2.7: Handle empty path samples by returning None
            
        Process:
            1. Get sample from base dataset
            2. Validate required fields
            3. Check if sample has empty paths (is_empty flag)
            4. If empty, return None to skip during pretraining
            5. Otherwise, sample min(k, available_triplets) triplets
            6. Return sample with sampled triplets
        """
        try:
            # Get base sample
            sample = self.original_dataset[idx]
            
            # Validate required fields
            required_fields = [
                "question", "is_empty", "q_entity", "a_entity", "answer",
                "question_embedding", "topk_linearized_triplets",
                "topk_linearized_triplet_embeddings", "topk_rel_data",
                "topK_rel_embeddings", "graph_features"
            ]
            
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                raise KeyError(
                    f"Sample {idx} missing required fields: {missing_fields}"
                )
            
            # Return None for empty samples (Requirement 2.7)
            if sample.get("is_empty", False):
                return None
            
            # Get number of available triplets
            num_triplets = len(sample["topk_rel_data"])
            
            # Return None if no triplets available
            if num_triplets == 0:
                logger.warning(f"Sample {idx}: No triplets available. Skipping sample.")
                return None
            
            # Validate consistency across fields
            if len(sample["topk_linearized_triplets"]) != num_triplets:
                raise ValueError(
                    f"Sample {idx}: Inconsistent data - topk_linearized_triplets has "
                    f"{len(sample['topk_linearized_triplets'])} items but topk_rel_data has {num_triplets}"
                )
            
            if sample["topk_linearized_triplet_embeddings"].shape[0] != num_triplets:
                raise ValueError(
                    f"Sample {idx}: Inconsistent data - topk_linearized_triplet_embeddings has "
                    f"{sample['topk_linearized_triplet_embeddings'].shape[0]} items but topk_rel_data has {num_triplets}"
                )
            
            if sample["topK_rel_embeddings"].shape[0] != num_triplets:
                raise ValueError(
                    f"Sample {idx}: Inconsistent data - topK_rel_embeddings has "
                    f"{sample['topK_rel_embeddings'].shape[0]} items but topk_rel_data has {num_triplets}"
                )
            
            if sample["graph_features"].shape[0] != num_triplets:
                raise ValueError(
                    f"Sample {idx}: Inconsistent data - graph_features has "
                    f"{sample['graph_features'].shape[0]} items but topk_rel_data has {num_triplets}"
                )
            
            # Determine how many to sample: min(k, available_triplets)
            num_to_sample = min(self.k, num_triplets)
            
            # Sample indices without replacement
            if num_to_sample < num_triplets:
                sampled_indices = random.sample(range(num_triplets), num_to_sample)
                sampled_indices.sort()  # Keep original order
            else:
                # Use all available triplets if k >= num_triplets
                sampled_indices = list(range(num_triplets))
            
            # Create sampled version of the sample
            sampled_sample = {
                "question": sample["question"],
                "is_empty": sample["is_empty"],
                "q_entity": sample["q_entity"],
                "a_entity": sample["a_entity"],
                "answer": sample["answer"],
                "question_embedding": sample["question_embedding"],
                "topk_linearized_triplets": [sample["topk_linearized_triplets"][i] for i in sampled_indices],
                "topk_linearized_triplet_embeddings": sample["topk_linearized_triplet_embeddings"][sampled_indices],
                "topk_rel_data": [sample["topk_rel_data"][i] for i in sampled_indices],
                "topK_rel_embeddings": sample["topK_rel_embeddings"][sampled_indices],
                "graph_features": sample["graph_features"][sampled_indices]
            }
            
            return sampled_sample
            
        except KeyError as e:
            logger.error(f"Sample {idx}: Missing required field - {e}")
            raise
        except ValueError as e:
            logger.error(f"Sample {idx}: Invalid data - {e}")
            raise
        except Exception as e:
            logger.error(f"Sample {idx}: Unexpected error during sampling - {e}")
            raise
