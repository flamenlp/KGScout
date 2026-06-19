"""
Sampled joint training dataset for main training phase.

This module implements SampledJointTrainingDataset which wraps a base dataset
and samples k triplets from each sample for training with configurable k parameter.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import random
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing sampled data with exactly min(k, available_triplets) triplets
            
        Raises:
            KeyError: If required fields are missing from base sample
            ValueError: If sample data is invalid
            
        Process:
            1. Get sample from base dataset
            2. Validate required fields
            3. Determine number of available triplets
            4. Sample min(k, available_triplets) triplets
            5. Return sample with sampled triplets
        """
        try:
            # Get base sample
            sample = self.dataset[idx]
            
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
            
            # Get number of available triplets
            num_triplets = len(sample["topk_rel_data"])
            
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
            
            # Handle empty samples
            if num_triplets == 0:
                logger.warning(f"Sample {idx}: No triplets available. Returning empty sample.")
                return sample
            
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
