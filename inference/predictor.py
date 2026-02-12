"""
Inference module for selecting top-k triplets from test data.

This module implements the Predictor class which loads a trained model
and runs inference to select the highest scoring triplets.
"""

import torch
import json
import os
import logging
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.rewards import compute_reward_v8

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Handles inference to select top-k triplets from test data.
    
    The predictor:
    - Loads a trained PathRankingModel
    - Runs forward pass on test samples
    - Selects top-k triplets based on ranking scores
    - Saves results to JSON files
    """
    
    # Required fields for inference results (Requirement 6.4)
    REQUIRED_RESULT_FIELDS = {
        "question",
        "question_entities",
        "answer_entities",
        "selected_triplets",
        "scores",
        "reward"
    }
    
    def __init__(self, model, device: str = "cuda"):
        """
        Initialize predictor with model and device.
        
        Args:
            model: Trained PathRankingModel instance
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @staticmethod
    def validate_batch_format(batch: Dict) -> None:
        """
        Validate that batch contains all required fields for inference.
        
        Args:
            batch: Batch dictionary to validate
            
        Raises:
            ValueError: If batch is missing required fields or has format mismatches
            
        Error Handling:
            - If test data format doesn't match training data format, raise ValueError with format details
        """
        required_fields = {
            "question",
            "q_entity",
            "a_entity",
            "question_embedding",
            "topk_linearized_triplet_embeddings",
            "topK_rel_embeddings",
            "graph_features",
            "topk_rel_data"
        }
        
        missing_fields = required_fields - set(batch.keys())
        if missing_fields:
            raise ValueError(
                f"Test data format mismatch: batch is missing required fields: {missing_fields}. "
                f"Required fields are: {required_fields}. "
                f"This may indicate that test data was not preprocessed with the same pipeline as training data."
            )
    
    @staticmethod
    def validate_result_structure(result: Dict) -> None:
        """
        Validate that result contains all required fields.
        
        Args:
            result: Result dictionary to validate
            
        Raises:
            ValueError: If result is missing required fields
            
        Requirements:
            - 6.4: WHEN saving results, THE System SHALL include selected triplets and metadata
        """
        missing_fields = Predictor.REQUIRED_RESULT_FIELDS - set(result.keys())
        if missing_fields:
            raise ValueError(
                f"Result structure is missing required fields: {missing_fields}. "
                f"Required fields are: {Predictor.REQUIRED_RESULT_FIELDS}"
            )
    
    def predict(
        self,
        test_dataloader: DataLoader,
        top_k: int,
        output_dir: str
    ):
        """
        Run inference on test dataset.
        
        Process:
        1. Load test samples from dataloader
        2. Forward pass to get ranking scores
        3. Select top-k triplets with highest scores
        4. Save results to JSON files in output directory
        
        Args:
            test_dataloader: DataLoader with test samples
            top_k: Number of top triplets to select
            output_dir: Directory to save results
            
        Raises:
            PermissionError: If output directory is not writable
            ValueError: If test data format doesn't match training data format
            
        Error Handling:
            - If output directory is not writable, raise PermissionError before inference starts
            - If test data format doesn't match training data format, raise ValueError with format details
            - If model produces NaN scores, log warning and use uniform distribution
        """
        # Check if output directory is writable before starting inference
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Test write permissions by creating a temporary file
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            raise PermissionError(
                f"Output directory '{output_dir}' is not writable. "
                f"Please check directory permissions or specify a different output directory. "
                f"Error: {e}"
            )
        
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Running inference")):
                # Validate batch format (Error Handling: format mismatches)
                try:
                    self.validate_batch_format(batch)
                except ValueError as e:
                    logger.error(f"Batch {batch_idx} format validation failed: {e}")
                    raise
                
                # Extract batch data
                question = batch["question"][0] if isinstance(batch["question"], list) else batch["question"]
                q_entities = batch["q_entity"]
                a_entities = batch["a_entity"]
                
                # Get embeddings and features
                question_embed = batch["question_embedding"].to(self.device)
                triplet_embeds = batch["topk_linearized_triplet_embeddings"].to(self.device)
                relation_embeds = batch["topK_rel_embeddings"].to(self.device)
                graph_scores = batch["graph_features"].to(self.device)
                
                # Get triplets
                triplets = [t[1] for t in batch["topk_rel_data"]]
                
                # Handle batch dimension - squeeze if needed
                if question_embed.dim() == 3:
                    question_embed = question_embed.squeeze(0)
                if triplet_embeds.dim() == 3:
                    triplet_embeds = triplet_embeds.squeeze(0)
                if relation_embeds.dim() == 3:
                    relation_embeds = relation_embeds.squeeze(0)
                if graph_scores.dim() == 3:
                    graph_scores = graph_scores.squeeze(0)
                
                # Ensure question_embed has correct shape (1, hidden_size)
                if question_embed.dim() == 1:
                    question_embed = question_embed.unsqueeze(0)
                
                # Forward pass to get ranking scores
                combined_scores, _ = self.model.forward(
                    question_embed=question_embed,
                    triplet_embeds=triplet_embeds,
                    relation_embeds=relation_embeds,
                    graph_scores=graph_scores
                )
                
                # Handle NaN scores (Error Handling: NaN scores with uniform distribution)
                scores_tensor = combined_scores.squeeze()
                if torch.isnan(scores_tensor).any():
                    logger.warning(
                        f"Batch {batch_idx}: Model produced NaN scores. "
                        f"Using uniform distribution for triplet selection."
                    )
                    # Use uniform distribution: all triplets have equal scores
                    scores_tensor = torch.ones_like(scores_tensor) / len(scores_tensor)
                
                # Select top-k triplets
                selected_triplets, selected_scores = self.select_top_k_triplets(
                    scores=scores_tensor,
                    triplets=triplets,
                    k=top_k
                )
                
                # Compute reward for selected triplets
                reward = compute_reward_v8(
                    triplets=selected_triplets,
                    q_entities=q_entities if isinstance(q_entities, list) else q_entities.tolist(),
                    a_entities=a_entities if isinstance(a_entities, list) else a_entities.tolist()
                )
                
                # Store result with all required fields
                result = {
                    "question": question,
                    "question_entities": q_entities if isinstance(q_entities, list) else q_entities.tolist(),
                    "answer_entities": a_entities if isinstance(a_entities, list) else a_entities.tolist(),
                    "selected_triplets": selected_triplets,
                    "scores": selected_scores,
                    "reward": reward
                }
                
                # Validate result structure (Requirement 6.4)
                self.validate_result_structure(result)
                
                all_results.append(result)
        
        # Save all results to JSON file
        output_file = os.path.join(output_dir, "inference_results.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Saved {len(all_results)} inference results to {output_file}")
        
        return all_results
    
    def select_top_k_triplets(
        self,
        scores: torch.Tensor,
        triplets: List[Tuple[str, str, str]],
        k: int
    ) -> Tuple[List[Tuple[str, str, str]], List[float]]:
        """
        Select k triplets with highest scores.
        
        Args:
            scores: Tensor of ranking scores for each triplet
            triplets: List of triplet tuples (subject, relation, object)
            k: Number of triplets to select
            
        Returns:
            selected_triplets: List of k selected triplet tuples
            selected_scores: List of k corresponding scores
        """
        # Handle case where k is larger than available triplets
        k = min(k, len(triplets))
        
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(scores, k)
        
        # Convert to lists
        top_k_indices = top_k_indices.cpu().tolist()
        top_k_scores = top_k_values.cpu().tolist()
        
        # Select triplets
        selected_triplets = [triplets[idx] for idx in top_k_indices]
        
        return selected_triplets, top_k_scores
