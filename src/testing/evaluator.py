"""
Evaluation metrics for model performance assessment.

This module implements the Evaluator class which computes evaluation metrics
for knowledge graph question answering models.
"""

from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx

from src.training.rewards import compute_reward_v8


class Evaluator:
    """
    Computes evaluation metrics for model performance.
    
    Requirements:
        - 5.1: Organize evaluate_answer_and_path_coverage method in testing/ directory
        - 5.2: Compute answer coverage metrics during evaluation
        - 5.3: Compute path coverage metrics during evaluation
        - 5.4: Support evaluation on test datasets
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            device: Device to use for evaluation (default: "cuda")
        """
        self.device = device
    
    def evaluate_answer_and_path_coverage(
        self,
        test_dataloader: DataLoader,
        trainer,
        top_k: int
    ) -> Dict[str, float]:
        """
        Evaluate answer coverage and path coverage.
        
        This method computes three key metrics:
        1. Answer coverage: Fraction of answer entities present in selected triplets
        2. Path coverage: Fraction of shortest path edges covered by selected triplets
        3. Average reward: Mean reward across test samples using compute_reward_v8
        
        Args:
            test_dataloader: DataLoader for test dataset
            trainer: Trainer instance with trained model
            top_k: Number of top triplets to select for evaluation
        
        Returns:
            Dictionary containing:
                - answer_coverage: Average answer coverage across test samples
                - path_coverage: Average path coverage across test samples
                - average_reward: Average reward across test samples
        
        Requirements:
            - 5.1: Organize evaluate_answer_and_path_coverage method in testing/ directory
            - 5.2: Compute answer coverage metrics during evaluation
            - 5.3: Compute path coverage metrics during evaluation
            - 5.4: Support evaluation on test datasets
        """
        # Set model to evaluation mode
        trainer.path_ranker.eval()
        
        # Initialize metric accumulators
        total_answer_coverage = 0.0
        total_path_coverage = 0.0
        total_reward = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                # Skip None batches (empty samples)
                if batch is None:
                    continue
                
                # Move batch to device
                question_embed = batch["question_embedding"].to(self.device)
                triplet_embeds = batch["topk_linearized_triplet_embeddings"].to(self.device)
                relation_embeds = batch["topK_rel_embeddings"].to(self.device)
                graph_scores = batch["graph_features"].to(self.device)
                
                # Get paths and triplets
                paths = batch["topk_linearized_triplets"]
                triplets = [triplet for _, triplet in batch["topk_rel_data"]]
                q_entities = batch["q_entity"]
                a_entities = batch["a_entity"]
                
                # Forward pass to get ranking scores
                ranking_scores, path_probs = trainer.path_ranker(
                    question_embed,
                    triplet_embeds,
                    relation_embeds,
                    graph_scores
                )
                
                # Select top-k triplets based on ranking scores
                k = min(top_k, len(triplets))
                top_k_indices = torch.topk(ranking_scores.squeeze(), k).indices.cpu().tolist()
                selected_triplets = [triplets[i] for i in top_k_indices]
                
                # Compute answer coverage
                answer_coverage = self._compute_answer_coverage(
                    selected_triplets,
                    a_entities
                )
                
                # Compute path coverage
                path_coverage = self._compute_path_coverage(
                    selected_triplets,
                    q_entities,
                    a_entities
                )
                
                # Compute reward
                reward = compute_reward_v8(
                    triplets=selected_triplets,
                    q_entities=q_entities,
                    a_entities=a_entities
                )
                
                # Accumulate metrics
                total_answer_coverage += answer_coverage
                total_path_coverage += path_coverage
                total_reward += reward
                num_samples += 1
        
        # Compute average metrics
        if num_samples == 0:
            return {
                "answer_coverage": 0.0,
                "path_coverage": 0.0,
                "average_reward": 0.0
            }
        
        return {
            "answer_coverage": total_answer_coverage / num_samples,
            "path_coverage": total_path_coverage / num_samples,
            "average_reward": total_reward / num_samples
        }
    
    def _compute_answer_coverage(
        self,
        selected_triplets,
        a_entities
    ) -> float:
        """
        Compute answer coverage metric.
        
        Answer coverage returns 1.0 if any answer entity appears in
        the selected triplets (either as subject or object), 0.0 otherwise.
        
        Args:
            selected_triplets: List of (subject, relation, object) tuples
            a_entities: List of answer entity strings
        
        Returns:
            1.0 if any answer entity is present, 0.0 otherwise
        
        Requirements:
            - 5.2: Compute answer coverage metrics during evaluation
        """
        if not a_entities:
            return 0.0
        
        # Check if any answer entity is present in any triplet
        for a in a_entities:
            a_lower = a.lower()
            for s, r, o in selected_triplets:
                if a_lower == s.lower() or a_lower == o.lower():
                    return 1.0
        
        return 0.0
    
    def _compute_path_coverage(
        self,
        selected_triplets,
        q_entities,
        a_entities
    ) -> float:
        """
        Compute path coverage metric.
        
        Path coverage returns 1.0 if a path exists between any question entity
        and any answer entity on an undirected graph built from selected triplets
        (matches notebook implementation).
        
        Args:
            selected_triplets: List of (subject, relation, object) tuples
            q_entities: List of question entity strings
            a_entities: List of answer entity strings
        
        Returns:
            1.0 if a reasoning path exists, 0.0 otherwise
        
        Requirements:
            - 5.3: Compute path coverage metrics during evaluation
        """
        if not selected_triplets or not q_entities or not a_entities:
            return 0.0
        
        # Build undirected graph from selected triplets (matches notebook)
        G = nx.Graph()
        for s, r, o in selected_triplets:
            G.add_edge(s.lower(), o.lower(), relation=r.lower())
        
        # Check if path exists between any question entity and any answer entity
        for q in q_entities:
            for a in a_entities:
                qn, an = q.lower(), a.lower()
                if qn not in G or an not in G:
                    continue
                try:
                    if nx.has_path(G, qn, an):
                        return 1.0
                except nx.NetworkXError:
                    continue
        
        return 0.0
