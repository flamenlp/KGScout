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

from training.rewards import compute_reward_v8


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
        
        Answer coverage is the fraction of answer entities that appear
        in the selected triplets (either as subject or object).
        
        Args:
            selected_triplets: List of (subject, relation, object) tuples
            a_entities: List of answer entity strings
        
        Returns:
            Answer coverage score in range [0.0, 1.0]
        
        Requirements:
            - 5.2: Compute answer coverage metrics during evaluation
        """
        if not a_entities:
            return 0.0
        
        # Build set of entities in selected triplets (lowercase for matching)
        entities_in_triplets = set()
        for s, r, o in selected_triplets:
            entities_in_triplets.add(s.lower())
            entities_in_triplets.add(o.lower())
        
        # Count how many answer entities are present
        present_count = sum(
            1 for a in a_entities
            if a.lower() in entities_in_triplets
        )
        
        # Return fraction of answer entities present
        return present_count / len(a_entities)
    
    def _compute_path_coverage(
        self,
        selected_triplets,
        q_entities,
        a_entities
    ) -> float:
        """
        Compute path coverage metric.
        
        Path coverage is the fraction of shortest path edges (between question
        and answer entities) that are covered by the selected triplets.
        
        Args:
            selected_triplets: List of (subject, relation, object) tuples
            q_entities: List of question entity strings
            a_entities: List of answer entity strings
        
        Returns:
            Path coverage score in range [0.0, 1.0]
        
        Requirements:
            - 5.3: Compute path coverage metrics during evaluation
        """
        if not selected_triplets or not q_entities or not a_entities:
            return 0.0
        
        # Build directed graph from selected triplets
        G = nx.DiGraph()
        for s, r, o in selected_triplets:
            G.add_edge(s.lower(), o.lower(), relation=r.lower())
        
        # Build set of edges in selected triplets
        triplet_edges = {
            (s.lower(), o.lower())
            for s, r, o in selected_triplets
        }
        
        # Compute path coverage for each Q-A pair
        coverage_scores = []
        for q in q_entities:
            for a in a_entities:
                qn, an = q.lower(), a.lower()
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(G, qn, an)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path exists
                    continue
                
                if len(path) < 2:
                    # Path too short (no edges)
                    continue
                
                # Count how many edges in the path are in our selected triplets
                path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
                matches = sum(1 for edge in path_edges if edge in triplet_edges)
                
                # Compute coverage for this path
                coverage = matches / len(path_edges)
                coverage_scores.append(coverage)
        
        # Return maximum coverage across all Q-A pairs
        # (same as compute_reward_v8 implementation)
        return max(coverage_scores) if coverage_scores else 0.0
