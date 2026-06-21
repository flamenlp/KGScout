"""
Triplet selection methods for KGscout and cosine retriever.

This module provides functions for selecting top-k triplets using different retrieval methods:
- KGscout: Uses trained PathRankingModel to rank triplets
- Cosine: Uses pre-computed cosine similarity scores from dataset
"""

import json
import os
import torch
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from src.model.path_ranker import PathRankingModel


def select_triplets_kgscout(
    model: PathRankingModel,
    data_sample: Dict[str, Any],
    k: int,
    device: str = "cuda"
) -> List[Tuple[str, str, str]]:
    """
    Select top-k triplets using KGscout model.
    
    Logic (matches notebook Architecture-v8):
    1. Forward pass → ranking_scores, path_probs
    2. sample_paths(path_probs, triplets, k, ranking_scores) → stochastic selection
    3. Sort selected paths by selected_probs descending
    
    Args:
        model: Trained PathRankingModel
        data_sample: Single dataset sample with embeddings and triplets
        k: Number of top triplets to select
        device: Device to run model on
    
    Returns:
        List of (subject, relation, object) tuples sorted by probability
    
    Requirements:
        - 10.1: Use the generate_selected_json method to select triplets when retriever-type is "kgscout"
        - 10.4: Validate that selected triplets contain required fields (subject, relation, object)
    """
    # Extract required data from sample
    question_embed = data_sample["question_embedding"].to(device)
    triplet_embeds = data_sample["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
    relation_embeds = data_sample["topK_rel_embeddings"].squeeze(0).to(device)
    graph_features = data_sample["graph_features"].to(device)
    
    # Get triplets from topk_rel_data
    triplets = [triplet for _, triplet in data_sample["topk_rel_data"]]
    
    # Handle empty triplets
    if len(triplets) == 0:
        return []
    
    k = min(k, len(triplets))
    
    # Forward pass to get ranking scores and probabilities
    with torch.no_grad():
        ranking_scores, path_probs = model(
            question_embed,
            triplet_embeds,
            relation_embeds,
            graph_features
        )
    
    # Step 2: sample_paths (stochastic selection, same as notebook)
    selected_triplets, selected_probs, _, _ = model.sample_paths(
        path_probs, triplets, k, ranking_scores
    )
    
    # Step 3: Sort selected paths by probability (descending)
    sorted_indices = torch.argsort(selected_probs, descending=True)
    selected_triplets = [selected_triplets[i] for i in sorted_indices.tolist()]
    
    # Validate triplet format
    for triplet in selected_triplets:
        if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
            raise ValueError(
                f"Invalid triplet format: {triplet}. "
                f"Expected 3-tuple (subject, relation, object)."
            )
    
    return selected_triplets


def select_triplets_cosine(
    data_sample: Dict[str, Any],
    k: int
) -> List[Tuple[str, str, str]]:
    """
    Select top-k triplets using cosine similarity (from dataset).
    
    Args:
        data_sample: Single dataset sample with topk_rel_data
        k: Number of top triplets to select
    
    Returns:
        List of (subject, relation, object) tuples
    
    Requirements:
        - 10.2: Extract triplets from the topk_linearized_triplets dataset field when retriever-type is "cosine"
        - 10.3: Ensure both retriever methods produce Selected_JSON in the same format
        - 10.4: Validate that selected triplets contain required fields (subject, relation, object)
    """
    # Get triplets from topk_rel_data (already sorted by cosine similarity)
    triplets = [triplet for _, triplet in data_sample["topk_rel_data"]]
    
    # Handle empty triplets
    if len(triplets) == 0:
        return []
    
    # Select top-k triplets
    k = min(k, len(triplets))
    selected_triplets = triplets[:k]
    
    # Validate triplet format
    for triplet in selected_triplets:
        if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
            raise ValueError(
                f"Invalid triplet format: {triplet}. "
                f"Expected 3-tuple (subject, relation, object)."
            )
    
    return selected_triplets


def generate_selected_json(
    data: List[Dict[str, Any]],
    model: PathRankingModel,
    output_dir: str,
    k: int,
    retriever_type: str = "kgscout",
    device: str = "cuda"
) -> str:
    """
    Generate selected triplets JSON file for all samples.
    
    This function creates a JSON file containing the top-k selected triplets
    for each question in the dataset, following the format used by lama-inference.py.
    
    Args:
        data: List of dataset samples
        model: Trained PathRankingModel (required if retriever_type='kgscout')
        output_dir: Directory to save selected triplets JSON
        k: Number of top triplets to select per question
        retriever_type: 'kgscout' or 'cosine'
        device: Device to run model on
    
    Returns:
        Path to saved selected_triplets.json file
    
    Requirements:
        - 10.1: Use the generate_selected_json method to select triplets when retriever-type is "kgscout"
        - 10.2: Extract triplets from the topk_linearized_triplets dataset field when retriever-type is "cosine"
        - 10.3: Ensure both retriever methods produce Selected_JSON in the same format
        - 10.5: Log the error with question ID and continue processing when triplet selection fails
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output data
    selected_data = []
    errors = []
    
    for idx, sample in enumerate(tqdm(data, desc=f"Selecting triplets ({retriever_type})")):
        try:
            # Select triplets based on retriever type
            if retriever_type == "kgscout":
                selected_triplets = select_triplets_kgscout(model, sample, k, device)
            elif retriever_type == "cosine":
                selected_triplets = select_triplets_cosine(sample, k)
            else:
                raise ValueError(f"Invalid retriever_type: {retriever_type}")
            
            # Get scores for selected triplets
            if retriever_type == "kgscout":
                # Use model scores
                question_embed = sample["question_embedding"].to(device)
                triplet_embeds = sample["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
                relation_embeds = sample["topK_rel_embeddings"].squeeze(0).to(device)
                graph_features = sample["graph_features"].to(device)
                
                with torch.no_grad():
                    ranking_scores, _ = model(
                        question_embed,
                        triplet_embeds,
                        relation_embeds,
                        graph_features
                    )
                
                # Get scores for selected triplets
                all_triplets = [triplet for _, triplet in sample["topk_rel_data"]]
                triplet_to_score = {
                    triplet: ranking_scores.squeeze()[i].item()
                    for i, triplet in enumerate(all_triplets)
                }
                scores = [triplet_to_score.get(triplet, 0.0) for triplet in selected_triplets]
            else:
                # Use cosine scores from dataset
                triplet_to_score = {triplet: score for score, triplet in sample["topk_rel_data"]}
                scores = [triplet_to_score.get(triplet, 0.0) for triplet in selected_triplets]
            
            # Format as linearized strings for LLM inference
            linearized_triplets = [
                f"{s}, {r}, {o}" for s, r, o in selected_triplets
            ]
            
            # Create output entry following lama-inference.py format
            entry = {
                "question": sample["question"],
                "answer": sample["answer"],
                "a_entity": sample["a_entity"],
                "reranker": linearized_triplets  # List of linearized triplet strings
            }
            
            selected_data.append(entry)
            
        except Exception as e:
            # Log error and continue processing
            error_msg = f"Question ID {idx}: Failed to select triplets. Error: {str(e)}"
            errors.append(error_msg)
            print(f"Warning: {error_msg}")
            
            # Add empty entry to maintain alignment
            entry = {
                "question": sample.get("question", ""),
                "answer": sample.get("answer", []),
                "a_entity": sample.get("a_entity", []),
                "reranker": []
            }
            selected_data.append(entry)
    
    # Save to JSON file
    output_path = os.path.join(output_dir, "selected_triplets.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)
    
    # Report errors if any
    if errors:
        print(f"\nWarning: {len(errors)} errors occurred during triplet selection:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return output_path
