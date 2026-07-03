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
    
    Logic:
    1. Forward pass → ranking_scores, path_probs
    2. Deterministic top-k selection via torch.topk(ranking_scores, k)
    
    Args:
        model: Trained PathRankingModel
        data_sample: Single dataset sample with embeddings and triplets
        k: Number of top triplets to select
        device: Device to run model on
    
    Returns:
        List of (subject, relation, object) tuples sorted by ranking score (descending)
    
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
    
    # Deterministic top-k selection (no sampling at inference)
    top_k_scores, top_k_indices = torch.topk(ranking_scores, k)
    selected_triplets = [triplets[i] for i in top_k_indices.tolist()]
    
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


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Generate selected_triplets.json from a trained model checkpoint.

    Usage:
        python -m src.utils.triplet_selector \
            --model-path results/k-ablation/k100/model/main_training_k100/best_model_k100.pt \
            --test-data /path/to/test.pt \
            --output-dir results/k-ablation/k100/triplet-analysis/ \
            --top-k 100

        # Or with cosine retriever (no model needed):
        python -m src.utils.triplet_selector \
            --test-data /path/to/test.pt \
            --output-dir results/cosine/triplet-analysis/ \
            --top-k 100 \
            --retriever cosine
    """
    import sys
    import argparse
    import torch as _torch

    from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR

    # Allow loading datasets saved from notebooks
    import __main__ as _main
    _main.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

    parser = argparse.ArgumentParser(
        description="Generate selected_triplets.json from a trained model"
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model checkpoint (.pt file). Required for kgscout retriever.")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test dataset .pt file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for selected_triplets.json")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of triplets to select per sample (default: 100)")
    parser.add_argument("--retriever", type=str, default="kgscout",
                        choices=["kgscout", "cosine"],
                        help="Retriever type (default: kgscout)")

    args = parser.parse_args()

    # Validate
    if args.retriever == "kgscout" and args.model_path is None:
        parser.error("--model-path is required for kgscout retriever")

    if args.model_path and not os.path.exists(args.model_path):
        print(f"ERROR: Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if _torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("GENERATE SELECTED TRIPLETS")
    print(f"  Model:     {args.model_path}")
    print(f"  Test data: {args.test_data}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Top-k:     {args.top_k}")
    print(f"  Retriever: {args.retriever}")
    print(f"  Device:    {device}")
    print("=" * 70)

    # Load model if kgscout
    model = None
    if args.retriever == "kgscout":
        print("Loading model...")
        model = PathRankingModel(hidden_size=384, device=device)
        ckpt = _torch.load(args.model_path, weights_only=False, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # save_pretrained format (component-level state dicts)
            model.question_triplet_attention.load_state_dict(ckpt['question_triplet_attention'])
            model.question_relation_attention.load_state_dict(ckpt['question_relation_attention'])
            model.gate_network.load_state_dict(ckpt['gate_network'])
            model.triplet_mlp.load_state_dict(ckpt['triplet_mlp'])
            model.relation_mlp.load_state_dict(ckpt['relation_mlp'])
            model.combiner_mlp.load_state_dict(ckpt['combiner_mlp'])
            model.temperature.data = ckpt['temperature'].to(device)
            model.baseline.data = ckpt['baseline'].to(device)
        model.to(device)
        model.eval()
        print("  Model loaded.")

    # Load test data
    print("Loading test data...")
    test_data = _torch.load(args.test_data, weights_only=False, map_location="cpu")
    print(f"  Test samples: {len(test_data)}")

    # Generate selected_triplets.json
    output_path = generate_selected_json(
        data=test_data,
        model=model,
        output_dir=args.output_dir,
        k=args.top_k,
        retriever_type=args.retriever,
        device=device,
    )

    print(f"\nDone. Output: {output_path}")
