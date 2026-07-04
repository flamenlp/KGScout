"""
Triplet selection methods for KGscout and cosine retriever.

This module provides functions for selecting top-k triplets using different retrieval methods:
- KGscout: Uses trained PathRankingModel (or ablation variant) to rank triplets
- Cosine: Uses pre-computed cosine similarity scores from dataset

Supports both raw dataset dicts and DataLoader-batched format (batch_size=1, default collate).
The DataLoader format is the standard pattern used throughout the project:
    DataLoader(SampledJointTrainingDataset(...), batch_size=1, shuffle=False)
"""

import json
import os
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


def _extract_triplets_from_batch(batch: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Extract structured triplets from a DataLoader-batched sample.

    In DataLoader(batch_size=1) with default collate, topk_rel_data becomes
    a collated nested structure where each element is:
        (score_tensor, (subject_tuple, relation_tuple, object_tuple))
    with each string wrapped as a 1-tuple due to batch collation.

    Args:
        batch: Single batch from DataLoader(batch_size=1)

    Returns:
        List of (subject, relation, object) string tuples
    """
    triplets = []
    for item in batch["topk_rel_data"]:
        # item = (score, (subj_tuple, rel_tuple, obj_tuple))
        # Each string is wrapped in a tuple of length 1 by default collate
        s = item[1][0][0]
        r = item[1][1][0]
        o = item[1][2][0]
        triplets.append((s, r, o))
    return triplets


def _is_dataloader_batch(data_sample: Dict[str, Any]) -> bool:
    """
    Detect whether data_sample is from a DataLoader (default collate, batch_size=1)
    or a raw dataset dict.

    Heuristic: In DataLoader-batched format, question_embedding has an extra batch dim
    (dim >= 2 with shape[0] == 1) or 'question' is a list of strings.
    """
    # Check if question is wrapped in a list (DataLoader collates strings into lists)
    if isinstance(data_sample.get("question"), (list, tuple)):
        return True
    return False


def select_triplets_kgscout(
    model: nn.Module,
    data_sample: Dict[str, Any],
    k: int,
    device: str = "cuda"
) -> List[Tuple[str, str, str]]:
    """
    Select top-k triplets using a trained model (PathRankingModel or ablation variant).

    Supports both:
    - Raw dataset dict (from JointTrainingDatasetv3PPR.__getitem__)
    - DataLoader-batched dict (from DataLoader(batch_size=1) with default collate)

    Logic:
    1. Forward pass → ranking_scores, path_probs
    2. Deterministic top-k selection via torch.topk(ranking_scores, k)

    Args:
        model: Trained model with forward(question_embed, triplet_embeds, relation_embeds, graph_features)
        data_sample: Dataset sample or DataLoader batch (batch_size=1)
        k: Number of top triplets to select
        device: Device to run model on

    Returns:
        List of (subject, relation, object) tuples sorted by ranking score (descending)
    """
    is_batch = _is_dataloader_batch(data_sample)

    # Extract tensors (handle batch dim from DataLoader)
    question_embed = data_sample["question_embedding"].to(device)
    triplet_embeds = data_sample["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
    relation_embeds = data_sample["topK_rel_embeddings"].squeeze(0).to(device)
    graph_features = data_sample["graph_features"].squeeze(0).to(device)

    # Extract triplets based on format
    if is_batch:
        triplets = _extract_triplets_from_batch(data_sample)
    else:
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

    return selected_triplets


def select_triplets_cosine(
    data_sample: Dict[str, Any],
    k: int
) -> List[Tuple[str, str, str]]:
    """
    Select top-k triplets using cosine similarity (from dataset).

    Since data in the dataset is already sorted by cosine similarity (descending),
    this simply takes the first k triplets.

    Supports both raw dataset dict and DataLoader-batched format.

    Args:
        data_sample: Dataset sample or DataLoader batch (batch_size=1)
        k: Number of top triplets to select

    Returns:
        List of (subject, relation, object) tuples
    """
    is_batch = _is_dataloader_batch(data_sample)

    # Extract triplets based on format
    if is_batch:
        triplets = _extract_triplets_from_batch(data_sample)
    else:
        triplets = [triplet for _, triplet in data_sample["topk_rel_data"]]

    # Handle empty triplets
    if len(triplets) == 0:
        return []

    # Select top-k triplets (already sorted by cosine similarity)
    k = min(k, len(triplets))
    selected_triplets = triplets[:k]

    return selected_triplets


def _extract_metadata_from_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract question, answer, entity metadata from a DataLoader-batched sample.

    Args:
        batch: Single batch from DataLoader(batch_size=1)

    Returns:
        Dict with question, answer, a_entity, q_entity as plain strings/lists
    """
    question = batch["question"][0] if isinstance(batch["question"], (list, tuple)) else batch["question"]
    answer = [p[0] for p in batch["answer"]] if isinstance(batch["answer"], (list, tuple)) and len(batch["answer"]) > 0 and isinstance(batch["answer"][0], (list, tuple)) else batch["answer"]
    a_entity = [p[0] for p in batch["a_entity"]] if isinstance(batch["a_entity"], (list, tuple)) and len(batch["a_entity"]) > 0 and isinstance(batch["a_entity"][0], (list, tuple)) else batch["a_entity"]
    q_entity = [p[0] for p in batch["q_entity"]] if isinstance(batch["q_entity"], (list, tuple)) and len(batch["q_entity"]) > 0 and isinstance(batch["q_entity"][0], (list, tuple)) else batch["q_entity"]

    return {
        "question": question,
        "answer": answer,
        "a_entity": a_entity,
        "q_entity": q_entity,
    }


def format_relation(rel: str) -> str:
    """Convert 'award.award_nomination.award_nominee' to 'award award nomination award nominee'."""
    return rel.replace('.', ' ').replace('_', ' ')


def generate_selected_json(
    dataloader: DataLoader,
    model: nn.Module,
    output_dir: str,
    k: int,
    retriever_type: str = "kgscout",
    device: str = "cuda"
) -> str:
    """
    Generate selected triplets JSON file from a DataLoader.

    Iterates over a DataLoader(batch_size=1), runs model inference to select top-k
    triplets per question, and saves results in the standard format used by
    LLM inference scripts.

    Args:
        dataloader: DataLoader(batch_size=1) over SampledJointTrainingDataset
        model: Trained model (PathRankingModel or ablation variant). Required if retriever_type='kgscout'.
        output_dir: Directory to save selected_triplets.json
        k: Number of top triplets to select per question
        retriever_type: 'kgscout' or 'cosine'
        device: Device to run model on

    Returns:
        Path to saved selected_triplets.json file
    """
    os.makedirs(output_dir, exist_ok=True)

    selected_data = []
    errors = []

    if model is not None:
        model.eval()

    for idx, batch in enumerate(tqdm(dataloader, desc=f"Selecting triplets ({retriever_type})")):
        try:
            # Select triplets based on retriever type
            if retriever_type == "kgscout":
                selected_triplets = select_triplets_kgscout(model, batch, k, device)
            elif retriever_type == "cosine":
                selected_triplets = select_triplets_cosine(batch, k)
            else:
                raise ValueError(f"Invalid retriever_type: {retriever_type}")

            # Extract metadata from batch
            meta = _extract_metadata_from_batch(batch)

            # Format as linearized strings for LLM inference
            linearized_triplets = [
                f"{s}, {format_relation(r)}, {o}" for s, r, o in selected_triplets
            ]

            # Create output entry following lama-inference.py format
            entry = {
                "question": meta["question"],
                "answer": meta["answer"],
                "a_entity": meta["a_entity"],
                "q_entity": meta["q_entity"],
                "reranker": linearized_triplets,
            }

            selected_data.append(entry)

        except Exception as e:
            error_msg = f"Question ID {idx}: Failed to select triplets. Error: {str(e)}"
            errors.append(error_msg)
            print(f"Warning: {error_msg}")

            # Add empty entry to maintain alignment
            try:
                meta = _extract_metadata_from_batch(batch)
            except Exception:
                meta = {"question": "", "answer": [], "a_entity": [], "q_entity": []}

            entry = {
                "question": meta["question"],
                "answer": meta["answer"],
                "a_entity": meta["a_entity"],
                "q_entity": meta["q_entity"],
                "reranker": [],
            }
            selected_data.append(entry)

    # Save to JSON file
    output_path = os.path.join(output_dir, "selected_triplets.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)

    # Report errors if any
    if errors:
        print(f"\nWarning: {len(errors)} errors occurred during triplet selection:")
        for error in errors[:5]:
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
    from torch.utils.data import DataLoader as _DataLoader

    from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
    from src.preprocess.sampled_dataset import SampledJointTrainingDataset
    from src.model.path_ranker import PathRankingModel

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
    parser.add_argument("--sample-k", type=int, default=1000,
                        help="Number of triplets to feed to model per sample (default: 1000)")
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
    print(f"  Sample-k:  {args.sample_k}")
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
            for key, val in ckpt.items():
                if key in ('temperature', 'baseline'):
                    getattr(model, key).data = val.to(device)
                elif hasattr(model, key):
                    getattr(model, key).load_state_dict(val)
        model.to(device)
        model.eval()
        print("  Model loaded.")

    # Load test data and create DataLoader
    print("Loading test data...")
    test_data = _torch.load(args.test_data, weights_only=False, map_location="cpu")
    print(f"  Test samples: {len(test_data)}")

    # Wrap in SampledJointTrainingDataset + DataLoader (consistent with ablation-2 pattern)
    test_dataset = SampledJointTrainingDataset(test_data, k=args.sample_k)
    test_dataloader = _DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate selected_triplets.json
    output_path = generate_selected_json(
        dataloader=test_dataloader,
        model=model,
        output_dir=args.output_dir,
        k=args.top_k,
        retriever_type=args.retriever,
        device=device,
    )

    print(f"\nDone. Output: {output_path}")
