#!/usr/bin/env python3
"""
Preprocess MetaQA dataset into KGScout-compatible format.

Takes the raw MetaQA files (kb.txt + qa_test.txt) and produces a .pt file
with the same structure KGScout expects:
  - question, q_entity, a_entity, answer
  - question_embedding (384-dim, sentence-transformers)
  - topk_linearized_triplets (cosine-sorted triplet strings)
  - topk_linearized_triplet_embeddings
  - topk_rel_data: List[(score, (s, r, o))]
  - topK_rel_embeddings (relation embeddings)
  - is_empty

Pipeline:
1. Load KG from kb.txt → build networkx graph
2. Load qa_test.txt → extract topic entities and answers
3. For each question: BFS from topic entity to get candidate subgraph
4. Compute embeddings (question + triplets + relations) using sentence-transformers
5. Rank triplets by cosine similarity to question → top-N
6. Save as .pt file
"""

import os
import sys
import re
import argparse
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_kg(kb_path: str) -> Tuple[nx.DiGraph, Dict[str, List[Tuple[str, str, str]]]]:
    """
    Load MetaQA knowledge graph from kb.txt.
    Format: subject|relation|object (one per line)

    Returns:
        graph: NetworkX directed graph
        entity_to_triplets: mapping from entity (lowercase) to list of triplets involving it
    """
    graph = nx.DiGraph()
    entity_to_triplets = defaultdict(list)

    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            graph.add_edge(s.lower(), o.lower(), relation=r)
            triplet = (s, r, o)
            entity_to_triplets[s.lower()].append(triplet)
            entity_to_triplets[o.lower()].append(triplet)

    print(f"KG loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, entity_to_triplets


def load_qa_file(qa_path: str) -> List[Dict]:
    """
    Load MetaQA QA file.
    Format: question[TAB]answer1|answer2|...
    Topic entity is in [brackets] within the question text.
    """
    samples = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            question = parts[0].strip()
            answers = [a.strip() for a in parts[1].split('|') if a.strip()]

            # Extract topic entity from [brackets]
            entity_match = re.findall(r'\[([^\]]+)\]', question)
            q_entities = entity_match if entity_match else []

            # Clean question text (remove brackets for embedding)
            clean_question = re.sub(r'\[([^\]]+)\]', r'\1', question)

            samples.append({
                "question": clean_question,
                "question_raw": question,
                "q_entity": q_entities,
                "a_entity": answers,  # In MetaQA, answers ARE the answer entities
                "answer": answers,
            })

    print(f"QA file loaded: {len(samples)} questions")
    return samples


def extract_subgraph_triplets(
    graph: nx.DiGraph,
    entity_to_triplets: Dict,
    topic_entities: List[str],
    hop: int,
    max_triplets: int = 1000
) -> List[Tuple[str, str, str]]:
    """
    Extract candidate triplets via BFS from topic entities up to `hop` hops.
    Returns deduplicated triplets (up to max_triplets).
    """
    visited_nodes = set()
    frontier = set()

    for entity in topic_entities:
        e_lower = entity.lower()
        if e_lower in graph:
            frontier.add(e_lower)

    # BFS up to `hop` levels
    for _ in range(hop):
        next_frontier = set()
        for node in frontier:
            visited_nodes.add(node)
            # Get successors and predecessors
            for neighbor in graph.successors(node):
                if neighbor not in visited_nodes:
                    next_frontier.add(neighbor)
            for neighbor in graph.predecessors(node):
                if neighbor not in visited_nodes:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    visited_nodes.update(frontier)

    # Collect all triplets that involve visited nodes
    # Prioritize topic entity's direct triplets first to ensure they're not cut off
    seen_triplets = set()
    triplets = []

    # First: add triplets directly involving topic entities
    for entity in topic_entities:
        e_lower = entity.lower()
        for triplet in entity_to_triplets.get(e_lower, []):
            triplet_key = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            if triplet_key not in seen_triplets:
                seen_triplets.add(triplet_key)
                triplets.append(triplet)

    # Then: add remaining triplets from other visited nodes
    for node in visited_nodes:
        for triplet in entity_to_triplets.get(node, []):
            triplet_key = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            if triplet_key not in seen_triplets:
                seen_triplets.add(triplet_key)
                triplets.append(triplet)

    # Limit to max_triplets
    return triplets[:max_triplets]


def compute_embeddings(
    texts: List[str],
    model,
    batch_size: int = 256
) -> torch.Tensor:
    """Compute sentence embeddings in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def preprocess_metaqa(
    kb_path: str,
    qa_path: str,
    output_dir: str,
    hop: int,
    max_triplets: int = 1000,
    embedding_model_name: str = "all-MiniLM-L6-v2"
):
    """
    Main preprocessing pipeline.
    """
    from sentence_transformers import SentenceTransformer

    print(f"=" * 60)
    print(f"PREPROCESSING MetaQA {hop}-hop")
    print(f"=" * 60)
    print(f"KB: {kb_path}")
    print(f"QA: {qa_path}")
    print(f"Output: {output_dir}")
    print(f"Max triplets per question: {max_triplets}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"=" * 60)

    # Load embedding model
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(embedding_model_name)

    # Load KG
    print("\nLoading knowledge graph...")
    graph, entity_to_triplets = load_kg(kb_path)

    # Load QA data
    print("\nLoading QA data...")
    samples = load_qa_file(qa_path)

    # Process each sample
    print(f"\nProcessing {len(samples)} samples...")
    processed_data = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"Processing {hop}-hop")):
        topic_entities = sample["q_entity"]

        # Extract subgraph triplets
        triplets = extract_subgraph_triplets(
            graph, entity_to_triplets, topic_entities, hop, max_triplets
        )

        if len(triplets) == 0:
            skipped += 1
            continue

        # Linearize triplets: "subject relation object" (replace underscores in relations)
        linearized = [
            f"{s} {r.replace('_', ' ')} {o}" for s, r, o in triplets
        ]

        # Compute question embedding
        question_embedding = embed_model.encode(
            sample["question"], convert_to_tensor=True
        ).cpu()

        # Compute triplet embeddings
        triplet_embeddings = compute_embeddings(linearized, embed_model)

        # Compute relation embeddings (just the relation text)
        relations = [r for _, r, _ in triplets]
        relation_embeddings = compute_embeddings(relations, embed_model)

        # Compute cosine similarity scores for ranking
        q_emb_norm = question_embedding / (question_embedding.norm() + 1e-10)
        t_emb_norm = triplet_embeddings / (triplet_embeddings.norm(dim=1, keepdim=True) + 1e-10)
        cosine_scores = torch.matmul(t_emb_norm, q_emb_norm).squeeze()

        # Sort by cosine similarity (descending)
        sorted_indices = torch.argsort(cosine_scores, descending=True)

        # Reorder everything by cosine score
        sorted_triplets = [triplets[i] for i in sorted_indices.tolist()]
        sorted_linearized = [linearized[i] for i in sorted_indices.tolist()]
        sorted_triplet_embeddings = triplet_embeddings[sorted_indices]
        sorted_relation_embeddings = relation_embeddings[sorted_indices]
        sorted_scores = cosine_scores[sorted_indices]

        # Build topk_rel_data: List[(score, (s, r, o))]
        topk_rel_data = [
            (sorted_scores[i].item(), sorted_triplets[i])
            for i in range(len(sorted_triplets))
        ]

        processed_entry = {
            "question": sample["question"],
            "q_entity": sample["q_entity"],
            "a_entity": sample["a_entity"],
            "answer": sample["answer"],
            "question_embedding": question_embedding,
            "topk_linearized_triplets": sorted_linearized,
            "topk_linearized_triplet_embeddings": sorted_triplet_embeddings,
            "topk_rel_data": topk_rel_data,
            "topK_rel_embeddings": sorted_relation_embeddings,
            "is_empty": False,
        }
        processed_data.append(processed_entry)

    # Save as .pt file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metaqa-{hop}hop-test.pt")
    torch.save(processed_data, output_path)

    num_processed = len(processed_data)

    # Clear memory
    del processed_data, graph, entity_to_triplets, samples, embed_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {num_processed} samples")
    print(f"Skipped (no triplets): {skipped}")
    print(f"Saved to: {output_path}")
    print(f"Memory cleared.")
    print(f"{'=' * 60}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MetaQA into KGScout format")
    parser.add_argument("--kb-path", type=str, required=True,
                        help="Path to MetaQA kb.txt file")
    parser.add_argument("--qa-path", type=str, required=True,
                        help="Path to MetaQA qa_test.txt file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed .pt file")
    parser.add_argument("--hop", type=int, required=True, choices=[1, 2, 3],
                        help="Number of hops (1, 2, or 3)")
    parser.add_argument("--max-triplets", type=int, default=1000,
                        help="Maximum candidate triplets per question (default: 1000)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    preprocess_metaqa(
        kb_path=args.kb_path,
        qa_path=args.qa_path,
        output_dir=args.output_dir,
        hop=args.hop,
        max_triplets=args.max_triplets,
        embedding_model_name=args.embedding_model
    )
