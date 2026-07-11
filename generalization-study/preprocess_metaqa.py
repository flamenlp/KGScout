#!/usr/bin/env python3
"""
Preprocess MetaQA dataset into KGScout JointTrainer-compatible format.

This script curates training/validation/test subsets from MetaQA with
specific distribution constraints for effective model training:

Training (4000 samples) & Validation (500 samples):
  - Condition 1 (MUST): ≥75% samples have a path between q_entity and a_entity
    in the top-1000 cosine-ranked triplets.
  - Condition 2 (GOOD TO HAVE): ≥10% samples have NO path between q_entity
    and a_entity in top-1000 triplets.
  - Condition 3 (GOOD TO HAVE): ≥15 samples where only a_entity is present
    in top-1000 triplets (but no path from q_entity to a_entity).
  - All question types present in the dataset must be represented.

Test: Full test set (no subsampling).

Output format: List of dicts compatible with JointTrainingDatasetv3PPR:
  - question, q_entity, a_entity, answer
  - question_embedding (384-dim, sentence-transformers)
  - topk_linearized_triplets (cosine-sorted triplet strings)
  - topk_linearized_triplet_embeddings
  - topk_rel_data: List[(score, (s, r, o))]
  - topK_rel_embeddings (relation embeddings)
  - is_empty

Pipeline:
1. Load KG from kb.txt → build networkx graph
2. Load QA file → extract topic entities and answers
3. For each question: BFS from topic entity to get candidate subgraph
4. Compute embeddings (question + triplets + relations)
5. Rank triplets by cosine similarity → top-N
6. Classify samples by path/coverage properties
7. Sample according to distribution constraints
8. Save as .pt file

Usage:
    # Preprocess 2-hop training data (curated 4000 subset)
    python generalization-study/preprocess_metaqa.py \
        --kb-path data/metaqa/kb.txt \
        --qa-train-path data/metaqa/2-hop/vanilla/qa_train.txt \
        --qa-dev-path data/metaqa/2-hop/vanilla/qa_dev.txt \
        --qa-test-path data/metaqa/2-hop/vanilla/qa_test.txt \
        --output-dir data/metaqa/processed \
        --hop 2

    # Preprocess 3-hop
    python generalization-study/preprocess_metaqa.py \
        --kb-path data/metaqa/kb.txt \
        --qa-train-path data/metaqa/3-hop/vanilla/qa_train.txt \
        --qa-dev-path data/metaqa/3-hop/vanilla/qa_dev.txt \
        --qa-test-path data/metaqa/3-hop/vanilla/qa_test.txt \
        --output-dir data/metaqa/processed \
        --hop 3
"""

import os
import sys
import re
import argparse
import random
import logging
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42

# =============================================================================
# Logging Setup
# =============================================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("preprocess_metaqa")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "preprocess_metaqa.log"), mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_formatter)
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_formatter)
logger.addHandler(_sh)

# =============================================================================
# Distribution Constraints (for curated sampling)
# =============================================================================
# Condition 1 (MUST): Minimum fraction of samples with path between q_entity and a_entity
MIN_PATH_EXISTS_FRACTION = 0.85
# Condition 2 (GOOD TO HAVE): Minimum fraction of samples with NO path between q_entity and a_entity
MIN_NO_PATH_FRACTION = 0.10
# Condition 3 (GOOD TO HAVE): Minimum fraction of samples where only a_entity is present (no path)
MIN_A_ENTITY_ONLY_FRACTION = 0.005


# =============================================================================
# KG Loading
# =============================================================================

def load_kg(kb_path: str) -> Tuple[nx.DiGraph, Dict[str, List[Tuple[str, str, str]]]]:
    """
    Load MetaQA knowledge graph from kb.txt.
    Format: subject|relation|object (one per line)

    Returns:
        graph: NetworkX directed graph (edges stored directionally,
               but BFS traverses both directions for reachability)
        entity_to_triplets: mapping from entity (lowercase) to list of
                           triplets involving it
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
            # Store edges in both directions for undirected BFS traversal
            # MetaQA KG relations are traversable in both directions for
            # multi-hop reasoning (e.g., "starred_actors" can be reversed)
            graph.add_edge(s.lower(), o.lower(), relation=r)
            graph.add_edge(o.lower(), s.lower(), relation=f"{r}_reverse")
            triplet = (s, r, o)
            entity_to_triplets[s.lower()].append(triplet)
            entity_to_triplets[o.lower()].append(triplet)

    logger.info(f"KG loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, entity_to_triplets


# =============================================================================
# QA File Loading
# =============================================================================

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
                "a_entity": answers,
                "answer": answers,
            })

    logger.info(f"QA file loaded: {len(samples)} questions")
    return samples


# =============================================================================
# Subgraph Extraction (BFS)
# =============================================================================

def extract_subgraph_triplets(
    graph: nx.DiGraph,
    entity_to_triplets: Dict,
    topic_entities: List[str],
    hop: int,
) -> List[Tuple[str, str, str]]:
    """
    Extract ALL candidate triplets via BFS from topic entities up to `hop` hops.

    The BFS traverses the graph in BOTH directions (successors + predecessors)
    since MetaQA reasoning paths can traverse relations in either direction.
    After BFS, we collect all ORIGINAL triplets (not reversed) that involve
    the visited nodes.

    Returns ALL deduplicated triplets from the subgraph. The caller is
    responsible for cosine-ranking and selecting top-k.
    """
    visited_nodes = set()
    frontier = set()

    for entity in topic_entities:
        e_lower = entity.lower()
        if e_lower in graph:
            frontier.add(e_lower)

    # BFS up to `hop` levels (traversing both directions)
    for _ in range(hop):
        next_frontier = set()
        for node in frontier:
            visited_nodes.add(node)
            for neighbor in graph.successors(node):
                if neighbor not in visited_nodes:
                    next_frontier.add(neighbor)
            for neighbor in graph.predecessors(node):
                if neighbor not in visited_nodes:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    visited_nodes.update(frontier)

    # Collect ORIGINAL triplets (not reversed) that involve visited nodes
    # Prioritize topic entity's direct triplets first
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

    return triplets


# =============================================================================
# Relation/Triplet Text Processing (matches WebQSP/CWQ notebook format)
# =============================================================================

def process_relation(relation: str) -> str:
    """
    Process a relation string to match the format used in WebQSP/CWQ data prep.
    Splits on dots and underscores, joins with spaces.

    e.g., "directed_by" → "directed by"
          "people.person.nationality" → "people person nationality"
    """
    relation_split = " ".join(relation.split("."))
    relation_split2 = " ".join(relation_split.split("_"))
    return relation_split2


def linearize_triplet(triplet: Tuple[str, str, str]) -> str:
    """
    Linearize a triplet to a sentence string, matching the notebook format.
    Format: "subject processed_relation object"

    e.g., ("Kismet", "directed_by", "William Dieterle")
        → "Kismet directed by William Dieterle"
    """
    s, r, o = triplet
    return f"{s} {process_relation(r)} {o}"


# =============================================================================
# Embedding Computation
# =============================================================================

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


# =============================================================================
# Sample Classification (for distribution constraints)
# =============================================================================

def classify_sample(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
) -> str:
    """
    Classify a sample based on path/coverage properties in top-1000 triplets.

    Categories:
      - "path_exists": Path exists between q_entity and a_entity in triplet graph
      - "a_entity_only": a_entity present in triplets but NO path from q_entity
      - "no_path": No path exists and a_entity may or may not be present

    Uses undirected graph for path checking (consistent with compute_path_coverage).
    """
    if not triplets or not q_entities or not a_entities:
        return "no_path"

    # Build undirected graph from triplets
    G = nx.Graph()
    for s, r, o in triplets:
        G.add_edge(s.lower(), o.lower())

    # Check entities in graph
    graph_nodes = set(G.nodes())
    a_entities_lower = [a.lower() for a in a_entities]
    q_entities_lower = [q.lower() for q in q_entities]

    # Check if path exists between any q_entity and any a_entity
    path_exists = False
    for q in q_entities_lower:
        for a in a_entities_lower:
            if q not in graph_nodes or a not in graph_nodes:
                continue
            try:
                if nx.has_path(G, q, a):
                    path_exists = True
                    break
            except nx.NetworkXError:
                continue
        if path_exists:
            break

    if path_exists:
        return "path_exists"

    # Check if any a_entity is at least present in the graph nodes
    a_present = any(a in graph_nodes for a in a_entities_lower)
    if a_present:
        return "a_entity_only"

    return "no_path"


def get_question_type(question_raw: str) -> str:
    """
    Extract the question template/type from a MetaQA question.
    MetaQA has fixed question templates per hop. We extract the template
    by replacing the entity in brackets with a placeholder.
    """
    return re.sub(r'\[([^\]]+)\]', '[ENTITY]', question_raw)


# =============================================================================
# Curated Sampling with Distribution Constraints
# =============================================================================

def curate_subset(
    classified_samples: List[Tuple[Dict, str, str]],
    target_size: int,
    min_path_fraction: float = MIN_PATH_EXISTS_FRACTION,
    min_no_path_fraction: float = MIN_NO_PATH_FRACTION,
    min_a_entity_only_fraction: float = MIN_A_ENTITY_ONLY_FRACTION,
) -> List[Dict]:
    """
    Sample a subset satisfying distribution constraints.

    Args:
        classified_samples: List of (processed_entry, category, question_type)
        target_size: Total samples to select
        min_path_fraction: Minimum fraction of "path_exists" samples (Condition 1)
        min_no_path_fraction: Minimum fraction of "no_path" samples (Condition 2)
        min_a_entity_only_fraction: Minimum fraction of "a_entity_only" samples (Condition 3)

    Returns:
        Curated list of processed entries
    """
    # Group by category
    path_exists_samples = []
    no_path_samples = []
    a_entity_only_samples = []

    for entry, category, qtype in classified_samples:
        if category == "path_exists":
            path_exists_samples.append((entry, qtype))
        elif category == "no_path":
            no_path_samples.append((entry, qtype))
        elif category == "a_entity_only":
            a_entity_only_samples.append((entry, qtype))

    logger.info(f"\n  Category distribution (total available):")
    logger.info(f"    path_exists:   {len(path_exists_samples)}")
    logger.info(f"    no_path:       {len(no_path_samples)}")
    logger.info(f"    a_entity_only: {len(a_entity_only_samples)}")

    # Calculate targets
    min_path_count = int(target_size * min_path_fraction)
    min_no_path_count = int(target_size * min_no_path_fraction)
    min_a_entity_only_count = max(1, int(target_size * min_a_entity_only_fraction))

    # Get all question types across all categories
    all_qtypes = set()
    for _, qtype in path_exists_samples:
        all_qtypes.add(qtype)
    for _, qtype in no_path_samples:
        all_qtypes.add(qtype)
    for _, qtype in a_entity_only_samples:
        all_qtypes.add(qtype)

    logger.info(f"  Question types found: {len(all_qtypes)}")
    logger.info(f"  Targets: path≥{min_path_count}, no_path≥{min_no_path_count}, a_only≥{min_a_entity_only_count}")

    # Step 1: Ensure all question types are represented
    # Take at least 1 sample per question type from path_exists (since it's the largest)
    selected = []
    selected_indices = set()

    # Group path_exists by question type
    path_by_qtype = defaultdict(list)
    for i, (entry, qtype) in enumerate(path_exists_samples):
        path_by_qtype[qtype].append((i, entry))

    no_path_by_qtype = defaultdict(list)
    for i, (entry, qtype) in enumerate(no_path_samples):
        no_path_by_qtype[qtype].append((i, entry))

    a_only_by_qtype = defaultdict(list)
    for i, (entry, qtype) in enumerate(a_entity_only_samples):
        a_only_by_qtype[qtype].append((i, entry))

    # Ensure each question type has at least 2 samples from path_exists
    path_selected_indices = set()
    for qtype, items in path_by_qtype.items():
        take = min(2, len(items))
        for idx, entry in items[:take]:
            if idx not in path_selected_indices:
                selected.append(("path_exists", entry))
                path_selected_indices.add(idx)

    # Step 2: Fill a_entity_only quota (Condition 3)
    a_only_selected_indices = set()
    random.shuffle(a_entity_only_samples)
    a_only_count = 0
    for i, (entry, qtype) in enumerate(a_entity_only_samples):
        if a_only_count >= min_a_entity_only_count:
            break
        selected.append(("a_entity_only", entry))
        a_only_selected_indices.add(i)
        a_only_count += 1

    # Step 3: Fill no_path quota (Condition 2)
    no_path_selected_indices = set()
    random.shuffle(no_path_samples)
    no_path_count = 0
    for i, (entry, qtype) in enumerate(no_path_samples):
        if no_path_count >= min_no_path_count:
            break
        selected.append(("no_path", entry))
        no_path_selected_indices.add(i)
        no_path_count += 1

    # Step 4: Fill remaining with path_exists to meet ≥75% and reach target_size
    remaining_needed = target_size - len(selected)
    remaining_path = [
        (i, entry) for i, (entry, qtype) in enumerate(path_exists_samples)
        if i not in path_selected_indices
    ]
    random.shuffle(remaining_path)

    for i, entry in remaining_path[:remaining_needed]:
        selected.append(("path_exists", entry))
        path_selected_indices.add(i)

    # If we still don't have enough, fill from other categories
    if len(selected) < target_size:
        remaining_no_path = [
            (i, entry) for i, (entry, qtype) in enumerate(no_path_samples)
            if i not in no_path_selected_indices
        ]
        random.shuffle(remaining_no_path)
        for i, entry in remaining_no_path:
            if len(selected) >= target_size:
                break
            selected.append(("no_path", entry))

    # Verify constraints
    final_path_count = sum(1 for cat, _ in selected if cat == "path_exists")
    final_no_path_count = sum(1 for cat, _ in selected if cat == "no_path")
    final_a_only_count = sum(1 for cat, _ in selected if cat == "a_entity_only")

    logger.info(f"\n  Final distribution ({len(selected)} samples):")
    logger.info(f"    path_exists:   {final_path_count} ({100*final_path_count/len(selected):.1f}%)")
    logger.info(f"    no_path:       {final_no_path_count} ({100*final_no_path_count/len(selected):.1f}%)")
    logger.info(f"    a_entity_only: {final_a_only_count}")

    if final_path_count / len(selected) < min_path_fraction:
        logger.warning(f"path_exists fraction ({final_path_count/len(selected):.2%}) "
                       f"is below target ({min_path_fraction:.0%})")

    # Return just the entries
    result = [entry for _, entry in selected]
    random.shuffle(result)  # Shuffle final order
    return result


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_samples(
    samples: List[Dict],
    graph: nx.DiGraph,
    entity_to_triplets: Dict,
    embed_model,
    hop: int,
    top_k_triplets: int = 1000,
    desc: str = "Processing",
) -> List[Tuple[Dict, str, str]]:
    """
    Process raw QA samples into JointTrainer format and classify them.

    For each sample:
    1. Extract ALL triplets from the hop-depth BFS subgraph
    2. Cosine-rank them against the question
    3. Take top-k (default 1000) after ranking
    4. Classify based on path/coverage in those top-k triplets

    Returns:
        List of (processed_entry_dict, category, question_type)
    """
    classified = []

    for sample in tqdm(samples, desc=desc):
        topic_entities = sample["q_entity"]
        question_type = get_question_type(sample.get("question_raw", sample["question"]))

        # Extract ALL triplets from the hop-depth BFS subgraph (no truncation)
        all_triplets = extract_subgraph_triplets(
            graph, entity_to_triplets, topic_entities, hop
        )

        if len(all_triplets) == 0:
            continue

        # Linearize ALL triplets matching notebook format:
        # "subject processed_relation object" (dots→spaces, underscores→spaces)
        all_linearized = [linearize_triplet(t) for t in all_triplets]

        # Compute question embedding
        question_embedding = embed_model.encode(
            sample["question"], convert_to_tensor=True
        ).cpu()

        # Compute triplet embeddings for ALL subgraph triplets
        all_triplet_embeddings = compute_embeddings(all_linearized, embed_model)

        # Compute cosine similarity scores for ranking
        q_emb_norm = question_embedding / (question_embedding.norm() + 1e-10)
        t_emb_norm = all_triplet_embeddings / (all_triplet_embeddings.norm(dim=1, keepdim=True) + 1e-10)
        cosine_scores = torch.matmul(t_emb_norm, q_emb_norm).squeeze()

        if cosine_scores.dim() == 0:
            cosine_scores = cosine_scores.unsqueeze(0)

        # Sort by cosine similarity (descending)
        sorted_indices = torch.argsort(cosine_scores, descending=True)

        # Take top-k after cosine ranking
        top_k = min(top_k_triplets, len(sorted_indices))
        sorted_indices = sorted_indices[:top_k]

        # Reorder and truncate to top-k
        sorted_triplets = [all_triplets[i] for i in sorted_indices.tolist()]
        sorted_linearized = [all_linearized[i] for i in sorted_indices.tolist()]
        sorted_triplet_embeddings = all_triplet_embeddings[sorted_indices]
        sorted_scores = cosine_scores[sorted_indices]

        # Compute relation embeddings for top-k: embed the PROCESSED relation text
        # (matches notebook: process_relation splits on dots and underscores)
        processed_relations = [process_relation(r) for _, r, _ in sorted_triplets]
        sorted_relation_embeddings = compute_embeddings(processed_relations, embed_model)

        # Build topk_rel_data matching notebook format:
        # List[(processed_relation_string, (s, r, o))]
        topk_rel_data = [
            (process_relation(sorted_triplets[i][1]), sorted_triplets[i])
            for i in range(len(sorted_triplets))
        ]

        # Classify sample based on path coverage in these top-k triplets
        category = classify_sample(
            sorted_triplets, sample["q_entity"], sample["a_entity"]
        )

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

        classified.append((processed_entry, category, question_type))

    return classified


def preprocess_metaqa(
    kb_path: str,
    qa_train_path: str,
    qa_dev_path: str,
    qa_test_path: str,
    output_dir: str,
    hop: int,
    max_triplets: int = 1000,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    train_size: int = 4000,
    val_size: int = 500,
):
    """
    Main preprocessing pipeline.

    Produces three .pt files:
      - metaqa-{hop}hop-train.pt  (curated train_size subset)
      - metaqa-{hop}hop-val.pt    (curated val_size subset)
      - metaqa-{hop}hop-test.pt   (full test set)
    """
    from sentence_transformers import SentenceTransformer

    random.seed(SEED)
    np.random.seed(SEED)

    logger.info(f"{'=' * 60}")
    logger.info(f"PREPROCESSING MetaQA {hop}-hop")
    logger.info(f"{'=' * 60}")
    logger.info(f"KB:         {kb_path}")
    logger.info(f"QA Train:   {qa_train_path}")
    logger.info(f"QA Dev:     {qa_dev_path}")
    logger.info(f"QA Test:    {qa_test_path}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Max triplets: {max_triplets}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Val size:   {val_size}")
    logger.info(f"Embedding:  {embedding_model_name}")
    logger.info(f"{'=' * 60}")

    # Load embedding model
    logger.info("\nLoading embedding model...")
    embed_model = SentenceTransformer(embedding_model_name)

    # Load KG
    logger.info("\nLoading knowledge graph...")
    graph, entity_to_triplets = load_kg(kb_path)

    # =========================================================================
    # Process TRAINING set (curated subset)
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 1: Processing TRAINING set (curating {train_size} from full train)")
    logger.info(f"{'=' * 60}")

    train_samples = load_qa_file(qa_train_path)
    logger.info(f"Processing all {len(train_samples)} training samples for classification...")

    train_classified = process_samples(
        train_samples, graph, entity_to_triplets, embed_model,
        hop, top_k_triplets=max_triplets, desc=f"Train {hop}-hop"
    )

    logger.info(f"\nClassified {len(train_classified)} samples. Curating {train_size} subset...")
    train_curated = curate_subset(train_classified, train_size)

    # Save training data
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, f"metaqa-{hop}hop-train.pt")
    torch.save(train_curated, train_output)
    logger.info(f"\n✓ Training data saved: {train_output} ({len(train_curated)} samples)")

    # Free memory
    del train_classified, train_samples, train_curated
    import gc
    gc.collect()

    # =========================================================================
    # Process VALIDATION set (curated subset)
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 2: Processing VALIDATION set (curating {val_size} from dev)")
    logger.info(f"{'=' * 60}")

    dev_samples = load_qa_file(qa_dev_path)
    logger.info(f"Processing all {len(dev_samples)} dev samples for classification...")

    dev_classified = process_samples(
        dev_samples, graph, entity_to_triplets, embed_model,
        hop, top_k_triplets=max_triplets, desc=f"Val {hop}-hop"
    )

    logger.info(f"\nClassified {len(dev_classified)} samples. Curating {val_size} subset...")
    val_curated = curate_subset(dev_classified, val_size)

    # Save validation data
    val_output = os.path.join(output_dir, f"metaqa-{hop}hop-val.pt")
    torch.save(val_curated, val_output)
    logger.info(f"\n✓ Validation data saved: {val_output} ({len(val_curated)} samples)")

    # Free memory
    del dev_classified, dev_samples, val_curated
    gc.collect()

    # =========================================================================
    # Process TEST set (full, no subsampling)
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 3: Processing TEST set (full)")
    logger.info(f"{'=' * 60}")

    test_samples = load_qa_file(qa_test_path)
    logger.info(f"Processing all {len(test_samples)} test samples...")

    test_classified = process_samples(
        test_samples, graph, entity_to_triplets, embed_model,
        hop, top_k_triplets=max_triplets, desc=f"Test {hop}-hop"
    )

    # For test, just extract the processed entries (no curation needed)
    test_data = [entry for entry, _, _ in test_classified]

    # Save test data
    test_output = os.path.join(output_dir, f"metaqa-{hop}hop-test.pt")
    torch.save(test_data, test_output)
    logger.info(f"\n✓ Test data saved: {test_output} ({len(test_data)} samples)")

    # Free memory
    del test_classified, test_samples, test_data
    del graph, entity_to_triplets, embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PREPROCESSING COMPLETE - MetaQA {hop}-hop")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Train: {train_output}")
    logger.info(f"  Val:   {val_output}")
    logger.info(f"  Test:  {test_output}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MetaQA into KGScout JointTrainer format"
    )
    parser.add_argument("--kb-path", type=str, required=True,
                        help="Path to MetaQA kb.txt file")
    parser.add_argument("--qa-train-path", type=str, required=True,
                        help="Path to MetaQA qa_train.txt file")
    parser.add_argument("--qa-dev-path", type=str, required=True,
                        help="Path to MetaQA qa_dev.txt file")
    parser.add_argument("--qa-test-path", type=str, required=True,
                        help="Path to MetaQA qa_test.txt file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed .pt files")
    parser.add_argument("--hop", type=int, required=True, choices=[1, 2, 3],
                        help="Number of hops (1, 2, or 3)")
    parser.add_argument("--max-triplets", type=int, default=1000,
                        help="Maximum candidate triplets per question (default: 1000)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--train-size", type=int, default=4000,
                        help="Number of curated training samples (default: 4000)")
    parser.add_argument("--val-size", type=int, default=500,
                        help="Number of curated validation samples (default: 500)")
    args = parser.parse_args()

    preprocess_metaqa(
        kb_path=args.kb_path,
        qa_train_path=args.qa_train_path,
        qa_dev_path=args.qa_dev_path,
        qa_test_path=args.qa_test_path,
        output_dir=args.output_dir,
        hop=args.hop,
        max_triplets=args.max_triplets,
        embedding_model_name=args.embedding_model,
        train_size=args.train_size,
        val_size=args.val_size,
    )
