#!/usr/bin/env python3
"""
Randomly sample 4000 training + 500 validation samples from MetaQA,
save them as .pt files, and report statistics on:
  - Answer entity presence in top-1000 and top-1500 cosine-ranked triplets
  - Path coverage (q_entity → a_entity) in top-1000 and top-1500 triplets

This is a baseline/analysis script to understand the natural distribution
of the MetaQA dataset without curated sampling constraints.

Usage:
    python scripts/metaqa_random_sample_stats.py \
        --kb-path data/metaqa/kb.txt \
        --qa-train-path data/metaqa/2-hop/vanilla/qa_train.txt \
        --qa-dev-path data/metaqa/2-hop/vanilla/qa_dev.txt \
        --qa-test-path data/metaqa/2-hop/vanilla/qa_test.txt \
        --output-dir data/metaqa/processed-random \
        --hop 2

    # Or use config.yml paths:
    python scripts/metaqa_random_sample_stats.py --from-config --hop 2
"""

import os
import sys
import json
import random
import logging
import argparse
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42

# =============================================================================
# Logging
# =============================================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("metaqa_random_sample_stats")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "metaqa_random_sample_stats.log"), mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_formatter)
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_formatter)
logger.addHandler(_sh)


# =============================================================================
# Reuse functions from preprocess_metaqa.py
# =============================================================================

def process_relation(relation: str) -> str:
    """Process relation: dots→spaces, underscores→spaces."""
    relation_split = " ".join(relation.split("."))
    return " ".join(relation_split.split("_"))


def linearize_triplet(triplet: Tuple[str, str, str]) -> str:
    """Linearize triplet matching notebook format."""
    s, r, o = triplet
    return f"{s} {process_relation(r)} {o}"


def load_kg(kb_path: str):
    """Load MetaQA KG from kb.txt."""
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
            graph.add_edge(o.lower(), s.lower(), relation=f"{r}_reverse")
            triplet = (s, r, o)
            entity_to_triplets[s.lower()].append(triplet)
            entity_to_triplets[o.lower()].append(triplet)

    logger.info(f"KG loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, entity_to_triplets


def load_qa_file(qa_path: str) -> List[Dict]:
    """Load MetaQA QA file."""
    import re
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
            entity_match = re.findall(r'\[([^\]]+)\]', question)
            q_entities = entity_match if entity_match else []
            clean_question = re.sub(r'\[([^\]]+)\]', r'\1', question)
            samples.append({
                "question": clean_question,
                "question_raw": question,
                "q_entity": q_entities,
                "a_entity": answers,
                "answer": answers,
            })
    logger.info(f"QA file loaded: {len(samples)} questions from {qa_path}")
    return samples


def extract_subgraph_triplets(graph, entity_to_triplets, topic_entities, hop):
    """Extract ALL triplets from BFS subgraph (no truncation)."""
    visited_nodes = set()
    frontier = set()

    for entity in topic_entities:
        e_lower = entity.lower()
        if e_lower in graph:
            frontier.add(e_lower)

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

    seen_triplets = set()
    triplets = []

    for entity in topic_entities:
        e_lower = entity.lower()
        for triplet in entity_to_triplets.get(e_lower, []):
            triplet_key = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            if triplet_key not in seen_triplets:
                seen_triplets.add(triplet_key)
                triplets.append(triplet)

    for node in visited_nodes:
        for triplet in entity_to_triplets.get(node, []):
            triplet_key = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            if triplet_key not in seen_triplets:
                seen_triplets.add(triplet_key)
                triplets.append(triplet)

    return triplets


def compute_embeddings(texts, model, batch_size=256):
    """Compute sentence embeddings in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_stats_for_topk(sorted_triplets: List[Tuple], q_entities: List[str],
                           a_entities: List[str], k: int) -> Dict:
    """
    Compute answer presence and path coverage for top-k triplets.
    """
    triplets_k = sorted_triplets[:k]

    if not triplets_k:
        return {"ans_present": False, "path_coverage": False}

    # Answer entity presence
    graph_entities = set()
    for s, r, o in triplets_k:
        graph_entities.add(s.lower())
        graph_entities.add(o.lower())

    ans_present = any(a.lower() in graph_entities for a in a_entities)

    # Path coverage (undirected graph)
    G = nx.Graph()
    for s, r, o in triplets_k:
        G.add_edge(s.lower(), o.lower())

    path_exists = False
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            if qn not in G or an not in G:
                continue
            try:
                if nx.has_path(G, qn, an):
                    path_exists = True
                    break
            except nx.NetworkXError:
                continue
        if path_exists:
            break

    return {"ans_present": ans_present, "path_coverage": path_exists}


# =============================================================================
# Main Processing
# =============================================================================

def process_and_stats(
    samples: List[Dict],
    graph,
    entity_to_triplets,
    embed_model,
    hop: int,
    desc: str = "Processing",
) -> Tuple[List[Dict], Dict]:
    """
    Process samples, save in JointTrainer format (with PPR graph_features),
    and collect statistics for top-1000 and top-1500.

    Returns:
        (processed_entries, stats_dict)

    Saves two files per split:
      - Full file with embeddings + PPR (for training)
      - Metadata-only file (question, q_entity, a_entity, answer, topk_linearized_triplets)
    """
    processed = []
    metadata_only = []
    stats = {
        "top_1000": {"ans_present": 0, "path_coverage": 0, "total": 0},
        "top_1500": {"ans_present": 0, "path_coverage": 0, "total": 0},
    }

    for sample in tqdm(samples, desc=desc):
        topic_entities = sample["q_entity"]

        all_triplets = extract_subgraph_triplets(
            graph, entity_to_triplets, topic_entities, hop
        )

        if len(all_triplets) == 0:
            continue

        # Linearize and embed
        all_linearized = [linearize_triplet(t) for t in all_triplets]

        question_embedding = embed_model.encode(
            sample["question"], convert_to_tensor=True
        ).cpu()

        all_triplet_embeddings = compute_embeddings(all_linearized, embed_model)

        # Cosine rank
        q_emb_norm = question_embedding / (question_embedding.norm() + 1e-10)
        t_emb_norm = all_triplet_embeddings / (all_triplet_embeddings.norm(dim=1, keepdim=True) + 1e-10)
        cosine_scores = torch.matmul(t_emb_norm, q_emb_norm).squeeze()

        if cosine_scores.dim() == 0:
            cosine_scores = cosine_scores.unsqueeze(0)

        sorted_indices = torch.argsort(cosine_scores, descending=True)

        # Get sorted triplets (full list for stats, top-1000 for saving)
        sorted_triplets_all = [all_triplets[i] for i in sorted_indices.tolist()]

        # Compute stats for top-1000 and top-1500
        for k_name, k_val in [("top_1000", 1000), ("top_1500", 1500)]:
            s = compute_stats_for_topk(
                sorted_triplets_all, sample["q_entity"], sample["a_entity"], k_val
            )
            stats[k_name]["total"] += 1
            if s["ans_present"]:
                stats[k_name]["ans_present"] += 1
            if s["path_coverage"]:
                stats[k_name]["path_coverage"] += 1

        # Save top-1000 for the .pt file (JointTrainer format)
        top_k = min(1000, len(sorted_indices))
        top_indices = sorted_indices[:top_k]

        sorted_triplets = [all_triplets[i] for i in top_indices.tolist()]
        sorted_linearized = [all_linearized[i] for i in top_indices.tolist()]
        sorted_triplet_embeddings = all_triplet_embeddings[top_indices]
        sorted_scores = cosine_scores[top_indices]

        # Relation embeddings (processed relation text)
        processed_relations = [process_relation(r) for _, r, _ in sorted_triplets]
        sorted_relation_embeddings = compute_embeddings(processed_relations, embed_model)

        # topk_rel_data: (processed_relation_string, triplet_tuple) — notebook format
        topk_rel_data = [
            (process_relation(sorted_triplets[i][1]), sorted_triplets[i])
            for i in range(len(sorted_triplets))
        ]

        # Compute PPR graph features (same logic as JointTrainingDatasetv3PPR)
        q_entity_lower = [e.lower() for e in sample["q_entity"]]
        triplets_for_ppr = [t[1] for t in topk_rel_data]

        if len(q_entity_lower) == 0 or len(triplets_for_ppr) == 0:
            graph_feats = torch.zeros((max(1, len(triplets_for_ppr)), 2), dtype=torch.float32)
        else:
            G_ppr = nx.DiGraph()
            for (s, r, o) in triplets_for_ppr:
                G_ppr.add_edge(s.lower(), o.lower(), relation=r.lower())

            personalization = {n: (1.0 if n in q_entity_lower else 0.0) for n in G_ppr.nodes()}

            if sum(personalization.values()) == 0:
                graph_feats = torch.zeros((len(triplets_for_ppr), 2), dtype=torch.float32)
            else:
                try:
                    ppr_scores = nx.pagerank(
                        G_ppr, alpha=0.85, personalization=personalization,
                        max_iter=100, tol=1e-05
                    )
                    graph_feats = []
                    for (s, r, o) in triplets_for_ppr:
                        s_, o_ = s.lower(), o.lower()
                        ppr_s = ppr_scores.get(s_, 0.0)
                        ppr_o = ppr_scores.get(o_, 0.0)
                        graph_feats.append([ppr_s, ppr_o])
                    graph_feats = torch.tensor(graph_feats, dtype=torch.float32)
                except Exception:
                    graph_feats = torch.zeros((len(triplets_for_ppr), 2), dtype=torch.float32)

        # Full entry (with embeddings + PPR)
        processed.append({
            "question": sample["question"],
            "q_entity": sample["q_entity"],
            "a_entity": sample["a_entity"],
            "answer": sample["answer"],
            "question_embedding": question_embedding,
            "topk_linearized_triplets": sorted_linearized,
            "topk_linearized_triplet_embeddings": sorted_triplet_embeddings,
            "topk_rel_data": topk_rel_data,
            "topK_rel_embeddings": sorted_relation_embeddings,
            "graph_features": graph_feats,
            "is_empty": False,
        })

        # Metadata-only entry (no embeddings)
        metadata_only.append({
            "question": sample["question"],
            "q_entity": sample["q_entity"],
            "a_entity": sample["a_entity"],
            "answer": sample["answer"],
            "topk_linearized_triplets": sorted_linearized,
        })

    return processed, metadata_only, stats


def main():
    parser = argparse.ArgumentParser(
        description="Random sample MetaQA + report coverage statistics"
    )
    parser.add_argument("--hop", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: from config processed_dir)")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # Read all paths from config.yml
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    g = cfg["generalization"]

    kb_path = g["kb_path"]
    qa_train_path = g[f"qa_train_{args.hop}hop"]
    qa_dev_path = g[f"qa_dev_{args.hop}hop"]
    qa_test_path = g[f"qa_test_{args.hop}hop"]
    embedding_model_name = g.get("embedding_model", "all-MiniLM-L6-v2")
    output_dir = args.output_dir or g["processed_dir"]

    logger.info("=" * 60)
    logger.info(f"MetaQA RANDOM SAMPLE + STATISTICS ({args.hop}-hop)")
    logger.info("=" * 60)
    logger.info(f"KB:         {kb_path}")
    logger.info(f"Train:      {qa_train_path}")
    logger.info(f"Dev:        {qa_dev_path}")
    logger.info(f"Test:       {qa_test_path}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Val size:   {args.val_size}")
    logger.info(f"Embedding:  {embedding_model_name}")
    logger.info("=" * 60)

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer(embedding_model_name)

    logger.info("Loading knowledge graph...")
    graph, entity_to_triplets = load_kg(kb_path)

    # =========================================================================
    # TRAIN: Random sample
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 1: TRAINING SET — random {args.train_size} samples")
    logger.info(f"{'=' * 60}")

    train_samples = load_qa_file(qa_train_path)
    if len(train_samples) > args.train_size:
        train_subset = random.sample(train_samples, args.train_size)
    else:
        train_subset = train_samples
    logger.info(f"Sampled {len(train_subset)} from {len(train_samples)} training questions")

    train_processed, train_metadata, train_stats = process_and_stats(
        train_subset, graph, entity_to_triplets, embed_model,
        args.hop, desc=f"Train {args.hop}-hop"
    )

    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-train.pt")
    train_meta_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-train-metadata.pt")
    torch.save(train_processed, train_output)
    torch.save(train_metadata, train_meta_output)
    logger.info(f"✓ Training data saved: {train_output} ({len(train_processed)} samples)")
    logger.info(f"✓ Training metadata saved: {train_meta_output}")

    # =========================================================================
    # VAL: Random sample
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 2: VALIDATION SET — random {args.val_size} samples")
    logger.info(f"{'=' * 60}")

    dev_samples = load_qa_file(qa_dev_path)
    if len(dev_samples) > args.val_size:
        val_subset = random.sample(dev_samples, args.val_size)
    else:
        val_subset = dev_samples
    logger.info(f"Sampled {len(val_subset)} from {len(dev_samples)} dev questions")

    val_processed, val_metadata, val_stats = process_and_stats(
        val_subset, graph, entity_to_triplets, embed_model,
        args.hop, desc=f"Val {args.hop}-hop"
    )

    val_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-val.pt")
    val_meta_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-val-metadata.pt")
    torch.save(val_processed, val_output)
    torch.save(val_metadata, val_meta_output)
    logger.info(f"✓ Validation data saved: {val_output} ({len(val_processed)} samples)")
    logger.info(f"✓ Validation metadata saved: {val_meta_output}")

    # =========================================================================
    # TEST: Full (no subsampling)
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 3: TEST SET — full")
    logger.info(f"{'=' * 60}")

    test_samples = load_qa_file(qa_test_path)

    test_processed, test_metadata, test_stats = process_and_stats(
        test_samples, graph, entity_to_triplets, embed_model,
        args.hop, desc=f"Test {args.hop}-hop"
    )

    test_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-test.pt")
    test_meta_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-test-metadata.pt")
    torch.save(test_processed, test_output)
    torch.save(test_metadata, test_meta_output)
    logger.info(f"✓ Test data saved: {test_output} ({len(test_processed)} samples)")
    logger.info(f"✓ Test metadata saved: {test_meta_output}")

    # =========================================================================
    # Report Statistics
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("COVERAGE STATISTICS")
    logger.info(f"{'=' * 60}")

    all_stats = {"train": train_stats, "val": val_stats, "test": test_stats}

    for split_name, split_stats in all_stats.items():
        logger.info(f"\n  [{split_name.upper()}]")
        for k_name in ["top_1000", "top_1500"]:
            total = split_stats[k_name]["total"]
            ans = split_stats[k_name]["ans_present"]
            path = split_stats[k_name]["path_coverage"]
            if total > 0:
                logger.info(f"    {k_name}: ans_present={ans}/{total} ({100*ans/total:.1f}%) | "
                            f"path_coverage={path}/{total} ({100*path/total:.1f}%)")
            else:
                logger.info(f"    {k_name}: no samples processed")

    # Save stats to JSON
    stats_output = os.path.join(output_dir, f"metaqa-{args.hop}hop-stats.json")
    stats_json = {}
    for split_name, split_stats in all_stats.items():
        stats_json[split_name] = {}
        for k_name in ["top_1000", "top_1500"]:
            total = split_stats[k_name]["total"]
            ans = split_stats[k_name]["ans_present"]
            path = split_stats[k_name]["path_coverage"]
            stats_json[split_name][k_name] = {
                "total": total,
                "ans_present": ans,
                "ans_present_pct": round(100 * ans / total, 2) if total > 0 else 0,
                "path_coverage": path,
                "path_coverage_pct": round(100 * path / total, 2) if total > 0 else 0,
            }

    with open(stats_output, 'w') as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"\n✓ Statistics saved: {stats_output}")

    logger.info(f"\n{'=' * 60}")
    logger.info("DONE")
    logger.info(f"  Train: {train_output}")
    logger.info(f"  Val:   {val_output}")
    logger.info(f"  Test:  {test_output}")
    logger.info(f"  Stats: {stats_output}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
