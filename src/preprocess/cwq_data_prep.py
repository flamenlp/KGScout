"""
CWQ Dataset Preprocessing Script for KGScout Training.

Produces a stratified 3000-sample training set from ComplexWebQuestions (CWQ)
using all-MiniLM-L6-v2 embeddings. Output is compatible with JointTrainingDatasetv3PPR.

Distribution:
  85% (2550) — path exists in top-k triplets
      40% 1-hop | 35% 2-hop | 20% 3-hop+ | 5% flexible
      Within each hop bucket: 85% reachable@1000, 15% reachable@1500-only
  15% (450) — no path in top-1500 triplets
"""

import os
import json
import random
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset

# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
TOTAL_SAMPLES = 3000
SBERT_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K_SAVE = 3000       # Max triplets to store per sample
TOP_K_1000 = 1000       # Reachability check threshold A
TOP_K_1500 = 1500       # Reachability check threshold B

# Distribution config
HAS_PATH_RATIO = 0.85   # 2550
NO_PATH_RATIO = 0.15    # 450

# Hop distribution within the 85% group
HOP_DISTRIBUTION = {
    "1hop": 0.40,        # 1020
    "2hop": 0.35,        # 893
    "3hop_plus": 0.20,   # 510
    "flexible": 0.05,    # 127
}

# Within each hop bucket: 85% from reachable@1000, 15% from reachable@1500-only
REACHABLE_1000_RATIO = 0.85
REACHABLE_1500_ONLY_RATIO = 0.15

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Utility Functions ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def process_relation(relation: str) -> str:
    """Convert dotted/underscored relation to readable text."""
    relation_split = " ".join(relation.split("."))
    relation_split2 = " ".join(relation_split.split("_"))
    return relation_split2


def linearize_triplet(triplet) -> str:
    """Convert (subject, relation, object) to a linearized sentence."""
    sub = str(triplet[0])
    rel = str(triplet[1])
    obj = str(triplet[2])
    processed_rel = process_relation(rel)
    return f"{sub} {processed_rel} {obj}"


def compute_hop_count(triplets: List, q_entities: List[str], a_entities: List[str]) -> int:
    """
    Compute the minimum hop count from any q_entity to any a_entity
    on the FULL subgraph. Returns 0 if no path exists.
    """
    if not triplets or not q_entities or not a_entities:
        return 0

    G = nx.DiGraph()
    for triplet in triplets:
        s, r, o = triplet[0], triplet[1], triplet[2]
        G.add_edge(s.lower(), o.lower(), relation=r.lower())

    min_hops = float("inf")
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            try:
                d = nx.shortest_path_length(G, qn, an)
                min_hops = min(min_hops, d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return min_hops if min_hops != float("inf") else 0


def check_reachability(sorted_triplets: List, q_entities: List[str],
                       a_entities: List[str], k: int) -> bool:
    """
    Check if there's a path from q_entity to a_entity in the top-k triplets.
    """
    top_k = sorted_triplets[:k]
    if not top_k or not q_entities or not a_entities:
        return False

    G = nx.DiGraph()
    for triplet in top_k:
        s, r, o = triplet[0], triplet[1], triplet[2]
        G.add_edge(s.lower(), o.lower(), relation=r.lower())

    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            try:
                nx.shortest_path_length(G, qn, an)
                return True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    return False


def compute_ppr_features(triplets: List, q_entities: List[str]) -> torch.Tensor:
    """
    Compute PPR features (ppr_subject, ppr_object) for each triplet.
    Returns tensor of shape (N, 2).
    """
    if not triplets or not q_entities:
        return torch.zeros((max(1, len(triplets)), 2), dtype=torch.float32)

    q_entity_lower = [e.lower() for e in q_entities]

    G = nx.DiGraph()
    for triplet in triplets:
        s, r, o = triplet[0], triplet[1], triplet[2]
        G.add_edge(s.lower(), o.lower(), relation=r.lower())

    personalization = {n: (1.0 if n in q_entity_lower else 0.0) for n in G.nodes()}

    if sum(personalization.values()) == 0:
        return torch.zeros((len(triplets), 2), dtype=torch.float32)

    try:
        ppr_scores = nx.pagerank(
            G, alpha=0.85, personalization=personalization,
            max_iter=100, tol=1e-05
        )
    except Exception:
        return torch.zeros((len(triplets), 2), dtype=torch.float32)

    graph_feats = []
    for triplet in triplets:
        s_, o_ = triplet[0].lower(), triplet[2].lower()
        ppr_s = ppr_scores.get(s_, 0.0)
        ppr_o = ppr_scores.get(o_, 0.0)
        graph_feats.append([ppr_s, ppr_o])

    return torch.tensor(graph_feats, dtype=torch.float32)


# ─── Phase 1: Load & Classify ────────────────────────────────────────────────

def phase1_classify(data) -> Dict[str, List[Dict]]:
    """
    Classify CWQ samples by hop count on the full subgraph.
    Returns pools keyed by hop category.
    """
    logger.info("Phase 1: Classifying samples by hop count on full subgraph...")

    pools = {
        "1hop": [],
        "2hop": [],
        "3hop_plus": [],
    }

    skipped = 0
    for item in tqdm(data, desc="Classifying by hop count"):
        triplets = item.get("graph", [])
        q_entities = item.get("q_entity", [])
        a_entities = item.get("a_entity", [])

        if not triplets or not q_entities or not a_entities:
            skipped += 1
            continue

        hop_count = compute_hop_count(triplets, q_entities, a_entities)

        if hop_count == 0:
            # No path in full graph — skip for now, might use for no-path group
            skipped += 1
            continue

        sample = {
            "question": item["question"],
            "answer": item["answer"],
            "q_entity": q_entities,
            "a_entity": a_entities,
            "graph": triplets,
            "id": item.get("id", ""),
            "hop_count": hop_count,
        }

        if hop_count == 1:
            pools["1hop"].append(sample)
        elif hop_count == 2:
            pools["2hop"].append(sample)
        else:  # 3+
            pools["3hop_plus"].append(sample)

    logger.info(f"Classification complete:")
    logger.info(f"  1-hop: {len(pools['1hop'])}")
    logger.info(f"  2-hop: {len(pools['2hop'])}")
    logger.info(f"  3-hop+: {len(pools['3hop_plus'])}")
    logger.info(f"  Skipped (no path in full graph or empty): {skipped}")

    return pools


# ─── Phase 2: Encode & Rank Triplets ─────────────────────────────────────────

def phase2_encode_and_rank(samples: List[Dict], sbert: SentenceTransformer,
                           device: str = "cpu") -> List[Dict]:
    """
    For each sample, encode question and triplets with MiniLM,
    rank triplets by cosine(question, linearized_triplet) descending.
    Stores sorted_triplets (the raw triplets in ranked order).
    """
    logger.info(f"Phase 2: Encoding and ranking triplets for {len(samples)} samples...")

    encoded_samples = []
    for sample in tqdm(samples, desc="Encoding & ranking"):
        triplets = sample["graph"]
        question = sample["question"]

        # Encode question
        question_embedding = sbert.encode(question, convert_to_tensor=True,
                                          show_progress_bar=False).to("cpu")

        # Linearize all triplets
        linearized = [linearize_triplet(t) for t in triplets]

        # Encode all linearized triplets
        triplet_embeddings = sbert.encode(linearized, batch_size=512,
                                          convert_to_tensor=True,
                                          show_progress_bar=False).to("cpu")

        # Compute cosine similarity: question vs each triplet
        cosine_scores = util.pytorch_cos_sim(question_embedding, triplet_embeddings)[0]

        # Sort descending
        sorted_indices = cosine_scores.argsort(descending=True).tolist()

        # Reorder triplets by rank
        sorted_triplets = [triplets[i] for i in sorted_indices]

        encoded_sample = {
            **sample,
            "question_embedding": question_embedding,
            "sorted_triplets": sorted_triplets,
            "sorted_indices": sorted_indices,
        }
        encoded_samples.append(encoded_sample)

    return encoded_samples


# ─── Phase 3: Check Reachability ──────────────────────────────────────────────

def phase3_reachability(encoded_samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Categorize samples into:
      A: reachable at top-1000
      B: reachable at top-1500 but NOT top-1000
      C: NOT reachable at top-1500 (candidates for no-path group)
    """
    logger.info(f"Phase 3: Checking reachability for {len(encoded_samples)} samples...")

    cat_A = []  # reachable@1000
    cat_B = []  # reachable@1500 only
    cat_C = []  # not reachable@1500

    for sample in tqdm(encoded_samples, desc="Reachability check"):
        sorted_triplets = sample["sorted_triplets"]
        q_entities = sample["q_entity"]
        a_entities = sample["a_entity"]

        reachable_1000 = check_reachability(sorted_triplets, q_entities, a_entities, TOP_K_1000)

        if reachable_1000:
            cat_A.append(sample)
        else:
            reachable_1500 = check_reachability(sorted_triplets, q_entities, a_entities, TOP_K_1500)
            if reachable_1500:
                cat_B.append(sample)
            else:
                cat_C.append(sample)

    logger.info(f"Reachability results:")
    logger.info(f"  Category A (reachable@1000): {len(cat_A)}")
    logger.info(f"  Category B (reachable@1500 only): {len(cat_B)}")
    logger.info(f"  Category C (not reachable@1500): {len(cat_C)}")

    return cat_A, cat_B, cat_C


# ─── Phase 4: Stratified Sampling ────────────────────────────────────────────

def phase4_stratified_sampling(
    pools: Dict[str, List[Dict]],
    sbert: SentenceTransformer,
    device: str = "cpu"
) -> List[Dict]:
    """
    Perform stratified sampling to get exactly 3000 samples.
    """
    logger.info("Phase 4: Stratified sampling...")

    total_has_path = int(TOTAL_SAMPLES * HAS_PATH_RATIO)  # 2550
    total_no_path = TOTAL_SAMPLES - total_has_path          # 450

    # Targets per hop (initial)
    hop_targets = {
        "1hop": int(total_has_path * HOP_DISTRIBUTION["1hop"]),      # 1020
        "2hop": int(total_has_path * HOP_DISTRIBUTION["2hop"]),      # 893
        "3hop_plus": int(total_has_path * HOP_DISTRIBUTION["3hop_plus"]),  # 510
    }
    flexible_count = total_has_path - sum(hop_targets.values())  # 127

    # Redistribute quota from undersized pools to the next lower hop
    # Order: 3hop_plus → 2hop → 1hop
    hop_order = ["3hop_plus", "2hop", "1hop"]
    fallback_map = {"3hop_plus": "2hop", "2hop": "1hop", "1hop": None}

    for hop_key in hop_order:
        pool_size = len(pools[hop_key])
        target = hop_targets[hop_key]
        if pool_size < target:
            shortfall = target - pool_size
            hop_targets[hop_key] = pool_size  # take what's available
            fallback = fallback_map[hop_key]
            if fallback:
                hop_targets[fallback] += shortfall
                logger.warning(f"  {hop_key} pool has only {pool_size} samples "
                               f"(need {target}). Redistributing {shortfall} to {fallback}.")
            else:
                # 1hop has no lower fallback, add to flexible
                flexible_count += shortfall
                logger.warning(f"  {hop_key} pool has only {pool_size} samples "
                               f"(need {target}). Adding {shortfall} to flexible.")

    logger.info(f"Final targets after redistribution: "
                f"1hop={hop_targets['1hop']}, 2hop={hop_targets['2hop']}, "
                f"3hop+={hop_targets['3hop_plus']}, flexible={flexible_count}, "
                f"no_path={total_no_path}")

    selected_samples = []
    no_path_candidates = []
    leftover_A = []  # leftover category A samples for flexible fill
    leftover_B = []  # leftover category B samples for flexible fill

    for hop_key in ["1hop", "2hop", "3hop_plus"]:
        pool = pools[hop_key]
        target = hop_targets[hop_key]

        if target == 0:
            logger.info(f"\nSkipping {hop_key} (target=0 after redistribution)")
            continue

        target_A = int(target * REACHABLE_1000_RATIO)   # 85%
        target_B = target - target_A                     # 15%

        logger.info(f"\nProcessing {hop_key} pool ({len(pool)} samples)...")
        logger.info(f"  Target: {target} (A={target_A}, B={target_B})")

        # Encode and rank
        encoded = phase2_encode_and_rank(pool, sbert, device)

        # Check reachability
        cat_A, cat_B, cat_C = phase3_reachability(encoded)

        logger.info(f"  Available: A={len(cat_A)}, B={len(cat_B)}, C={len(cat_C)}")

        # Sample from A
        random.shuffle(cat_A)
        sampled_A = cat_A[:target_A]

        # Sample from B
        random.shuffle(cat_B)
        sampled_B = cat_B[:target_B]

        # Handle shortfalls within A/B
        shortfall_A = target_A - len(sampled_A)
        shortfall_B = target_B - len(sampled_B)

        if shortfall_A > 0 and len(cat_B) > target_B:
            extra_from_B = cat_B[target_B:target_B + shortfall_A]
            sampled_B.extend(extra_from_B)
            logger.warning(f"  Shortfall in A ({shortfall_A}), borrowed from B")

        if shortfall_B > 0 and len(cat_A) > target_A:
            extra_from_A = cat_A[target_A:target_A + shortfall_B]
            sampled_A.extend(extra_from_A)
            logger.warning(f"  Shortfall in B ({shortfall_B}), borrowed from A")

        for s in sampled_A:
            s["category"] = "has_path"
            s["reachability"] = "top_1000"
        for s in sampled_B:
            s["category"] = "has_path"
            s["reachability"] = "top_1500_only"

        selected_samples.extend(sampled_A)
        selected_samples.extend(sampled_B)

        # Track leftovers for flexible fill
        used_A = len(sampled_A)
        used_B = len(sampled_B)
        leftover_A.extend(cat_A[used_A:])
        leftover_B.extend(cat_B[used_B:])

        # Collect no-path candidates
        no_path_candidates.extend(cat_C)

    # Flexible overflow: fill from leftover A/B samples
    current_count = len(selected_samples)
    needed_flexible = total_has_path - current_count
    if needed_flexible > 0:
        logger.info(f"\nFilling {needed_flexible} flexible slots from leftovers...")
        random.shuffle(leftover_A)
        random.shuffle(leftover_B)
        flexible_pool = leftover_A + leftover_B
        flexible_selected = flexible_pool[:needed_flexible]
        for s in flexible_selected:
            s["category"] = "has_path"
            s["reachability"] = s.get("reachability", "top_1000")
        selected_samples.extend(flexible_selected)
        logger.info(f"  Filled {len(flexible_selected)} flexible slots")

    # No-path group
    logger.info(f"\nSelecting {total_no_path} no-path samples from {len(no_path_candidates)} candidates...")
    random.shuffle(no_path_candidates)

    if len(no_path_candidates) < total_no_path:
        logger.warning(f"  Only {len(no_path_candidates)} no-path candidates available "
                       f"(need {total_no_path}). Using all available.")

    sampled_no_path = no_path_candidates[:total_no_path]
    for s in sampled_no_path:
        s["category"] = "no_path"
        s["reachability"] = "none"

    selected_samples.extend(sampled_no_path)

    logger.info(f"\nTotal selected: {len(selected_samples)}")
    return selected_samples


# ─── Phase 5: Build Final Dataset ────────────────────────────────────────────

def phase5_build_dataset(selected_samples: List[Dict],
                         sbert: SentenceTransformer,
                         device: str = "cpu") -> List[Dict]:
    """
    Build the final dataset in JointTrainingDatasetv3PPR-compatible format.
    For each sample, takes top-3000 triplets and computes all required fields.
    """
    logger.info(f"Phase 5: Building final dataset for {len(selected_samples)} samples...")

    final_data = []

    for sample in tqdm(selected_samples, desc="Building dataset"):
        sorted_triplets = sample["sorted_triplets"]
        question = sample["question"]
        q_entities = sample["q_entity"]
        a_entities = sample["a_entity"]
        answer = sample["answer"]
        question_embedding = sample["question_embedding"]

        # Take top-3000 (or all if fewer)
        top_triplets = sorted_triplets[:TOP_K_SAVE]
        num_triplets = len(top_triplets)

        if num_triplets == 0:
            # Edge case: empty
            final_data.append({
                "question": question,
                "is_empty": True,
                "q_entity": q_entities,
                "a_entity": a_entities,
                "answer": answer,
                "question_embedding": question_embedding,
                "topk_linearized_triplets": [],
                "topk_linearized_triplet_embeddings": torch.zeros((1, EMBED_DIM)),
                "topk_rel_data": [],
                "topK_rel_embeddings": torch.zeros((1, EMBED_DIM)),
                "graph_features": torch.zeros((1, 2)),
            })
            continue

        # Build topk_rel_data: List[(processed_relation, (s, r, o))]
        topk_rel_data = []
        for triplet in top_triplets:
            processed_rel = process_relation(str(triplet[1]))
            topk_rel_data.append((processed_rel, tuple(triplet)))

        # Build topk_linearized_triplets
        topk_linearized_triplets = [linearize_triplet(t) for t in top_triplets]

        # Encode linearized triplets → topk_linearized_triplet_embeddings
        topk_linearized_triplet_embeddings = sbert.encode(
            topk_linearized_triplets, batch_size=512,
            convert_to_tensor=True, show_progress_bar=False
        ).to("cpu")

        # Encode processed relations → topK_rel_embeddings
        processed_relations = [topk_rel_data[i][0] for i in range(num_triplets)]
        topK_rel_embeddings = sbert.encode(
            processed_relations, batch_size=512,
            convert_to_tensor=True, show_progress_bar=False
        ).to("cpu")

        # Compute PPR graph features
        graph_features = compute_ppr_features(top_triplets, q_entities)

        # Determine is_empty
        is_empty = sample.get("category") == "no_path"

        final_data.append({
            "id": sample.get("id", ""),
            "question": question,
            "is_empty": is_empty,
            "q_entity": q_entities,
            "a_entity": a_entities,
            "answer": answer,
            "question_embedding": question_embedding,
            "topk_linearized_triplets": topk_linearized_triplets,
            "topk_linearized_triplet_embeddings": topk_linearized_triplet_embeddings,
            "topk_rel_data": topk_rel_data,
            "topK_rel_embeddings": topK_rel_embeddings,
            "graph_features": graph_features,
        })

    logger.info(f"Final dataset size: {len(final_data)}")
    return final_data


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main(output_dir: str, device: str = "cpu"):
    set_seed(SEED)

    # Load CWQ dataset
    logger.info("Loading CWQ dataset from rmanluo/RoG-cwq...")
    cwq_train = load_dataset("rmanluo/RoG-cwq", split="train")
    logger.info(f"Loaded {len(cwq_train)} training samples")

    # Initialize sentence transformer
    logger.info(f"Loading SentenceTransformer: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL)

    # Phase 1: Classify by hop count
    pools = phase1_classify(cwq_train)

    # Phase 4: Stratified sampling (includes Phase 2 & 3 internally per hop)
    selected_samples = phase4_stratified_sampling(pools, sbert, device)

    # Phase 5: Build final dataset
    final_data = phase5_build_dataset(selected_samples, sbert, device)

    # Save final .pt dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cwq_train_3k_miniLM.pt")
    torch.save(final_data, output_path)
    logger.info(f"Dataset saved to {output_path}")

    # Save secondary JSON with raw triplets (not linearized)
    raw_data = []
    for sample in selected_samples:
        sorted_triplets = sample["sorted_triplets"]
        top_triplets = sorted_triplets[:TOP_K_SAVE]
        # Convert to list of lists for JSON serialization
        triplets_serializable = [list(t) if not isinstance(t, list) else t
                                 for t in top_triplets]
        raw_data.append({
            "id": sample.get("id", ""),
            "question": sample["question"],
            "q_entity": sample["q_entity"],
            "a_entity": sample["a_entity"],
            "answer": sample["answer"],
            "triplets": triplets_serializable,
        })

    raw_json_path = os.path.join(output_dir, "cwq_train_3k_raw.json")
    with open(raw_json_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Raw data JSON saved to {raw_json_path}")

    # Save metadata
    metadata = {
        "total_samples": len(final_data),
        "model": SBERT_MODEL,
        "embed_dim": EMBED_DIM,
        "top_k_save": TOP_K_SAVE,
        "seed": SEED,
        "distribution": {
            "has_path": sum(1 for s in selected_samples if s["category"] == "has_path"),
            "no_path": sum(1 for s in selected_samples if s["category"] == "no_path"),
        },
        "hop_counts": {
            "1hop": sum(1 for s in selected_samples
                        if s.get("hop_count") == 1 and s["category"] == "has_path"),
            "2hop": sum(1 for s in selected_samples
                        if s.get("hop_count") == 2 and s["category"] == "has_path"),
            "3hop_plus": sum(1 for s in selected_samples
                            if s.get("hop_count", 0) >= 3 and s["category"] == "has_path"),
        },
        "reachability": {
            "top_1000": sum(1 for s in selected_samples
                           if s.get("reachability") == "top_1000"),
            "top_1500_only": sum(1 for s in selected_samples
                                if s.get("reachability") == "top_1500_only"),
            "none": sum(1 for s in selected_samples
                        if s.get("reachability") == "none"),
        }
    }
    metadata_path = os.path.join(output_dir, "cwq_train_3k_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CWQ Data Preprocessing for KGScout")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the processed dataset")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for encoding (cpu or cuda)")
    args = parser.parse_args()

    main(args.output_dir, args.device)
