"""
Used by StatisticalAnalysisService to compute extended metrics for:

  Case 5 (cosine outperforms KGScout):
    - Lexical overlap: word Jaccard between question and cosine triplet entities
    - Hop count: minimum path length in cosine top-k graph

  KGScout failure funnel (Case 5 + Case 6):
    - Level 1 — KG incomplete: answer not in full 2-hop subgraph
    - Level 2 — Candidate missing: answer not in cosine top-1000 pool
    - Level 3 — Selection failure: answer in pool but not selected in top-k
"""

import re
from typing import List, Tuple, Optional, Dict

from src.utils.metrics import compute_answer_coverage


# ---------------------------------------------------------------------------
# Stopwords for lexical overlap
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "what", "who", "where", "when", "which", "is", "the", "a", "an",
    "of", "in", "did", "does", "was", "were", "are", "to", "for",
    "by", "on", "at", "from", "with", "that", "this", "how", "do",
    "has", "have", "had", "be", "been", "its", "it", "and", "or",
    "not", "no", "name", "give", "me", "tell", "us", "their",
}


# ---------------------------------------------------------------------------
# Case 5: Lexical overlap
# ---------------------------------------------------------------------------

def compute_lexical_overlap(
    question: str,
    triplets: List[Tuple[str, str, str]],
) -> float:
    """
    Compute word-level Jaccard similarity between question text and
    entity names (subject + object) in the triplets.

    High overlap means the question text shares surface words with the
    retrieved entity names — cosine wins by text matching, not reasoning.

    Args:
        question: Raw question string.
        triplets: List of (subject, relation, object) tuples.

    Returns:
        Jaccard similarity in [0, 1]. Returns 0.0 if either set is empty.
    """
    question_tokens = set(
        tok for tok in re.split(r"[\s\W]+", question.lower())
        if tok and tok not in _STOPWORDS
    )

    # Collect tokens from entity names (subject and object only, not relation)
    entity_tokens = set()
    for s, _, o in triplets:
        for tok in re.split(r"[\s\W]+", s.lower()):
            if tok and tok not in _STOPWORDS:
                entity_tokens.add(tok)
        for tok in re.split(r"[\s\W]+", o.lower()):
            if tok and tok not in _STOPWORDS:
                entity_tokens.add(tok)

    union = question_tokens | entity_tokens
    if not union:
        return 0.0

    intersection = question_tokens & entity_tokens
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Case 5: Hop count
# ---------------------------------------------------------------------------

def compute_min_hop(paths: List[List[str]]) -> Optional[int]:
    """
    Compute the minimum reasoning hop count from a list of paths.

    Each path is a list of nodes e.g. [q_entity, node_A, answer_entity].
    Hop count = len(path) - 1.

    We take the minimum because if any direct 1-hop path exists,
    the question is structurally simple regardless of longer alternatives.

    Args:
        paths: List of node sequences from _get_all_paths.

    Returns:
        Minimum hop count, or None if paths is empty (no path found).
    """
    if not paths:
        return None
    valid = [len(p) - 1 for p in paths if len(p) >= 2]
    return min(valid) if valid else None


# ---------------------------------------------------------------------------
# KGScout failure funnel
# ---------------------------------------------------------------------------

def classify_kgscout_failure(
    full_triplets: List[Tuple[str, str, str]],
    pool_triplets: List[Tuple[str, str, str]],
    kgscout_triplets: List[Tuple[str, str, str]],
    a_ents: List[str],
) -> str:
    """
    Classify the root cause of a KGScout retrieval failure using a 3-level funnel.

    Call this only for questions where KGScout failed to retrieve the answer
    (i.e. Case 5 and Case 6).

    Funnel levels:
      1. KG incomplete     — answer not in full 2-hop subgraph
      2. Candidate missing — answer in full subgraph but not in cosine top-1000 pool
      3. Selection failure — answer in pool but KGScout did not select it in top-k

    Args:
        full_triplets: All triplets from full_dataset[idx]["topk_rel_data"] (no truncation).
        pool_triplets: Cosine top-1000 triplets from select_triplets_cosine(batch, k=1000).
        kgscout_triplets: KGScout top-k selected triplets (already computed).
        a_ents: Answer entities (lowercased).

    Returns:
        One of: "kg_incomplete", "candidate_missing", "selection_failure".
    """
    # Level 1: Is answer in the full 2-hop subgraph?
    if not compute_answer_coverage(full_triplets, a_ents):
        return "kg_incomplete"

    # Level 2: Is answer in the cosine top-1000 pool?
    if not compute_answer_coverage(pool_triplets, a_ents):
        return "candidate_missing"

    # Level 3: Answer was in pool but KGScout didn't select it in top-k
    return "selection_failure"


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def aggregate_case5_stats(records: List[Dict]) -> Dict:
    """
    Aggregate Case 5 extended metrics across all Case 5 questions.

    Args:
        records: List of dicts with keys:
            - "lexical_overlap": float
            - "min_hop": int or None

    Returns:
        Dict with count, avg_lexical_overlap, avg_min_hop, hop_distribution.
    """
    if not records:
        return {
            "count": 0,
            "avg_lexical_overlap": 0.0,
            "avg_min_hop": None,
            "hop_distribution": {"1": 0.0, "2": 0.0, "3+": 0.0},
        }

    overlaps = [r["lexical_overlap"] for r in records]
    hops = [r["min_hop"] for r in records if r["min_hop"] is not None]

    hop_counts = {"1": 0, "2": 0, "3+": 0}
    for h in hops:
        if h == 1:
            hop_counts["1"] += 1
        elif h == 2:
            hop_counts["2"] += 1
        else:
            hop_counts["3+"] += 1

    total_with_hops = len(hops)
    hop_distribution = {
        k: round(v / total_with_hops * 100, 2) if total_with_hops > 0 else 0.0
        for k, v in hop_counts.items()
    }

    return {
        "count": len(records),
        "avg_lexical_overlap": round(sum(overlaps) / len(overlaps), 4),
        "avg_min_hop": round(sum(hops) / len(hops), 2) if hops else None,
        "hop_distribution": hop_distribution,
    }


def aggregate_failure_funnel(funnel_records: List[Dict]) -> Dict:
    """
    Aggregate KGScout failure funnel across all failing questions.

    Args:
        funnel_records: List of dicts with keys:
            - "case": "case5" or "case6"
            - "label": "kg_incomplete" | "candidate_missing" | "selection_failure"

    Returns:
        Dict with breakdown for all failures combined, Case 5 only, and Case 6 only.
    """
    labels = ["kg_incomplete", "candidate_missing", "selection_failure"]

    def _summarise(subset: List[Dict]) -> Dict:
        total = len(subset)
        counts = {lbl: sum(1 for r in subset if r["label"] == lbl) for lbl in labels}
        return {
            "total": total,
            **{
                lbl: {
                    "count": counts[lbl],
                    "pct": round(counts[lbl] / total * 100, 2) if total > 0 else 0.0,
                }
                for lbl in labels
            },
        }

    case5_records = [r for r in funnel_records if r["case"] == "case5"]
    case6_records = [r for r in funnel_records if r["case"] == "case6"]

    return {
        "all_kgscout_failures": _summarise(funnel_records),
        "case5_failures": _summarise(case5_records),
        "case6_failures": _summarise(case6_records),
    }
