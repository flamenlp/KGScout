"""
Reward computation for reinforcement learning training.

This module implements graph connectivity-based reward functions
for evaluating selected triplets in knowledge graph question answering.

Reward Functions:
- compute_reward_v8: Full reward (4*presence + 6*connection), capped at 10.0
- reward_only_presence: Ablation — only entity presence component (4*presence)
- reward_only_connection: Ablation — only connectivity component (6*connection)
"""

from typing import List, Tuple, Optional, Callable
import networkx as nx


def _compute_components(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
    lambda_lin: float = 0.2,
) -> Tuple[float, float]:
    """
    Compute the two reward components from a set of triplets.

    Args:
        triplets: List of (subject, relation, object) tuples
        q_entities: List of question entity strings
        a_entities: List of answer entity strings
        lambda_lin: Linear decay factor for connectivity

    Returns:
        Tuple of (frac_presence, conn_score)
    """
    if not triplets:
        return 0.0, 0.0

    # Build directed graph from triplets
    G = nx.DiGraph()
    for s, p, o in triplets:
        s_l, o_l, p_l = s.lower(), o.lower(), p.lower()
        G.add_edge(s_l, o_l, relation=p_l)

    # 1. Fractional Answer Presence
    present = sum(1 for a in a_entities if a.lower() in G)
    frac_presence = present / len(a_entities) if a_entities else 0.0

    # 2. Graded Connectivity
    conn_score = 0.0
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            try:
                d = nx.shortest_path_length(G, qn, an)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            conn = max(0.0, 1.0 - lambda_lin * (d - 1))
            conn_score = max(conn_score, conn)

    return frac_presence, conn_score


def compute_reward_v8(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
    connectivity_mode: str = "linear",
    alpha: float = 0.8,
    lambda_lin: float = 0.2,
    max_hops: int = 5,
) -> float:
    """
    Compute reward based on graph connectivity metrics.

    Components combined with weights (w_pres=4, w_conn=6), capped at 10.0.
    reward = min(4*presence + 6*connection, 10.0)

    Args:
        triplets: List of (subject, relation, object) tuples
        q_entities: List of question entity strings
        a_entities: List of answer entity strings
        connectivity_mode: Mode for connectivity computation (default: "linear")
        alpha: Alpha parameter for connectivity (default: 0.8)
        lambda_lin: Linear decay factor for connectivity (default: 0.2)
        max_hops: Maximum number of hops to consider (default: 5)

    Returns:
        float: Computed reward value, capped at 10.0
    """
    frac_presence, conn_score = _compute_components(triplets, q_entities, a_entities, lambda_lin)
    total = 4.0 * frac_presence + 6.0 * conn_score
    return min(total, 10.0)


def reward_only_presence(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
    **kwargs,
) -> float:
    """
    Ablation reward: only entity presence component.

    reward = min(4.0 * frac_presence, 10.0)

    Args:
        triplets: List of (subject, relation, object) tuples
        q_entities: List of question entity strings
        a_entities: List of answer entity strings

    Returns:
        float: Reward based on presence only
    """
    frac_presence, _ = _compute_components(triplets, q_entities, a_entities)
    return min(4.0 * frac_presence, 10.0)


def reward_only_connection(
    triplets: List[Tuple[str, str, str]],
    q_entities: List[str],
    a_entities: List[str],
    **kwargs,
) -> float:
    """
    Ablation reward: only connectivity component.

    reward = min(6.0 * conn_score, 10.0)

    Args:
        triplets: List of (subject, relation, object) tuples
        q_entities: List of question entity strings
        a_entities: List of answer entity strings

    Returns:
        float: Reward based on connectivity only
    """
    _, conn_score = _compute_components(triplets, q_entities, a_entities)
    return min(6.0 * conn_score, 10.0)


# ============================================================================
# REWARD FUNCTION REGISTRY
# ============================================================================

REWARD_FUNCTIONS = {
    "default": compute_reward_v8,
    "only_presence": reward_only_presence,
    "only_connection": reward_only_connection,
}


def get_reward_function(name: Optional[str] = None) -> Callable:
    """
    Get a reward function by name.

    Args:
        name: Name of the reward function. If None, returns the default (compute_reward_v8).

    Returns:
        Callable reward function with signature (triplets, q_entities, a_entities) -> float

    Raises:
        ValueError: If name is not a valid reward function name
    """
    if name is None:
        return compute_reward_v8

    if name not in REWARD_FUNCTIONS:
        valid = list(REWARD_FUNCTIONS.keys())
        raise ValueError(f"Unknown reward function: '{name}'. Valid options: {valid}")

    return REWARD_FUNCTIONS[name]
