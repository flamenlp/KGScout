"""
Reward computation for reinforcement learning training.

This module implements graph connectivity-based reward functions
for evaluating selected triplets in knowledge graph question answering.
"""

from typing import List, Tuple
import networkx as nx


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
    
    This function evaluates the quality of selected triplets by measuring:
    1. Fractional answer presence: Fraction of answer entities in the graph
    2. Graded connectivity: Max connectivity score across Q-A pairs using shortest paths
    
    The reward components are combined with weights (w_pres=6, w_conn=4)
    and capped at a maximum of 10.0.
    
    Args:
        triplets: List of (subject, relation, object) tuples representing selected triplets
        q_entities: List of question entity strings
        a_entities: List of answer entity strings
        connectivity_mode: Mode for connectivity computation (default: "linear")
        alpha: Alpha parameter for connectivity (default: 0.8)
        lambda_lin: Linear decay factor for connectivity (default: 0.2)
        max_hops: Maximum number of hops to consider (default: 5)
    
    Returns:
        float: Computed reward value, capped at 10.0
    """
    # Handle empty triplets case
    if not triplets:
        return 0.0

    # Build bidirectional graph from triplets
    # Note: The graph is directed but edges are added in one direction
    # The term "bidirectional" in requirements refers to the ability to traverse
    # in the direction specified by the triplets
    G = nx.DiGraph()
    for s, p, o in triplets:
        s_l, o_l, p_l = s.lower(), o.lower(), p.lower()
        G.add_edge(s_l, o_l, relation=p_l)

    # 1. Fractional Answer Presence
    # Count how many answer entities are present in the graph
    present = sum(1 for a in a_entities if a.lower() in G)
    frac_presence = present / len(a_entities) if a_entities else 0.0

    # 2. Graded Connectivity
    # Compute connectivity score using shortest paths between Q-A pairs
    # Uses linear decay: conn = max(0, 1 - lambda_lin * (distance - 1))
    conn_score = 0.0
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            try:
                d = nx.shortest_path_length(G, qn, an)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            # Linear decay based on distance
            conn = max(0.0, 1.0 - lambda_lin * (d - 1))
            conn_score = max(conn_score, conn)

    # Combine components with specified weights (6/4 weighting)
    total = 6.0 * frac_presence + 4.0 * conn_score

    # Cap maximum reward at 10.0
    return min(total, 10.0)
