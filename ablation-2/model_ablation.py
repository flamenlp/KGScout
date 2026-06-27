"""
Model Architecture Ablation for Reversed Attention (ablation-2).

Base: PathRankingModelReversedAttention from run_reversed_attention.py
  - Reversed attention: Query=Question, Key=Value=Triplets/Relations
  - Gate blends triplet_weights + relation_weights → gated_attention_weights
  - Combiner: [tower_A, tower_B, gated_attention_weights]

Variants:
  no-ppr:  No PPR graph_scores in tower inputs (MLP input: 3*d, not 3*d+2)
  no-rt:   No relation tower. Combiner: [tower_A, gated_weights]. Gate+relation attn remain.
  no-tt:   No triplet tower. Combiner: [tower_B, gated_weights]. Gate+triplet attn remain.
  no-gate: No gate. Combiner: [tower_A, tower_B]. No gated blending.
  no-ra:   No relation attn/gate/relation tower/combiner. Only triplet attn → triplet tower.
  no-ta:   No triplet attn/gate/triplet tower/combiner. Only relation attn → relation tower.

Results: ./results/ablation-2/model-ablation/<variant>/
"""

import os
import sys
import json
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import logging
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

torch.manual_seed(100)
np.random.seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("ablation2.model")


# ============================================================================
# SHARED SAMPLE_PATHS
# ============================================================================

def _sample_paths_impl(probabilities, paths, k, ranking_scores):
    if len(paths) <= k:
        log_probs = torch.log(probabilities + 1e-10)
        return paths, probabilities, ranking_scores, log_probs
    selected_indices, log_probs_list = [], []
    remaining = torch.ones(len(probabilities), dtype=torch.bool, device=probabilities.device)
    for _ in range(min(k, len(paths))):
        masked = probabilities * remaining.float()
        masked = masked / (masked.sum() + 1e-10)
        dist = torch.distributions.Categorical(probs=masked)
        idx = dist.sample()
        selected_indices.append(idx.item())
        log_probs_list.append(dist.log_prob(idx))
        remaining[idx] = False
    idx_t = torch.tensor(selected_indices, device=probabilities.device)
    return ([paths[i] for i in selected_indices],
            probabilities[idx_t], ranking_scores[idx_t], torch.stack(log_probs_list))


# ============================================================================
# MODEL VARIANTS
# ============================================================================

class ReversedOriginal(nn.Module):
    """Full reversed attention model (same as run_reversed_attention.py)."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Sigmoid())
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        # Reversed attention
        t_att, t_w = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        r_att, r_w = self.question_relation_attention(q, relation_embeds, relation_embeds)
        t_w = t_w.squeeze(0).squeeze(0)
        r_w = r_w.squeeze(0).squeeze(0)
        t_att = t_att.expand(N, -1)
        r_att = r_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        # Gate
        gate_in = torch.cat([q_exp, triplet_embeds, relation_embeds], dim=-1)
        sigma = self.gate_network(gate_in).squeeze(-1)
        gated_w = sigma * t_w + (1 - sigma) * r_w
        # Tower A
        a_in = torch.cat([triplet_embeds, t_att, q_exp, graph_scores], dim=-1)
        s_a = self.triplet_mlp(a_in).squeeze(-1)
        # Tower B
        b_in = torch.cat([relation_embeds, r_att, q_exp, graph_scores], dim=-1)
        s_b = self.relation_mlp(b_in).squeeze(-1)
        # Combiner
        comb = torch.stack([s_a, s_b, gated_w], dim=-1)
        scores = self.combiner_mlp(comb).squeeze(-1)
        temp = self.temperature.clamp(min=0.1, max=5.0)
        return scores, F.softmax(scores / temp, dim=0)

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            'gate_network': self.gate_network.state_dict(),
            'triplet_mlp': self.triplet_mlp.state_dict(),
            'relation_mlp': self.relation_mlp.state_dict(),
            'combiner_mlp': self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu()
        }, os.path.join(d, "path_ranker.pt"))


class ReversedNoPPR(nn.Module):
    """Reversed model without PPR graph_scores (tower input: 3*d)."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Sigmoid())
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        t_att, t_w = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        r_att, r_w = self.question_relation_attention(q, relation_embeds, relation_embeds)
        t_w = t_w.squeeze(0).squeeze(0); r_w = r_w.squeeze(0).squeeze(0)
        t_att = t_att.expand(N, -1); r_att = r_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        sigma = self.gate_network(torch.cat([q_exp, triplet_embeds, relation_embeds], dim=-1)).squeeze(-1)
        gated_w = sigma * t_w + (1 - sigma) * r_w
        # No graph_scores
        s_a = self.triplet_mlp(torch.cat([triplet_embeds, t_att, q_exp], dim=-1)).squeeze(-1)
        s_b = self.relation_mlp(torch.cat([relation_embeds, r_att, q_exp], dim=-1)).squeeze(-1)
        scores = self.combiner_mlp(torch.stack([s_a, s_b, gated_w], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_triplet_attention': self.question_triplet_attention.state_dict(), 'question_relation_attention': self.question_relation_attention.state_dict(), 'gate_network': self.gate_network.state_dict(), 'triplet_mlp': self.triplet_mlp.state_dict(), 'relation_mlp': self.relation_mlp.state_dict(), 'combiner_mlp': self.combiner_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


class ReversedNoRT(nn.Module):
    """Reversed model without Relation Tower. Combiner: [tower_A, gated_weights]."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Sigmoid())
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(2, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        t_att, t_w = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        _, r_w = self.question_relation_attention(q, relation_embeds, relation_embeds)
        t_w = t_w.squeeze(0).squeeze(0); r_w = r_w.squeeze(0).squeeze(0)
        t_att = t_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        sigma = self.gate_network(torch.cat([q_exp, triplet_embeds, relation_embeds], dim=-1)).squeeze(-1)
        gated_w = sigma * t_w + (1 - sigma) * r_w
        s_a = self.triplet_mlp(torch.cat([triplet_embeds, t_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        scores = self.combiner_mlp(torch.stack([s_a, gated_w], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_triplet_attention': self.question_triplet_attention.state_dict(), 'question_relation_attention': self.question_relation_attention.state_dict(), 'gate_network': self.gate_network.state_dict(), 'triplet_mlp': self.triplet_mlp.state_dict(), 'combiner_mlp': self.combiner_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


class ReversedNoTT(nn.Module):
    """Reversed model without Triplet Tower. Combiner: [tower_B, gated_weights]."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Sigmoid())
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(2, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        _, t_w = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        r_att, r_w = self.question_relation_attention(q, relation_embeds, relation_embeds)
        t_w = t_w.squeeze(0).squeeze(0); r_w = r_w.squeeze(0).squeeze(0)
        r_att = r_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        sigma = self.gate_network(torch.cat([q_exp, triplet_embeds, relation_embeds], dim=-1)).squeeze(-1)
        gated_w = sigma * t_w + (1 - sigma) * r_w
        s_b = self.relation_mlp(torch.cat([relation_embeds, r_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        scores = self.combiner_mlp(torch.stack([s_b, gated_w], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_triplet_attention': self.question_triplet_attention.state_dict(), 'question_relation_attention': self.question_relation_attention.state_dict(), 'gate_network': self.gate_network.state_dict(), 'relation_mlp': self.relation_mlp.state_dict(), 'combiner_mlp': self.combiner_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


class ReversedNoGate(nn.Module):
    """Reversed model without Gate. Combiner: [tower_A, tower_B]. No gated blending."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(2, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        t_att, _ = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        r_att, _ = self.question_relation_attention(q, relation_embeds, relation_embeds)
        t_att = t_att.expand(N, -1); r_att = r_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        s_a = self.triplet_mlp(torch.cat([triplet_embeds, t_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        s_b = self.relation_mlp(torch.cat([relation_embeds, r_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        scores = self.combiner_mlp(torch.stack([s_a, s_b], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_triplet_attention': self.question_triplet_attention.state_dict(), 'question_relation_attention': self.question_relation_attention.state_dict(), 'triplet_mlp': self.triplet_mlp.state_dict(), 'relation_mlp': self.relation_mlp.state_dict(), 'combiner_mlp': self.combiner_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


class ReversedNoRA(nn.Module):
    """No relation attention/gate/relation tower/combiner. Only triplet attention → triplet tower."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = triplet_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        t_att, _ = self.question_triplet_attention(q, triplet_embeds, triplet_embeds)
        t_att = t_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        scores = self.triplet_mlp(torch.cat([triplet_embeds, t_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_triplet_attention': self.question_triplet_attention.state_dict(), 'triplet_mlp': self.triplet_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


class ReversedNoTA(nn.Module):
    """No triplet attention/gate/triplet tower/combiner. Only relation attention → relation tower."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_relation_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        N = relation_embeds.size(0)
        q = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        r_att, _ = self.question_relation_attention(q, relation_embeds, relation_embeds)
        r_att = r_att.expand(N, -1)
        q_exp = q.expand(N, -1)
        scores = self.relation_mlp(torch.cat([relation_embeds, r_att, q_exp, graph_scores], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

    def sample_paths(self, prob, paths, k, scores): return _sample_paths_impl(prob, paths, k, scores)
    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({'question_relation_attention': self.question_relation_attention.state_dict(), 'relation_mlp': self.relation_mlp.state_dict(), 'temperature': self.temperature.detach().cpu(), 'baseline': self.baseline.detach().cpu()}, os.path.join(d, "path_ranker.pt"))


# ============================================================================
# REWARD FUNCTION
# ============================================================================

def compute_reward_v8(triplets, q_entities, a_entities, lambda_lin=0.2):
    if not triplets:
        return 0.0
    G = nx.DiGraph()
    for s, p, o in triplets:
        G.add_edge(s.lower(), o.lower(), relation=p.lower())
    present = sum(1 for a in a_entities if a.lower() in G)
    frac_presence = present / len(a_entities) if a_entities else 0.0
    conn_score = 0.0
    for q in q_entities:
        for a in a_entities:
            try:
                d = nx.shortest_path_length(G, q.lower(), a.lower())
                conn_score = max(conn_score, max(0.0, 1.0 - lambda_lin * (d - 1)))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    triplet_pairs = {(s.lower(), o.lower()) for s, _, o in triplets}
    cov_scores = []
    for q in q_entities:
        for a in a_entities:
            try:
                path = nx.shortest_path(G, q.lower(), a.lower())
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) < 2:
                continue
            matches = sum(1 for u, v in zip(path, path[1:]) if (u, v) in triplet_pairs)
            cov_scores.append(matches / (len(path) - 1))
    path_cov = max(cov_scores) if cov_scores else 0.0
    return min(3 * frac_presence + 4 * conn_score + 3 * path_cov, 10.0)


# ============================================================================
# DATASETS
# ============================================================================

class SampledJointTrainingDataset(Dataset):
    def __init__(self, dataset, k=500):
        self.dataset = dataset
        self.k = k
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        n = min(self.k, data["topk_linearized_triplet_embeddings"].shape[0])
        return {k: data[k] for k in ("question", "is_empty", "q_entity", "a_entity", "answer", "question_embedding")} | {
            "topk_linearized_triplets": data["topk_linearized_triplets"][:n],
            "topk_linearized_triplet_embeddings": data["topk_linearized_triplet_embeddings"][:n],
            "topk_rel_data": data["topk_rel_data"][:n],
            "topK_rel_embeddings": data["topK_rel_embeddings"][:n],
            "graph_features": data["graph_features"][:n]}

class CosinePretrainingDataset(Dataset):
    def __init__(self, dataset, k=500):
        self.dataset = dataset
        self.k = k
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data["topk_linearized_triplets"]) == 0 or len(data["q_entity"]) == 0:
            return None
        n = min(self.k, data["topk_linearized_triplet_embeddings"].shape[0])
        return {"question_embedding": data["question_embedding"],
                "path_embeddings": data["topk_linearized_triplet_embeddings"][:n],
                "rel_embeddings": data["topK_rel_embeddings"][:n],
                "cosine_targets": torch.exp(-0.01 * torch.arange(n, dtype=torch.float)),
                "graph_features": data["graph_features"][:n]}

def collate_fn_pretrain(batch):
    batch = [x for x in batch if x is not None]
    return batch[0] if batch else None


# ============================================================================
# PRETRAINER & TRAINER (same logic, uses reversed models)
# ============================================================================

class CosinePretrainer:
    def __init__(self, path_ranker, device="cuda", checkpoint_dir="pretrain"):
        self.device = device
        self.path_ranker = path_ranker.to(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)

    def _ranking_loss(self, pred, tgt, pairs=100):
        bs = pred.size(0)
        if bs < 2: return torch.tensor(0.0, device=self.device, requires_grad=True)
        all_p = [(i,j) for i in range(bs) for j in range(i+1, bs)]
        sel = np.random.choice(len(all_p), min(pairs, len(all_p)), replace=False) if len(all_p) > pairs else range(len(all_p))
        pi = torch.tensor([all_p[i][0] for i in sel], device=self.device)
        pj = torch.tensor([all_p[i][1] for i in sel], device=self.device)
        return self.ranking_loss(pred[pi], pred[pj], torch.sign(tgt[pi] - tgt[pj]))

    def train(self, train_dl, val_dl=None, num_epochs=5, lr=1e-4, accum=8):
        opt = torch.optim.AdamW(self.path_ranker.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
        best = float('inf')
        for ep in range(num_epochs):
            logger.info(f"  Pretrain Epoch {ep+1}/{num_epochs}")
            self.path_ranker.train(); loss_sum = 0; cnt = 0; opt.zero_grad()
            for bi, b in enumerate(tqdm(train_dl, desc="  Pretrain", leave=False)):
                if b is None: continue
                q = b["question_embedding"].to(self.device)
                p = b["path_embeddings"].to(self.device)
                r = b["rel_embeddings"].to(self.device)
                t = b["cosine_targets"].to(self.device)
                g = b["graph_features"].to(self.device)
                pred, _ = self.path_ranker(q.unsqueeze(0), p, r, g)
                loss = 0.5*self.mse_loss(pred, t) + 0.5*self._ranking_loss(pred, t)
                (loss/accum).backward(); loss_sum += loss.item(); cnt += 1
                if (bi+1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), 1.0); opt.step(); opt.zero_grad()
            if cnt % accum != 0: opt.step(); opt.zero_grad()
            logger.info(f"    Loss: {loss_sum/max(cnt,1):.4f}")
            if val_dl:
                self.path_ranker.eval(); vl = 0; vc = 0
                with torch.no_grad():
                    for b in val_dl:
                        if b is None: continue
                        pred, _ = self.path_ranker(b["question_embedding"].to(self.device).unsqueeze(0), b["path_embeddings"].to(self.device), b["rel_embeddings"].to(self.device), b["graph_features"].to(self.device))
                        vl += (0.5*self.mse_loss(pred, b["cosine_targets"].to(self.device))).item(); vc += 1
                avg_v = vl/max(vc,1); logger.info(f"    Val: {avg_v:.4f}"); sched.step(avg_v)
                if avg_v < best: best = avg_v; self._save(ep+1)
        self._save(num_epochs)

    def _save(self, ep):
        torch.save({'epoch': ep, 'model_state_dict': self.path_ranker.state_dict()},
                   os.path.join(self.checkpoint_dir, f'best_pretrained_model-{ep}.pt'))


class JointTrainer:
    def __init__(self, path_ranker, reward_func, max_grad_norm=1.0, gradient_accumulation_steps=32,
                 checkpoint_dir="ckpt", gamma=0.99, baseline_decay=0.9):
        self.reward_func = reward_func
        self.path_ranker = path_ranker.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.accum = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.running_baseline = 0; self.reward_buffer = []; self.best_val_reward = float('-inf')

    def _reinforce_loss(self, lp, rew, bl):
        return -(lp * (rew - bl).detach()).mean()

    def _update_baseline(self):
        if self.reward_buffer:
            avg = sum(self.reward_buffer)/len(self.reward_buffer)
            if self.running_baseline == 0: self.running_baseline = avg*0.8
            else:
                err = avg - self.running_baseline
                if abs(err) > 0.5: self.running_baseline += 0.1*err
            self.running_baseline = min(self.running_baseline, avg*0.9)
            self.path_ranker.baseline.data = torch.tensor([self.running_baseline], device=self.device)
            self.reward_buffer = []

    def train_step(self, batch, k):
        q_ent = [p[0] for p in batch['q_entity']]
        a_ent = [p[0] for p in batch['a_entity']]
        triplets = [(d[1][0][0], d[1][1][0], d[1][2][0]) for d in batch["topk_rel_data"]]
        if not q_ent: return None, None
        qe = batch["question_embedding"].to(self.device)
        te = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(self.device)
        re = batch["topK_rel_embeddings"].squeeze(0).to(self.device)
        gf = batch["graph_features"].squeeze(0).to(self.device)
        scores, probs = self.path_ranker(qe, te, re, gf)
        sel_t, _, _, lp = self.path_ranker.sample_paths(probs, triplets, k, scores)
        rew = self.reward_func(sel_t, q_ent, a_ent)
        if rew is None: return None, None
        r = torch.tensor([rew], device=self.device)
        self.reward_buffer.append(r.item())
        loss = self._reinforce_loss(lp, r.expand(lp.size(0)), torch.tensor([self.running_baseline], device=self.device))
        return loss, r

    @torch.no_grad()
    def validate(self, dl, k):
        self.path_ranker.eval(); tl=0; tr=0; n=0
        for b in tqdm(dl, desc="  Val", leave=False):
            l, r = self.train_step(b, k)
            if l is None: continue
            tl += l.item(); tr += r.item(); n += 1
        return tl/max(n,1), tr/max(n,1)

    def train(self, train_dl, val_dl, num_epochs=30, learning_rate=1e-4, warmup_steps=100,
              scheduler_type='cosine', validation_interval=1, early_stopping_patience=3, k=100):
        opt = torch.optim.AdamW(self.path_ranker.parameters(), lr=learning_rate)
        total = (len(train_dl)*num_epochs)//self.accum
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total)
        pat = 0
        for ep in range(num_epochs):
            logger.info(f"  Epoch {ep+1}/{num_epochs}")
            self.path_ranker.train(); rews=[]; losses=[]; opt.zero_grad(); vc=0
            for bi, b in enumerate(tqdm(train_dl, desc="  Train", leave=False)):
                l, r = self.train_step(b, k)
                if l is None: continue
                if math.isnan(r.item()): continue
                rews.append(r.item()); losses.append(l.item()); vc+=1
                (l/self.accum).backward()
                if vc % self.accum == 0:
                    self._update_baseline()
                    torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), self.max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad()
            if vc % self.accum != 0:
                self._update_baseline(); torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), self.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad()
            logger.info(f"    Reward: {np.mean(rews) if rews else 0:.4f}, Loss: {np.mean(losses) if losses else 0:.4f}")
            if (ep+1) % validation_interval == 0 and val_dl:
                vl, vr = self.validate(val_dl, k)
                logger.info(f"    Val Reward: {vr:.4f}")
                if vr > self.best_val_reward:
                    self.best_val_reward = vr; pat = 0
                    self._save_ckpt(ep+1, vl, True)
                else:
                    pat += 1; self._save_ckpt(ep+1, vl, False)
                if pat >= early_stopping_patience:
                    logger.info(f"    Early stop at {ep+1}"); break
        logger.info(f"  Done. Best reward: {self.best_val_reward:.4f}")

    def _save_ckpt(self, ep, vl, best):
        tag = f"checkpoint_best_epoch_{ep}" if best else f"checkpoint_epoch_{ep}"
        self.path_ranker.save_pretrained(os.path.join(self.checkpoint_dir, tag))


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def generate_selected_json(tst_dl, output_dir, trainer, top_k):
    os.makedirs(output_dir, exist_ok=True)
    results = []; trainer.path_ranker.eval(); cnt = 0
    for i, b in enumerate(tqdm(tst_dl, desc="  Inference", leave=False)):
        question = b['question'][0]
        paths = [p[0] for p in b['topk_linearized_triplets']]
        gt = [p[0] for p in b["answer"]]
        if not gt or not paths: continue
        if len(paths) >= top_k:
            qe = b["question_embedding"].to(device)
            te = b["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
            re = b["topK_rel_embeddings"].squeeze(0).to(device)
            gf = b["graph_features"].squeeze(0).to(device)
            scores, probs = trainer.path_ranker(qe, te, re, gf)
            sp, sprobs, _, _ = trainer.path_ranker.sample_paths(probs, paths, top_k, scores)
            si = torch.argsort(sprobs, descending=True)
            sorted_p = [sp[j] for j in si.tolist()]
            results.append({"question": question, "answer": gt, "a_entity": [p[0] for p in b["a_entity"]], "reranker": sorted_p})
        else:
            results.append({"question": question, "answer": gt, "a_entity": [p[0] for p in b["a_entity"]], "reranker": paths}); cnt+=1
    with open(os.path.join(output_dir, 'selected_triplets.json'), "w") as f:
        json.dump(results, f)
    logger.info(f"  Saved {len(results)} samples ({cnt} < top_k)")


# ============================================================================
# CONFIG & MAIN
# ============================================================================

MODEL_ABLATION_CONFIGS = {
    "no-ppr":  {"model_class": ReversedNoPPR,  "description": "Reversed without PPR scores"},
    "no-rt":   {"model_class": ReversedNoRT,   "description": "Reversed without Relation Tower"},
    "no-tt":   {"model_class": ReversedNoTT,   "description": "Reversed without Triplet Tower"},
    "no-gate": {"model_class": ReversedNoGate, "description": "Reversed without Gate"},
    "no-ra":   {"model_class": ReversedNoRA,   "description": "Reversed without Relation Attention path"},
    "no-ta":   {"model_class": ReversedNoTA,   "description": "Reversed without Triplet Attention path"},
}


def _find_last_checkpoint(train_dir):
    if not os.path.exists(train_dir): return None
    ckpts = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and "checkpoint" in d]
    if not ckpts: return None
    def ep(p):
        m = re.search(r'epoch_(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(ckpts, key=ep, reverse=True)[0]


def run_model_ablation(train_path, val_path, test_path, output_base="./results/ablation-2/model-ablation", experiments=None):
    logger.info("=" * 70)
    logger.info("ABLATION-2 MODEL ARCHITECTURE STUDIES (Reversed Attention)")
    logger.info("=" * 70)
    configs = experiments if experiments else list(MODEL_ABLATION_CONFIGS.keys())

    # Phase 1: Train
    logger.info("PHASE 1: Training")
    train_data = torch.load(train_path, weights_only=False, map_location="cpu")
    val_data = torch.load(val_path, weights_only=False, map_location="cpu")
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    for name in configs:
        cfg = MODEL_ABLATION_CONFIGS[name]
        logger.info(f"{'='*60}\n  {name}: {cfg['description']}\n{'='*60}")
        exp_dir = os.path.join(output_base, name)
        pt_dir = os.path.join(exp_dir, "model", "pretrained")
        tr_dir = os.path.join(exp_dir, "model", "trained")
        os.makedirs(exp_dir, exist_ok=True)

        model = cfg["model_class"](device=str(device))
        # Pretrain
        logger.info(f"  [1/2] Pretrain n=500, 5 ep")
        pt_ds = CosinePretrainingDataset(train_data, k=500)
        pv_ds = CosinePretrainingDataset(val_data, k=500)
        pt_dl = DataLoader(pt_ds, batch_size=1, shuffle=True, collate_fn=collate_fn_pretrain)
        pv_dl = DataLoader(pv_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_pretrain)
        CosinePretrainer(model, str(device), pt_dir).train(pt_dl, pv_dl, num_epochs=5)
        ckpt = torch.load(os.path.join(pt_dir, "best_pretrained_model-5.pt"), weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        # Train
        logger.info(f"  [2/2] Train k=1000, sample=100, 30 ep")
        tr_ds = SampledJointTrainingDataset(train_data, k=1000)
        vl_ds = SampledJointTrainingDataset(val_data, k=1000)
        tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True)
        vl_dl = DataLoader(vl_ds, batch_size=1, shuffle=False)
        JointTrainer(model, compute_reward_v8, checkpoint_dir=tr_dir).train(tr_dl, vl_dl, k=100)
        logger.info(f"  Saved: {tr_dir}")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    del train_data, val_data; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Phase 2: Inference
    logger.info(f"{'='*70}\nPHASE 2: Inference\n{'='*70}")
    test_data = torch.load(test_path, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)}")

    for name in configs:
        cfg = MODEL_ABLATION_CONFIGS[name]
        tr_dir = os.path.join(output_base, name, "model", "trained")
        res_dir = os.path.join(output_base, name, "triplet-result")
        ckpt_dir = _find_last_checkpoint(tr_dir)
        if not ckpt_dir:
            logger.warning(f"  {name}: no checkpoint, skip"); continue
        ckpt_path = os.path.join(ckpt_dir, "path_ranker.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  {name}: no path_ranker.pt, skip"); continue
        logger.info(f"  {name}: loading {ckpt_dir}")
        model = cfg["model_class"](device=str(device))
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        for key, val in state.items():
            if key in ('temperature', 'baseline'): getattr(model, key).data = val
            elif hasattr(model, key): getattr(model, key).load_state_dict(val)
            else: logger.warning(f"  Unexpected key: '{key}'")
        model.to(device)
        trainer = JointTrainer(model, compute_reward_v8, checkpoint_dir=tr_dir)
        tst_ds = SampledJointTrainingDataset(test_data, k=1000)
        tst_dl = DataLoader(tst_ds, batch_size=1, shuffle=False)
        generate_selected_json(tst_dl, res_dir, trainer, top_k=100)
        del model, trainer; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("MODEL ABLATION COMPLETE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", default="./results/ablation-2/model-ablation")
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_model_ablation(args.train_data, args.val_data, args.test_data, args.output_dir, args.experiments)
