"""
Model Architecture Ablation Variants for Reversed Attention.

Base: PathRankingModel from src/model/path_ranker.py
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
"""

import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SHARED SAMPLE_PATHS
# ============================================================================

def _sample_paths_impl(probabilities, paths, k, ranking_scores):
    """Sample k paths using categorical sampling without replacement (REINFORCE)."""
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

class ReversedNoPPR(nn.Module):
    """Reversed model without PPR graph_scores (tower input: 3*d instead of 3*d+2)."""

    REQUIRED_COMPONENTS = [
        'question_triplet_attention', 'question_relation_attention',
        'gate_network', 'triplet_mlp', 'relation_mlp', 'combiner_mlp',
        'temperature', 'baseline'
    ]

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
        # No graph_scores in tower inputs
        s_a = self.triplet_mlp(torch.cat([triplet_embeds, t_att, q_exp], dim=-1)).squeeze(-1)
        s_b = self.relation_mlp(torch.cat([relation_embeds, r_att, q_exp], dim=-1)).squeeze(-1)
        scores = self.combiner_mlp(torch.stack([s_a, s_b, gated_w], dim=-1)).squeeze(-1)
        return scores, F.softmax(scores / self.temperature.clamp(0.1, 5.0), dim=0)

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
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoPPR checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoPPR checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoPPR checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_triplet_attention.load_state_dict(state['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(state['question_relation_attention'])
        model.gate_network.load_state_dict(state['gate_network'])
        model.triplet_mlp.load_state_dict(state['triplet_mlp'])
        model.relation_mlp.load_state_dict(state['relation_mlp'])
        model.combiner_mlp.load_state_dict(state['combiner_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model


class ReversedNoRT(nn.Module):
    """Reversed model without Relation Tower. Combiner: [tower_A, gated_weights]."""

    REQUIRED_COMPONENTS = [
        'question_triplet_attention', 'question_relation_attention',
        'gate_network', 'triplet_mlp', 'combiner_mlp',
        'temperature', 'baseline'
    ]

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

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            'gate_network': self.gate_network.state_dict(),
            'triplet_mlp': self.triplet_mlp.state_dict(),
            'combiner_mlp': self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoRT checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoRT checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoRT checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_triplet_attention.load_state_dict(state['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(state['question_relation_attention'])
        model.gate_network.load_state_dict(state['gate_network'])
        model.triplet_mlp.load_state_dict(state['triplet_mlp'])
        model.combiner_mlp.load_state_dict(state['combiner_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model


class ReversedNoTT(nn.Module):
    """Reversed model without Triplet Tower. Combiner: [tower_B, gated_weights]."""

    REQUIRED_COMPONENTS = [
        'question_triplet_attention', 'question_relation_attention',
        'gate_network', 'relation_mlp', 'combiner_mlp',
        'temperature', 'baseline'
    ]

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

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            'gate_network': self.gate_network.state_dict(),
            'relation_mlp': self.relation_mlp.state_dict(),
            'combiner_mlp': self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoTT checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoTT checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoTT checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_triplet_attention.load_state_dict(state['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(state['question_relation_attention'])
        model.gate_network.load_state_dict(state['gate_network'])
        model.relation_mlp.load_state_dict(state['relation_mlp'])
        model.combiner_mlp.load_state_dict(state['combiner_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model


class ReversedNoGate(nn.Module):
    """Reversed model without Gate. Combiner: [tower_A, tower_B]. No gated blending."""

    REQUIRED_COMPONENTS = [
        'question_triplet_attention', 'question_relation_attention',
        'triplet_mlp', 'relation_mlp', 'combiner_mlp',
        'temperature', 'baseline'
    ]

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

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            'triplet_mlp': self.triplet_mlp.state_dict(),
            'relation_mlp': self.relation_mlp.state_dict(),
            'combiner_mlp': self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoGate checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoGate checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoGate checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_triplet_attention.load_state_dict(state['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(state['question_relation_attention'])
        model.triplet_mlp.load_state_dict(state['triplet_mlp'])
        model.relation_mlp.load_state_dict(state['relation_mlp'])
        model.combiner_mlp.load_state_dict(state['combiner_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model


class ReversedNoRA(nn.Module):
    """No relation attention/gate/relation tower/combiner. Only triplet attention → triplet tower."""

    REQUIRED_COMPONENTS = [
        'question_triplet_attention', 'triplet_mlp',
        'temperature', 'baseline'
    ]

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

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'triplet_mlp': self.triplet_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoRA checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoRA checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoRA checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_triplet_attention.load_state_dict(state['question_triplet_attention'])
        model.triplet_mlp.load_state_dict(state['triplet_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model


class ReversedNoTA(nn.Module):
    """No triplet attention/gate/triplet tower/combiner. Only relation attention → relation tower."""

    REQUIRED_COMPONENTS = [
        'question_relation_attention', 'relation_mlp',
        'temperature', 'baseline'
    ]

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

    def sample_paths(self, prob, paths, k, scores):
        return _sample_paths_impl(prob, paths, k, scores)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save({
            'question_relation_attention': self.question_relation_attention.state_dict(),
            'relation_mlp': self.relation_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu(),
        }, os.path.join(d, "path_ranker.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ReversedNoTA checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing = [k for k in cls.REQUIRED_COMPONENTS if k not in state]
        unexpected = [k for k in state if k not in cls.REQUIRED_COMPONENTS]
        if missing:
            raise ValueError(f"ReversedNoTA checkpoint missing components: {missing}")
        if unexpected:
            raise ValueError(f"ReversedNoTA checkpoint has unexpected keys: {unexpected}. Wrong model variant?")
        model = cls(hidden_size=384, device=device)
        model.question_relation_attention.load_state_dict(state['question_relation_attention'])
        model.relation_mlp.load_state_dict(state['relation_mlp'])
        model.temperature.data = state['temperature'].to(device)
        model.baseline.data = state['baseline'].to(device)
        model.to(device)
        return model
