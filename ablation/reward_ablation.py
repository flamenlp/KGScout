"""
Reward Function Ablation Studies for KGScout.

Runs 6 ablation experiments on the reward function components:
1. no_pres:   Remove w_pres * frac_presence
2. no_conn:   Remove w_conn * conn_score
3. no_path:   Remove w_cov * path_cov
4. only_pres: Only frac_presence (no weights)
5. only_conn: Only conn_score (no weights)
6. only_cov:  Only path_cov (no weights)

Uses the ORIGINAL full model architecture for all experiments.
Each experiment: pretrain (n=500, 5 epochs) -> train (k=1000, n=100, 30 epochs) -> inference (k=1000, top_k=100)
Results saved to ./results/reward-ablation/<variant>/
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Required for torch.load() to unpickle datasets saved from notebooks
# The .pt files were saved with the class in __main__, so we register it here
from preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# Set seeds
torch.manual_seed(100)
np.random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level logger (inherits from root 'ablation' logger configured in run_ablation.py)
logger = logging.getLogger("ablation.reward")

# Set seeds
torch.manual_seed(100)
np.random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# REWARD FUNCTION VARIANTS
# ============================================================================

def _compute_reward_components(triplets, q_entities, a_entities, lambda_lin=0.2):
    """Compute individual reward components (shared helper)."""
    if not triplets:
        return 0.0, 0.0, 0.0

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

    # 3. Path Coverage
    triplet_pairs = {(s.lower(), o.lower()) for s, _, o in triplets}
    cov_scores = []
    for q in q_entities:
        for a in a_entities:
            qn, an = q.lower(), a.lower()
            try:
                path = nx.shortest_path(G, qn, an)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) < 2:
                continue
            matches = sum(1 for u, v in zip(path, path[1:]) if (u, v) in triplet_pairs)
            cov_scores.append(matches / (len(path) - 1))
    path_cov = max(cov_scores) if cov_scores else 0.0

    return frac_presence, conn_score, path_cov


def reward_no_pres(triplets, q_entities, a_entities):
    """Reward WITHOUT w_pres * frac_presence."""
    frac_presence, conn_score, path_cov = _compute_reward_components(triplets, q_entities, a_entities)
    # Only conn + path_cov
    w_conn, w_cov = 4, 3
    total = w_conn * conn_score + w_cov * path_cov
    return min(total, 10.0)


def reward_no_conn(triplets, q_entities, a_entities):
    """Reward WITHOUT w_conn * conn_score."""
    frac_presence, conn_score, path_cov = _compute_reward_components(triplets, q_entities, a_entities)
    # Only pres + path_cov
    w_pres, w_cov = 3, 3
    total = w_pres * frac_presence + w_cov * path_cov
    return min(total, 10.0)


def reward_no_path(triplets, q_entities, a_entities):
    """Reward WITHOUT w_cov * path_cov."""
    frac_presence, conn_score, path_cov = _compute_reward_components(triplets, q_entities, a_entities)
    # Only pres + conn
    w_pres, w_conn = 3, 4
    total = w_pres * frac_presence + w_conn * conn_score
    return min(total, 10.0)


def reward_only_pres(triplets, q_entities, a_entities):
    """Reward with ONLY frac_presence (unweighted)."""
    frac_presence, _, _ = _compute_reward_components(triplets, q_entities, a_entities)
    return frac_presence


def reward_only_conn(triplets, q_entities, a_entities):
    """Reward with ONLY conn_score (unweighted)."""
    _, conn_score, _ = _compute_reward_components(triplets, q_entities, a_entities)
    return conn_score


def reward_only_cov(triplets, q_entities, a_entities):
    """Reward with ONLY path_cov (unweighted)."""
    _, _, path_cov = _compute_reward_components(triplets, q_entities, a_entities)
    return path_cov


# ============================================================================
# ORIGINAL MODEL (used for all reward ablations)
# ============================================================================

class PathRankingModelOriginal(nn.Module):
    """Original full model with PPR + Triplet Tower + Relation Tower + Gate."""
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.question_relation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1), nn.Sigmoid())
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        num_triplets = triplet_embeds.size(0)
        question_embed = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        triplet_attended, _ = self.question_triplet_attention(triplet_embeds, question_embed, question_embed)
        relation_attended, _ = self.question_relation_attention(relation_embeds, question_embed, question_embed)
        question_expanded = question_embed.expand(num_triplets, -1)
        gate_input = torch.cat([question_expanded, triplet_embeds, relation_embeds], dim=-1)
        path_gates = self.gate_network(gate_input).squeeze(-1)
        triplet_centric_input = torch.cat([triplet_embeds, triplet_attended, question_expanded, graph_scores], dim=-1)
        tower_A_scores = self.triplet_mlp(triplet_centric_input).squeeze(-1)
        relation_centric_input = torch.cat([relation_embeds, relation_attended, question_expanded, graph_scores], dim=-1)
        tower_B_scores = self.relation_mlp(relation_centric_input).squeeze(-1)
        combiner_input = torch.stack([tower_A_scores, tower_B_scores, path_gates], dim=-1)
        combined_scores = self.combiner_mlp(combiner_input).squeeze(-1)
        temp = self.temperature.clamp(min=0.1, max=5.0)
        path_probs = F.softmax(combined_scores / temp, dim=0)
        return combined_scores, path_probs

    def sample_paths(self, probabilities, paths, k, ranking_scores):
        return _sample_paths_impl(probabilities, paths, k, ranking_scores)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save({
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            "gate_network": self.gate_network.state_dict(),
            "triplet_mlp": self.triplet_mlp.state_dict(),
            "relation_mlp": self.relation_mlp.state_dict(),
            "combiner_mlp": self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu()
        }, os.path.join(save_directory, "path_ranker.pt"))


def _sample_paths_impl(probabilities, paths, k, ranking_scores):
    """Shared sample_paths implementation."""
    if len(paths) <= k:
        log_probs = torch.log(probabilities + 1e-10)
        return paths, probabilities, ranking_scores, log_probs
    selected_indices = []
    log_probs_list = []
    remaining_indices = torch.ones(len(probabilities), dtype=torch.bool, device=probabilities.device)
    for _ in range(min(k, len(paths))):
        masked_probs = probabilities * remaining_indices.float()
        masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
        masked_dist = torch.distributions.Categorical(probs=masked_probs)
        idx = masked_dist.sample()
        log_prob = masked_dist.log_prob(idx)
        selected_indices.append(idx.item())
        log_probs_list.append(log_prob)
        remaining_indices[idx] = False
    selected_indices_tensor = torch.tensor(selected_indices, device=probabilities.device)
    log_probs = torch.stack(log_probs_list)
    selected_paths = [paths[i] for i in selected_indices]
    selected_probs = probabilities[selected_indices_tensor]
    selected_ranking_scores = ranking_scores[selected_indices_tensor]
    return selected_paths, selected_probs, selected_ranking_scores, log_probs


# ============================================================================
# DATASET CLASSES
# ============================================================================

class SampledJointTrainingDataset(Dataset):
    """Wraps a dataset and limits triplets to k per sample."""
    def __init__(self, dataset, k=500):
        self.dataset = dataset
        self.k = k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_available = data["topk_linearized_triplet_embeddings"].shape[0]
        use_nums = min(self.k, num_available)
        return {
            "question": data["question"],
            "is_empty": data["is_empty"],
            "q_entity": data["q_entity"],
            "a_entity": data["a_entity"],
            "answer": data["answer"],
            "question_embedding": data["question_embedding"],
            "topk_linearized_triplets": data["topk_linearized_triplets"][:use_nums],
            "topk_linearized_triplet_embeddings": data["topk_linearized_triplet_embeddings"][:use_nums],
            "topk_rel_data": data["topk_rel_data"][:use_nums],
            "topK_rel_embeddings": data["topK_rel_embeddings"][:use_nums],
            "graph_features": data["graph_features"][:use_nums]
        }


class CosinePretrainingDataset(Dataset):
    """Dataset for pretraining with cosine similarity targets."""
    def __init__(self, original_dataset, k=500):
        self.original_dataset = original_dataset
        self.k = k

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data = self.original_dataset[idx]
        if len(data["topk_linearized_triplets"]) == 0 or len(data["q_entity"]) == 0:
            return None
        num_available = data["topk_linearized_triplet_embeddings"].shape[0]
        use_nums = min(self.k, num_available)
        path_embeddings = data["topk_linearized_triplet_embeddings"][:use_nums]
        rel_embeddings = data["topK_rel_embeddings"][:use_nums]
        decay_rate = 0.01
        cosine_targets = torch.exp(-decay_rate * torch.arange(use_nums, dtype=torch.float))
        return {
            "question_embedding": data["question_embedding"],
            "path_embeddings": path_embeddings,
            "rel_embeddings": rel_embeddings,
            "cosine_targets": cosine_targets,
            "question": data["question"],
            "paths": data["topk_linearized_triplets"][:use_nums],
            "num_paths": use_nums,
            "graph_features": data["graph_features"][:use_nums]
        }


def collate_fn_pretrain(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch[0]


# ============================================================================
# PRETRAINER
# ============================================================================

class CosinePretrainer:
    """Pretrains the path ranker using cosine similarity ordering."""
    def __init__(self, path_ranker, device="cuda", checkpoint_dir="pretrain_checkpoints"):
        self.device = device
        self.path_ranker = path_ranker.to(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)

    def compute_ranking_loss(self, predicted_scores, target_scores, sample_pairs=100):
        batch_size = predicted_scores.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        all_pairs = [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)]
        max_pairs = min(sample_pairs, len(all_pairs))
        if len(all_pairs) > max_pairs:
            sampled = np.random.choice(len(all_pairs), max_pairs, replace=False)
            pairs = [all_pairs[i] for i in sampled]
        else:
            pairs = all_pairs
        if not pairs:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        pairs_i = torch.tensor([p[0] for p in pairs], device=self.device)
        pairs_j = torch.tensor([p[1] for p in pairs], device=self.device)
        targets = torch.sign(target_scores[pairs_i] - target_scores[pairs_j])
        return self.ranking_loss(predicted_scores[pairs_i], predicted_scores[pairs_j], targets)

    def train_step(self, batch):
        if batch is None:
            return None
        question_embed = batch["question_embedding"].to(self.device)
        path_embeds = batch["path_embeddings"].to(self.device)
        rel_embeds = batch["rel_embeddings"].to(self.device)
        cosine_targets = batch["cosine_targets"].to(self.device)
        graph_features = batch["graph_features"].to(self.device)
        predicted_scores, _ = self.path_ranker(question_embed.unsqueeze(0), path_embeds, rel_embeds, graph_features)
        mse = self.mse_loss(predicted_scores, cosine_targets)
        ranking = self.compute_ranking_loss(predicted_scores, cosine_targets)
        return 0.5 * mse + 0.5 * ranking

    def train(self, train_dataloader, val_dataloader=None, num_epochs=5,
              learning_rate=1e-4, weight_decay=1e-5, gradient_accumulation_steps=8):
        optimizer = torch.optim.AdamW(self.path_ranker.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"  Pretrain Epoch {epoch + 1}/{num_epochs}")
            self.path_ranker.train()
            epoch_loss = 0
            valid_batches = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="  Pretraining", leave=False)):
                loss = self.train_step(batch)
                if loss is None:
                    continue
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                epoch_loss += loss.item()
                valid_batches += 1
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if valid_batches % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / max(valid_batches, 1)
            logger.info(f"    Train Loss: {avg_loss:.4f}")

            if val_dataloader:
                self.path_ranker.eval()
                val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        l = self.train_step(batch)
                        if l is not None:
                            val_loss += l.item()
                            val_count += 1
                avg_val = val_loss / max(val_count, 1)
                logger.info(f"    Val Loss: {avg_val:.4f}")
                scheduler.step(avg_val)
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    self._save_checkpoint(epoch + 1)

        self._save_checkpoint(num_epochs)
        logger.info(f"  Pretraining complete. Checkpoint: {self.checkpoint_dir}")

    def _save_checkpoint(self, epoch):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.path_ranker.state_dict()}
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'best_pretrained_model-{epoch}.pt'))


# ============================================================================
# JOINT TRAINER (REINFORCE)
# ============================================================================

class JointTrainer:
    """Trains the path ranker using REINFORCE with a configurable reward function."""
    def __init__(self, path_ranker, reward_func, max_grad_norm=1.0,
                 gradient_accumulation_steps=32, checkpoint_dir="checkpoints",
                 gamma=0.99, baseline_decay=0.9):
        self.reward_func = reward_func
        self.path_ranker = path_ranker.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self.running_baseline = 0
        self.reward_buffer = []
        self.best_val_reward = float('-inf')

    def compute_reinforce_loss(self, log_probs, rewards, baseline):
        advantages = rewards - baseline
        return -(log_probs * advantages.detach()).mean()

    def update_baseline_with_buffer(self):
        if len(self.reward_buffer) > 0:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
            if self.running_baseline == 0:
                self.running_baseline = avg_reward * 0.8
            else:
                error = avg_reward - self.running_baseline
                if abs(error) > 0.5:
                    self.running_baseline += 0.1 * error
            max_reasonable_baseline = avg_reward * 0.9
            self.running_baseline = min(self.running_baseline, max_reasonable_baseline)
            self.path_ranker.baseline.data = torch.tensor([self.running_baseline], device=self.device)
            self.reward_buffer = []

    def train_step(self, batch, k=10):
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        q_entity = [p[0] for p in batch['q_entity']]
        a_entity = [p[0] for p in batch['a_entity']]
        triplets = [(data[1][0][0], data[1][1][0], data[1][2][0]) for data in batch["topk_rel_data"]]
        ques_embed = batch["question_embedding"].to(self.device)
        linearized_triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(self.device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(self.device)
        graph_features = batch["graph_features"].squeeze(0).to(self.device)

        if len(q_entity) == 0:
            return None, None
        ranking_scores, path_probs = self.path_ranker(ques_embed, linearized_triplet_embeds, relation_embeds, graph_features)
        selected_triplets, selected_probs, selected_ranking_scores, log_probs = self.path_ranker.sample_paths(
            path_probs, triplets, k, ranking_scores)
        answer_reward = self.reward_func(selected_triplets, q_entity, a_entity)
        if answer_reward is None:
            return None, None
        reward = torch.tensor([answer_reward], device=self.device)
        self.reward_buffer.append(reward.item())
        reinforcement_loss = self.compute_reinforce_loss(
            log_probs, reward.expand(log_probs.size(0)),
            torch.tensor([self.running_baseline], device=self.device))
        return reinforcement_loss, reward

    @torch.no_grad()
    def validate(self, val_dataloader, k=10):
        self.path_ranker.eval()
        total_loss = 0
        total_reward = 0
        valid_samples = 0
        for batch in tqdm(val_dataloader, desc="  Validation", leave=False):
            loss, reward = self.train_step(batch, k)
            if loss is None:
                continue
            total_loss += loss.item()
            total_reward += reward.item()
            valid_samples += 1
        avg_loss = total_loss / max(valid_samples, 1)
        avg_reward = total_reward / max(valid_samples, 1)
        return avg_loss, avg_reward

    def train(self, train_dataloader, val_dataloader, num_epochs=30,
              learning_rate=1e-4, warmup_steps=100, scheduler_type='cosine',
              validation_interval=1, early_stopping_patience=3, k=30):
        optimizer = torch.optim.AdamW([{'params': self.path_ranker.parameters(), 'lr': learning_rate}])
        total_steps = (len(train_dataloader) * num_epochs) // self.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"  Train Epoch {epoch + 1}/{num_epochs}")
            self.path_ranker.train()
            epoch_rewards = []
            epoch_losses = []
            optimizer.zero_grad()
            valid_batch_count = 0

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"  Training", leave=False)):
                loss, reward = self.train_step(batch, k)
                if loss is None:
                    continue
                if math.isnan(reward.item()):
                    continue
                epoch_rewards.append(reward.item())
                epoch_losses.append(loss.item())
                valid_batch_count += 1
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                if valid_batch_count % self.gradient_accumulation_steps == 0:
                    self.update_baseline_with_buffer()
                    torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if valid_batch_count % self.gradient_accumulation_steps != 0:
                self.update_baseline_with_buffer()
                torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            logger.info(f"    Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % validation_interval == 0 and val_dataloader is not None:
                val_loss, val_reward = self.validate(val_dataloader, k)
                logger.info(f"    Val Reward: {val_reward:.4f}, Val Loss: {val_loss:.4f}")
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    patience_counter = 0
                    self._save_checkpoint(epoch + 1, val_loss, is_best=True)
                else:
                    patience_counter += 1
                    self._save_checkpoint(epoch + 1, val_loss, is_best=False)
                if patience_counter >= early_stopping_patience:
                    logger.info(f"    Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"  Training complete. Best val reward: {self.best_val_reward:.4f}")

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        if is_best:
            save_dir = os.path.join(self.checkpoint_dir, f"checkpoint_best_epoch_{epoch}")
        else:
            save_dir = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        self.path_ranker.save_pretrained(save_dir)
        torch.save({'epoch': epoch, 'val_loss': val_loss, 'best_val_reward': self.best_val_reward},
                   os.path.join(save_dir, "training_state.pt"))


# ============================================================================
# INFERENCE: generate_selected_json
# ============================================================================

@torch.no_grad()
def generate_selected_json(tst_dataloader, output_dir, trainer, top_k):
    """Generate selected triplets JSON for evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    detailed_results_file = os.path.join(output_dir, 'selected_triplets.json')
    results = []
    trainer.path_ranker.eval()
    count = 0
    for i, batch in enumerate(tqdm(tst_dataloader, desc="  Inference", leave=False)):
        question = batch['question'][0]
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        ground_truth = [p[0] for p in batch["answer"]]
        if not ground_truth or len(paths) == 0:
            continue
        if len(paths) >= top_k:
            ques_embed = batch["question_embedding"].to(device)
            linearized_triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
            relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
            graph_features = batch["graph_features"].squeeze(0).to(device)
            ranking_scores, path_probs = trainer.path_ranker(
                ques_embed, linearized_triplet_embeds, relation_embeds, graph_features)
            selected_paths, selected_probs, _, _ = trainer.path_ranker.sample_paths(
                path_probs, paths, top_k, ranking_scores)
            sorted_indices = torch.argsort(selected_probs, descending=True)
            sorted_paths = [selected_paths[i] for i in sorted_indices.tolist()]
            detail_data = {
                "question": question,
                "answer": ground_truth,
                "a_entity": [p[0] for p in batch["a_entity"]],
                "reranker": sorted_paths,
            }
        else:
            detail_data = {
                "question": question,
                "answer": ground_truth,
                "a_entity": [p[0] for p in batch["a_entity"]],
                "reranker": paths,
            }
            count += 1
        results.append(detail_data)
    with open(detailed_results_file, "w") as f:
        json.dump(results, f)
    logger.info(f"  Inference complete. {len(results)} samples saved. ({count} had < top_k paths)")
    return count


# ============================================================================
# MAIN: RUN REWARD ABLATION EXPERIMENTS
# ============================================================================

REWARD_ABLATION_CONFIGS = {
    "no_pres": {
        "reward_func": reward_no_pres,
        "description": "Reward without w_pres * frac_presence"
    },
    "no_conn": {
        "reward_func": reward_no_conn,
        "description": "Reward without w_conn * conn_score"
    },
    "no_path": {
        "reward_func": reward_no_path,
        "description": "Reward without w_cov * path_cov"
    },
    "only_pres": {
        "reward_func": reward_only_pres,
        "description": "Reward with only frac_presence"
    },
    "only_conn": {
        "reward_func": reward_only_conn,
        "description": "Reward with only conn_score"
    },
    "only_cov": {
        "reward_func": reward_only_cov,
        "description": "Reward with only path_cov"
    },
}


def run_reward_ablation(train_dataset_path: str, val_dataset_path: str,
                        test_dataset_path: str, output_base_dir: str = "./results/reward-ablation",
                        experiments: Optional[List[str]] = None):
    """
    Run all reward function ablation experiments.
    Phase 1: Train all models (pretrain + REINFORCE train)
    Phase 2: Load test data once, then load each trained model and run inference.

    Args:
        train_dataset_path: Path to training dataset (.pt file)
        val_dataset_path: Path to validation dataset (.pt file)
        test_dataset_path: Path to test dataset (.pt file)
        output_base_dir: Base directory for results
        experiments: List of experiment names to run (None = all)
    """
    logger.info("=" * 70)
    logger.info("REWARD FUNCTION ABLATION STUDIES")
    logger.info("=" * 70)

    configs_to_run = experiments if experiments else list(REWARD_ABLATION_CONFIGS.keys())

    # ==================================================================
    # PHASE 1: Train all models
    # ==================================================================
    logger.info("PHASE 1: Training all reward variants")
    logger.info("Loading train/val datasets...")
    train_data = torch.load(train_dataset_path, weights_only=False, map_location="cpu")
    val_data = torch.load(val_dataset_path, weights_only=False, map_location="cpu")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")

    for exp_name in configs_to_run:
        config = REWARD_ABLATION_CONFIGS[exp_name]
        logger.info(f"{'='*70}")
        logger.info(f"EXPERIMENT: {exp_name} - {config['description']}")
        logger.info(f"{'='*70}")

        exp_dir = os.path.join(output_base_dir, exp_name)
        model_dir = os.path.join(exp_dir, "model")
        pretrain_dir = os.path.join(model_dir, "pretrained")
        train_dir = os.path.join(model_dir, "trained")
        os.makedirs(exp_dir, exist_ok=True)

        # --- Step 1: Pretrain with n=500 ---
        logger.info(f"[1/2] Pretraining (n=500, 5 epochs)...")
        # Fresh model initialization - ensures no weight leakage from previous experiment
        model = PathRankingModelOriginal(device=str(device))
        pretrain_dataset = CosinePretrainingDataset(train_data, k=500)
        pretrain_val_dataset = CosinePretrainingDataset(val_data, k=500)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_pretrain)
        pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_pretrain)

        pretrainer = CosinePretrainer(model, device=str(device), checkpoint_dir=pretrain_dir)
        pretrainer.train(pretrain_loader, pretrain_val_loader, num_epochs=5)

        # Load best pretrained weights
        pretrained_ckpt = torch.load(os.path.join(pretrain_dir, "best_pretrained_model-5.pt"), weights_only=False, map_location="cpu")
        model.load_state_dict(pretrained_ckpt["model_state_dict"])

        # --- Step 2: Train with ablated reward function ---
        logger.info(f"[2/2] Training with reward variant: {exp_name} (k=1000, sample k=100, 30 epochs)...")
        train_sampled = SampledJointTrainingDataset(train_data, k=1000)
        val_sampled = SampledJointTrainingDataset(val_data, k=1000)
        train_loader = DataLoader(train_sampled, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sampled, batch_size=1, shuffle=False)

        trainer = JointTrainer(
            model, config["reward_func"],
            max_grad_norm=1.0, gradient_accumulation_steps=32,
            checkpoint_dir=train_dir, gamma=0.99, baseline_decay=0.9)
        trainer.train(
            train_loader, val_loader,
            num_epochs=30, learning_rate=1e-4, warmup_steps=100,
            scheduler_type='cosine', validation_interval=1,
            early_stopping_patience=10, k=100)

        logger.info(f"  Model trained and saved to: {train_dir}")

        # --- Cleanup: offload model from GPU before next experiment ---
        del trainer, pretrainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info(f"  GPU memory released for next experiment.")

    # Free train/val data from memory before loading test data
    del train_data, val_data
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================================================================
    # PHASE 2: Inference - load test data once, then load each model
    # ==================================================================
    logger.info(f"{'='*70}")
    logger.info("PHASE 2: Running inference on test data")
    logger.info(f"{'='*70}")
    logger.info("Loading test dataset...")
    test_data = torch.load(test_dataset_path, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)} samples")

    for exp_name in configs_to_run:
        config = REWARD_ABLATION_CONFIGS[exp_name]
        logger.info(f"  Inference for: {exp_name}")

        exp_dir = os.path.join(output_base_dir, exp_name)
        train_dir = os.path.join(exp_dir, "model", "trained")
        result_dir = os.path.join(exp_dir, "triplet-result")

        # Find the last checkpoint (epoch 30, or highest available)
        checkpoint_dir = _find_last_checkpoint(train_dir)
        if checkpoint_dir is None:
            logger.warning(f"  No checkpoint found in {train_dir}, skipping inference.")
            continue

        logger.info(f"  Loading checkpoint: {checkpoint_dir}")

        # Reload model from checkpoint (all reward ablations use original architecture)
        model = PathRankingModelOriginal(device=str(device))
        ckpt_path = os.path.join(checkpoint_dir, "path_ranker.pt")
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        # The checkpoint is a dict of sub-module state dicts (not a flat model state_dict)
        for key, value in state.items():
            if key in ('temperature', 'baseline'):
                getattr(model, key).data = value
            elif hasattr(model, key):
                getattr(model, key).load_state_dict(value)
            else:
                logger.warning(f"  Unexpected key in checkpoint: '{key}' (model has no such attribute)")
        model.to(device)

        # Create a minimal trainer wrapper for inference
        trainer = JointTrainer(
            model, config["reward_func"],
            checkpoint_dir=train_dir)

        # Run inference
        test_sampled = SampledJointTrainingDataset(test_data, k=1000)
        test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)
        generate_selected_json(test_loader, result_dir, trainer, top_k=100)

        logger.info(f"  Results saved to: {result_dir}")

        # Cleanup
        del trainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"{'='*70}")
    logger.info("ALL REWARD ABLATION EXPERIMENTS COMPLETE")
    logger.info(f"{'='*70}")


def _find_last_checkpoint(train_dir: str) -> Optional[str]:
    """Find the last epoch checkpoint directory (prefers highest epoch number)."""
    if not os.path.exists(train_dir):
        return None
    checkpoints = []
    for d in os.listdir(train_dir):
        full_path = os.path.join(train_dir, d)
        if os.path.isdir(full_path) and "checkpoint" in d:
            checkpoints.append(full_path)
    if not checkpoints:
        return None
    import re
    def get_epoch(path):
        name = os.path.basename(path)
        match = re.search(r'epoch_(\d+)', name)
        return int(match.group(1)) if match else 0
    checkpoints.sort(key=get_epoch, reverse=True)
    return checkpoints[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run reward function ablation studies")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset .pt file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val dataset .pt file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset .pt file")
    parser.add_argument("--output_dir", type=str, default="./results/reward-ablation", help="Output directory")
    parser.add_argument("--experiments", nargs="+", default=None,
                        choices=["no_pres", "no_conn", "no_path", "only_pres", "only_conn", "only_cov"],
                        help="Specific experiments to run (default: all)")
    args = parser.parse_args()
    run_reward_ablation(args.train_data, args.val_data, args.test_data, args.output_dir, args.experiments)
