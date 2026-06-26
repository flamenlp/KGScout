#!/usr/bin/env python3
"""
Ablation-2 Variant 2: Reversed Attention + Attention Weights in Towers.

Same as run_reversed_attention.py but additionally passes attention weights
(α_T_i, α_R_i) as scalar features to each tower's MLP input.

Tower A input: [T_i ‖ Q_trip ‖ q ‖ G_i ‖ α_T_i]  → (3d + 3)
Tower B input: [R_i ‖ Q_rel ‖ q ‖ G_i ‖ α_R_i]   → (3d + 3)
Combiner:      [s_A, s_B, w_i]                     → MLP_C → ŝ_i
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.training.rewards import compute_reward_v8
from src.utils.metrics import (
    compute_answer_coverage, compute_path_coverage,
    extract_predictions_from_response, compute_hit_score,
    compute_hit_at_1, compute_precision, compute_recall,
    compute_f1_score, should_use_double_check, preprocess_date_answers,
)
from src.utils.llm_inference import load_llm_model, format_prompt, run_llm_inference
from src.training.monitor import TrainingMonitor

# Allow loading datasets saved from notebooks
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

torch.manual_seed(100)
np.random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("ablation2")


# ============================================================================
# MODEL: Reversed Attention PathRankingModel
# ============================================================================

class PathRankingModelReversedAttention(nn.Module):
    """
    Model with REVERSED attention: Query=Question, Key=Value=Triplets/Relations.
    
    Difference from original:
      - Original:  MHA(query=T, key=q, value=q) → N×d (all rows identical)
      - Reversed:  MHA(query=q, key=T, value=T) → 1×d (question attending over triplets)
                   then expanded to N×d for downstream concatenation.
    """
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
            nn.Linear(hidden_size * 3 + 3, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1))
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3, hidden_size),
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

        # REVERSED: Query=question(1,d), Key=Value=triplets(N,d)
        triplet_attended, triplet_weights = self.question_triplet_attention(
            question_embed, triplet_embeds, triplet_embeds
        )  # triplet_attended: (1, d), triplet_weights: (1, N)
        # REVERSED: Query=question(1,d), Key=Value=relations(N,d)
        relation_attended, relation_weights = self.question_relation_attention(
            question_embed, relation_embeds, relation_embeds
        )  # relation_attended: (1, d), relation_weights: (1, N)

        # Extract attention weights as per-triplet relevance scores
        triplet_weights = triplet_weights.squeeze(0).squeeze(0)  # (N,)
        relation_weights = relation_weights.squeeze(0).squeeze(0)  # (N,)

        # Expand attended outputs to (N, d) for per-triplet scoring
        triplet_attended = triplet_attended.expand(num_triplets, -1)
        relation_attended = relation_attended.expand(num_triplets, -1)

        question_expanded = question_embed.expand(num_triplets, -1)
        gate_input = torch.cat([question_expanded, triplet_embeds, relation_embeds], dim=-1)
        path_gates = self.gate_network(gate_input).squeeze(-1)  # σ ∈ [0,1]^N

        # Gated combination of attention weights:
        # w_i = σ_i * triplet_weights_i + (1 - σ_i) * relation_weights_i
        gated_attention_weights = path_gates * triplet_weights + (1 - path_gates) * relation_weights

        triplet_centric_input = torch.cat([
            triplet_embeds, triplet_attended, question_expanded, graph_scores,
            triplet_weights.unsqueeze(-1)], dim=-1)
        tower_A_scores = self.triplet_mlp(triplet_centric_input).squeeze(-1)

        relation_centric_input = torch.cat([
            relation_embeds, relation_attended, question_expanded, graph_scores,
            relation_weights.unsqueeze(-1)], dim=-1)
        tower_B_scores = self.relation_mlp(relation_centric_input).squeeze(-1)

        # Combiner takes 3 signals: tower_A, tower_B, gated_attention_weight
        combiner_input = torch.stack([
            tower_A_scores, tower_B_scores, gated_attention_weights], dim=-1)
        combined_scores = self.combiner_mlp(combiner_input).squeeze(-1)

        temp = self.temperature.clamp(min=0.1, max=5.0)
        path_probs = F.softmax(combined_scores / temp, dim=0)
        return combined_scores, path_probs

    def sample_paths(self, probabilities, paths, k, ranking_scores):
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


# ============================================================================
# DATASETS
# ============================================================================

class SampledDataset(Dataset):
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
            "graph_features": data["graph_features"][:use_nums]
        }


def collate_fn_pretrain(batch):
    batch = [item for item in batch if item is not None]
    return batch[0] if batch else None


# ============================================================================
# PRETRAINER
# ============================================================================

class CosinePretrainer:
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
        predicted_scores, _ = self.path_ranker(
            question_embed.unsqueeze(0), path_embeds, rel_embeds, graph_features)
        mse = self.mse_loss(predicted_scores, cosine_targets)
        ranking = self.compute_ranking_loss(predicted_scores, cosine_targets)
        return 0.5 * mse + 0.5 * ranking

    def train(self, train_dataloader, val_dataloader=None, num_epochs=5,
              learning_rate=1e-4, gradient_accumulation_steps=8):
        optimizer = torch.optim.AdamW(self.path_ranker.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            logger.info(f"  Pretrain Epoch {epoch + 1}/{num_epochs}")
            self.path_ranker.train()
            epoch_loss, valid_batches = 0, 0
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="  Pretraining", leave=False)):
                loss = self.train_step(batch)
                if loss is None:
                    continue
                (loss / gradient_accumulation_steps).backward()
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
                val_loss, val_count = 0, 0
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
        logger.info(f"  Pretraining complete.")

    def _save_checkpoint(self, epoch):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.path_ranker.state_dict()}
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'best_pretrained_model-{epoch}.pt'))


# ============================================================================
# REINFORCE TRAINER
# ============================================================================

class JointTrainer:
    def __init__(self, path_ranker, reward_func, max_grad_norm=1.0,
                 gradient_accumulation_steps=32, checkpoint_dir="checkpoints",
                 baseline_decay=0.9, monitor=None):
        self.reward_func = reward_func
        self.path_ranker = path_ranker.to(device)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.baseline_decay = baseline_decay
        self.running_baseline = 0
        self.reward_buffer = []
        self.best_val_reward = float('-inf')
        self.best_epoch = 0
        self.monitor = monitor

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
            self.running_baseline = min(self.running_baseline, avg_reward * 0.9)
            self.path_ranker.baseline.data = torch.tensor([self.running_baseline], device=device)
            self.reward_buffer = []

    def train_step(self, batch, k=100):
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        q_entity = [p[0] for p in batch['q_entity']]
        a_entity = [p[0] for p in batch['a_entity']]
        triplets = [(d[1][0][0], d[1][1][0], d[1][2][0]) for d in batch["topk_rel_data"]]
        ques_embed = batch["question_embedding"].to(device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
        graph_features = batch["graph_features"].squeeze(0).to(device)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)
        if len(q_entity) == 0:
            return None, None
        ranking_scores, path_probs = self.path_ranker(ques_embed, triplet_embeds, relation_embeds, graph_features)
        selected_triplets, selected_probs, _, log_probs = self.path_ranker.sample_paths(
            path_probs, triplets, k, ranking_scores)
        answer_reward = self.reward_func(selected_triplets, q_entity, a_entity)
        if answer_reward is None:
            return None, None
        reward = torch.tensor([answer_reward], device=device)
        self.reward_buffer.append(reward.item())
        loss = self.compute_reinforce_loss(
            log_probs, reward.expand(log_probs.size(0)),
            torch.tensor([self.running_baseline], device=device))
        return loss, reward

    @torch.no_grad()
    def validate(self, val_dataloader, k=100):
        self.path_ranker.eval()
        total_loss, total_reward, valid_samples = 0, 0, 0
        for batch in tqdm(val_dataloader, desc="  Validation", leave=False):
            loss, reward = self.train_step(batch, k)
            if loss is None:
                continue
            total_loss += loss.item()
            total_reward += reward.item()
            valid_samples += 1
        return (total_loss / max(valid_samples, 1),
                total_reward / max(valid_samples, 1))

    def train(self, train_dataloader, val_dataloader, num_epochs=30,
              learning_rate=1e-4, warmup_steps=100, validation_interval=1,
              early_stopping_patience=3, k=100):
        optimizer = torch.optim.AdamW(self.path_ranker.parameters(), lr=learning_rate)
        total_steps = (len(train_dataloader) * num_epochs) // self.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        patience_counter = 0
        for epoch in range(num_epochs):
            logger.info(f"  Train Epoch {epoch + 1}/{num_epochs}")
            self.path_ranker.train()
            epoch_rewards, epoch_losses = [], []
            optimizer.zero_grad()
            valid_batch_count = 0
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="  Training", leave=False)):
                loss, reward = self.train_step(batch, k)
                if loss is None:
                    continue
                if math.isnan(reward.item()):
                    continue
                epoch_rewards.append(reward.item())
                epoch_losses.append(loss.item())
                valid_batch_count += 1
                (loss / self.gradient_accumulation_steps).backward()
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
                # Log to monitor
                if self.monitor:
                    self.monitor.log_epoch({
                        'train_reward': avg_reward,
                        'train_loss': avg_loss,
                        'val_reward': val_reward,
                        'val_loss': val_loss,
                    }, epoch + 1)
                # Save checkpoint for every epoch
                self._save_checkpoint(epoch + 1, is_best=False)
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                    self._save_checkpoint(epoch + 1, is_best=True)
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"    Early stopping at epoch {epoch + 1}")
                    break
        # Generate plots at end of training
        if self.monitor:
            self.monitor.plot_metrics()
        logger.info(f"  Training complete. Best val reward: {self.best_val_reward:.4f} at epoch {self.best_epoch}")

    def _save_checkpoint(self, epoch, is_best=False):
        if is_best:
            save_dir = os.path.join(self.checkpoint_dir, f"complete_{epoch}_best")
        else:
            save_dir = os.path.join(self.checkpoint_dir, f"complete_{epoch}")
        self.path_ranker.save_pretrained(save_dir)


# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

@torch.no_grad()
def run_coverage_analysis(test_data, model, top_k, output_dir, model_path=None):
    """Compute answer coverage and path coverage on test set."""
    logger.info("Running coverage analysis...")
    model.eval()
    test_sampled = SampledDataset(test_data, k=1000)
    test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)

    answer_cov_count, path_cov_count, total = 0, 0, 0
    for batch in tqdm(test_loader, desc="  Coverage", leave=False):
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        q_entity = [p[0] for p in batch['q_entity']]
        a_entity = [p[0] for p in batch['a_entity']]
        triplets_structured = [(d[1][0][0], d[1][1][0], d[1][2][0]) for d in batch["topk_rel_data"]]
        if len(paths) == 0 or len(q_entity) == 0:
            continue
        ques_embed = batch["question_embedding"].to(device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
        graph_features = batch["graph_features"].squeeze(0).to(device)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)

        ranking_scores, path_probs = model(ques_embed, triplet_embeds, relation_embeds, graph_features)
        k = min(top_k, len(triplets_structured))
        selected_triplets, selected_probs, _, _ = model.sample_paths(
            path_probs, triplets_structured, k, ranking_scores)

        if compute_answer_coverage(selected_triplets, a_entity):
            answer_cov_count += 1
        if compute_path_coverage(selected_triplets, q_entity, a_entity):
            path_cov_count += 1
        total += 1

    ans_cov = answer_cov_count / total if total > 0 else 0
    path_cov = path_cov_count / total if total > 0 else 0
    logger.info(f"  Answer Coverage: {ans_cov:.4f} ({answer_cov_count}/{total})")
    logger.info(f"  Path Coverage:   {path_cov:.4f} ({path_cov_count}/{total})")

    results = {"answer_coverage": ans_cov, "path_coverage": path_cov,
               "answer_coverage_count": answer_cov_count,
               "path_coverage_count": path_cov_count, "total": total,
               "model_path": model_path}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "coverage_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================================
# LLM EVALUATION
# ============================================================================

@torch.no_grad()
def run_llm_evaluation(test_data, model, top_k, output_dir, llm_model_name="llama", model_path=None):
    """Run LLM inference and compute QA metrics."""
    logger.info(f"Running LLM evaluation with {llm_model_name}...")
    model.eval()

    logger.info("  Loading LLM...")
    llm_model, tokenizer = load_llm_model(llm_model_name, str(device))

    test_sampled = SampledDataset(test_data, k=1000)
    test_loader = DataLoader(test_sampled, batch_size=1, shuffle=False)

    hit_list, hit1_list, f1_list = [], [], []
    precision_list, recall_list = [], []
    results = []

    for i, batch in enumerate(tqdm(test_loader, desc="  LLM Eval", leave=False)):
        question = batch['question'][0]
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        ground_truth = [p[0] for p in batch["answer"]]
        if not ground_truth or len(paths) == 0:
            continue

        ques_embed = batch["question_embedding"].to(device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
        graph_features = batch["graph_features"].squeeze(0).to(device)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)

        ranking_scores, path_probs = model(ques_embed, triplet_embeds, relation_embeds, graph_features)
        k = min(top_k, len(paths))
        selected_paths, selected_probs, _, _ = model.sample_paths(
            path_probs, paths, k, ranking_scores)

        sorted_indices = torch.argsort(selected_probs, descending=True)
        sorted_paths = [selected_paths[idx] for idx in sorted_indices.tolist()]

        prompt = format_prompt(question, sorted_paths, topk=top_k)
        try:
            raw_prediction = run_llm_inference(llm_model, tokenizer, prompt)
            prediction = extract_predictions_from_response(raw_prediction)
            prediction = [s for s in prediction if s != ""]
        except Exception as e:
            logger.warning(f"  LLM failed for question {i}: {e}")
            prediction = []

        answer = preprocess_date_answers(question, ground_truth)
        double_check = should_use_double_check(question)

        prec, _, _ = compute_precision(prediction, answer, double_check)
        rec, _, _ = compute_recall(prediction, answer, double_check)
        f1 = compute_f1_score(prec, rec)
        hit = compute_hit_score(prediction, answer, double_check)
        hit1 = compute_hit_at_1(prediction, answer, double_check)

        hit_list.append(hit)
        hit1_list.append(hit1)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

        results.append({"question": question, "prediction": prediction,
                        "ground_truth": answer, "hit": hit, "hit_at_1": hit1,
                        "f1": f1, "precision": prec, "recall": rec})

    n = len(hit_list)
    metrics = {
        "hit": np.mean(hit_list) * 100 if n else 0,
        "hit_at_1": np.mean(hit1_list) * 100 if n else 0,
        "macro_f1": np.mean(f1_list) * 100 if n else 0,
        "macro_precision": np.mean(precision_list) * 100 if n else 0,
        "macro_recall": np.mean(recall_list) * 100 if n else 0,
        "exact_match": (np.array(f1_list) == 1).sum() / n * 100 if n else 0,
        "total_samples": n,
        "model_path": model_path,
    }

    logger.info(f"  Hit: {metrics['hit']:.2f}, Hit@1: {metrics['hit_at_1']:.2f}, "
                f"F1: {metrics['macro_f1']:.2f}, Prec: {metrics['macro_precision']:.2f}, "
                f"Recall: {metrics['macro_recall']:.2f}, EM: {metrics['exact_match']:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "llm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "llm_detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation-2: Reversed Attention Study")
    parser.add_argument("--dataset", type=str, required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--train-data", type=str, required=True, help="Path to train .pt file")
    parser.add_argument("--val-data", type=str, required=True, help="Path to val .pt file")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test .pt file")
    parser.add_argument("--output-dir", type=str, default="results/ablation-2-v2/")
    parser.add_argument("--llm-model", type=str, default="llama", choices=["llama", "qwen", "deepseek"])
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM evaluation (coverage only)")
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Path to a trained model checkpoint directory. If provided, skips training and runs evaluation only.")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "run.log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("=" * 70)
    logger.info("ABLATION-2: REVERSED ATTENTION ARCHITECTURE")
    logger.info(f"Dataset: {args.dataset}, Device: {device}")
    logger.info("=" * 70)

    # Validate paths
    if not os.path.exists(args.test_data):
        logger.error(f"test data not found: {args.test_data}")
        sys.exit(1)

    if args.model_checkpoint is not None:
        # --- EVALUATION-ONLY MODE: Load checkpoint and run metrics ---
        model_path = args.model_checkpoint
        logger.info(f"Model checkpoint provided: {model_path}")
        logger.info("Skipping training. Running evaluation only.")

        if not os.path.exists(os.path.join(model_path, "path_ranker.pt")):
            logger.error(f"path_ranker.pt not found in checkpoint dir: {model_path}")
            sys.exit(1)

        # Load model
        model = PathRankingModelReversedAttention(device=str(device))
        ckpt = torch.load(os.path.join(model_path, "path_ranker.pt"),
                          weights_only=False, map_location="cpu")
        model.question_triplet_attention.load_state_dict(ckpt['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(ckpt['question_relation_attention'])
        model.gate_network.load_state_dict(ckpt['gate_network'])
        model.triplet_mlp.load_state_dict(ckpt['triplet_mlp'])
        model.relation_mlp.load_state_dict(ckpt['relation_mlp'])
        model.combiner_mlp.load_state_dict(ckpt['combiner_mlp'])
        model.temperature.data = ckpt['temperature'].to(device)
        model.baseline.data = ckpt['baseline'].to(device)
        model = model.to(device)
        logger.info("  Model loaded from checkpoint.")

    else:
        # --- FULL TRAINING MODE ---
        model_path = None

        for name, path in [("train", args.train_data), ("val", args.val_data)]:
            if not os.path.exists(path):
                logger.error(f"{name} data not found: {path}")
                sys.exit(1)

        # --- Step 1: Load Data ---
        logger.info("Step 1: Loading data...")
        train_data = torch.load(args.train_data, weights_only=False, map_location="cpu")
        val_data = torch.load(args.val_data, weights_only=False, map_location="cpu")
        logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        # --- Step 2: Pretrain (n=500, 5 epochs) ---
        logger.info("Step 2: Pretraining (n=500, 5 epochs)...")
        model = PathRankingModelReversedAttention(device=str(device))
        pretrain_dir = os.path.join(output_dir, "model", "pretrained")
        pretrain_ds = CosinePretrainingDataset(train_data, k=500)
        pretrain_val_ds = CosinePretrainingDataset(val_data, k=500)
        pretrain_loader = DataLoader(pretrain_ds, batch_size=1, shuffle=True, collate_fn=collate_fn_pretrain)
        pretrain_val_loader = DataLoader(pretrain_val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_pretrain)
        pretrainer = CosinePretrainer(model, device=str(device), checkpoint_dir=pretrain_dir)
        pretrainer.train(pretrain_loader, pretrain_val_loader, num_epochs=5)

        # Load best pretrained weights
        ckpt = torch.load(os.path.join(pretrain_dir, "best_pretrained_model-5.pt"),
                          weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("  Pretrained weights loaded.")

        # --- Step 3: Train with REINFORCE (n=1000, k=100, 30 epochs) ---
        logger.info("Step 3: Training with REINFORCE (n=1000, k=100, 30 epochs)...")
        train_dir = os.path.join(output_dir, "model", "trained")
        monitor_dir = os.path.join(output_dir, "training_plots")
        monitor = TrainingMonitor(log_dir=monitor_dir)
        train_sampled = SampledDataset(train_data, k=1000)
        val_sampled = SampledDataset(val_data, k=1000)
        train_loader = DataLoader(train_sampled, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sampled, batch_size=1, shuffle=False)
        trainer = JointTrainer(model, compute_reward_v8, checkpoint_dir=train_dir, monitor=monitor)
        trainer.train(train_loader, val_loader, num_epochs=30, k=100, early_stopping_patience=10)

        # Set model_path to the best checkpoint
        model_path = os.path.join(train_dir, f"complete_{trainer.best_epoch}_best")
        logger.info(f"  Best model path: {model_path}")

        # Free train/val memory
        del train_data, val_data, train_loader, val_loader
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 4: Load test data & run coverage ---
    logger.info("Step 4: Loading test data and running coverage analysis...")
    test_data = torch.load(args.test_data, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)}")

    coverage_dir = os.path.join(output_dir, "coverage")
    run_coverage_analysis(test_data, model, top_k=100, output_dir=coverage_dir, model_path=model_path)

    # --- Step 5: LLM evaluation ---
    if not args.skip_llm:
        logger.info("Step 5: Running LLM evaluation...")
        llm_dir = os.path.join(output_dir, "llm-results")
        run_llm_evaluation(test_data, model, top_k=100, output_dir=llm_dir,
                           llm_model_name=args.llm_model, model_path=model_path)
    else:
        logger.info("Step 5: Skipped LLM evaluation (--skip-llm)")

    logger.info("=" * 70)
    logger.info("ABLATION-2 COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
