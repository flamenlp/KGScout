"""
Main training logic for PathRankingModel using REINFORCE algorithm.

Aligned with the working JointTrainer in ablation-2/model_ablation.py:
- REINFORCE with running baseline (buffered, decay-clamped)
- AdamW + cosine schedule with warmup
- Gradient accumulation (default 32)
- Early stopping on best val reward
- save_pretrained checkpoint format (component-level state dicts)
"""

import os
import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from transformers import get_cosine_schedule_with_warmup

from src.training.rewards import compute_reward_v8


class Trainer:
    """
    Main training with REINFORCE algorithm.

    Matches JointTrainer from ablation-2/model_ablation.py:
    - Running baseline with buffered update (no learnable baseline MSE)
    - Cosine schedule with warmup
    - Handles DataLoader-batched format (batch_size=1, default collate)
    - Saves checkpoints using model.save_pretrained()
    """

    def __init__(
        self,
        path_ranker,
        checkpoint_dir: str,
        device: str = "cuda",
        gradient_accumulation_steps: int = 32,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            path_ranker: PathRankingModel (should have pretrained weights loaded)
            checkpoint_dir: Directory for checkpoints
            device: Device ('cuda' or 'cpu')
            gradient_accumulation_steps: Gradient accumulation steps (default 32)
            max_grad_norm: Max gradient norm for clipping (default 1.0)
        """
        self.path_ranker = path_ranker.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Running baseline state
        self.running_baseline = 0.0
        self.reward_buffer = []
        self.best_val_reward = float("-inf")

    def _reinforce_loss(self, log_probs, reward, baseline):
        """REINFORCE policy gradient loss."""
        return -(log_probs * (reward - baseline).detach()).mean()

    def _update_baseline(self):
        """Update running baseline from reward buffer (same logic as model_ablation.py)."""
        if self.reward_buffer:
            avg = sum(self.reward_buffer) / len(self.reward_buffer)
            if self.running_baseline == 0:
                self.running_baseline = avg * 0.8
            else:
                err = avg - self.running_baseline
                if abs(err) > 0.5:
                    self.running_baseline += 0.1 * err
            self.running_baseline = min(self.running_baseline, avg * 0.9)
            self.path_ranker.baseline.data = torch.tensor(
                [self.running_baseline], device=self.device
            )
            self.reward_buffer = []

    def train_step(self, batch, k):
        """
        Single REINFORCE training step.

        Handles DataLoader-batched format (batch_size=1, default collate):
        - Entities are wrapped in tuples: batch['q_entity'] = [('ent1',), ('ent2',)]
        - Triplet data is nested: batch['topk_rel_data'] = [(score, ((s,), (r,), (o,))), ...]
        - Tensors have extra batch dim that needs squeeze(0)

        Args:
            batch: DataLoader batch dict
            k: Number of triplets to sample

        Returns:
            Tuple of (loss, reward_tensor) or (None, None) if invalid
        """
        # Extract entities (unwrap collate tuples)
        q_ent = [p[0] for p in batch["q_entity"]]
        a_ent = [p[0] for p in batch["a_entity"]]

        # Extract structured triplets from DataLoader-batched format
        triplets = [(d[1][0][0], d[1][1][0], d[1][2][0]) for d in batch["topk_rel_data"]]

        if not q_ent:
            return None, None

        # Move tensors to device (squeeze batch dim from DataLoader)
        qe = batch["question_embedding"].to(self.device)
        te = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(self.device)
        re = batch["topK_rel_embeddings"].squeeze(0).to(self.device)
        gf = batch["graph_features"].squeeze(0).to(self.device)

        # Forward pass
        scores, probs = self.path_ranker(qe, te, re, gf)

        # Sample paths using REINFORCE
        sel_triplets, _, _, log_probs = self.path_ranker.sample_paths(
            probs, triplets, k, scores
        )

        # Compute reward
        rew = compute_reward_v8(sel_triplets, q_ent, a_ent)
        if rew is None:
            return None, None

        r = torch.tensor([rew], device=self.device)
        self.reward_buffer.append(r.item())

        # REINFORCE loss
        baseline_t = torch.tensor([self.running_baseline], device=self.device)
        loss = self._reinforce_loss(log_probs, r.expand(log_probs.size(0)), baseline_t)

        return loss, r

    @torch.no_grad()
    def validate(self, val_dl, k):
        """Run validation and return (avg_loss, avg_reward)."""
        self.path_ranker.eval()
        total_loss = 0.0
        total_reward = 0.0
        count = 0

        for batch in tqdm(val_dl, desc="  Val", leave=False):
            loss, reward = self.train_step(batch, k)
            if loss is None:
                continue
            total_loss += loss.item()
            total_reward += reward.item()
            count += 1

        if count == 0:
            return 0.0, 0.0
        return total_loss / count, total_reward / count

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        validation_interval: int = 1,
        early_stopping_patience: int = 10,
        k: int = 100,
        **kwargs,  # Accept extra kwargs for backward compat (monitor, weight_decay, etc.)
    ):
        """
        Run main REINFORCE training.

        Args:
            train_dataloader: Training DataLoader (batch_size=1)
            val_dataloader: Validation DataLoader (batch_size=1), optional
            num_epochs: Number of epochs (default 30)
            learning_rate: Learning rate (default 1e-4)
            warmup_steps: Warmup steps for cosine schedule (default 100)
            validation_interval: Validate every N epochs (default 1)
            early_stopping_patience: Patience for early stopping (default 10)
            k: Number of triplets to select per sample (default 100)
        """
        optimizer = torch.optim.AdamW(self.path_ranker.parameters(), lr=learning_rate)
        total_steps = (len(train_dataloader) * num_epochs) // self.accum
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n  Epoch {epoch + 1}/{num_epochs} (k={k})")

            # --- Training ---
            self.path_ranker.train()
            rewards = []
            losses = []
            valid_count = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="  Train", leave=False)):
                loss, reward = self.train_step(batch, k)
                if loss is None:
                    continue
                if math.isnan(reward.item()):
                    continue

                rewards.append(reward.item())
                losses.append(loss.item())
                valid_count += 1

                (loss / self.accum).backward()

                if valid_count % self.accum == 0:
                    self._update_baseline()
                    torch.nn.utils.clip_grad_norm_(
                        self.path_ranker.parameters(), self.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Flush remaining gradients
            if valid_count % self.accum != 0:
                self._update_baseline()
                torch.nn.utils.clip_grad_norm_(
                    self.path_ranker.parameters(), self.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_reward = np.mean(rewards) if rewards else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"    Reward: {avg_reward:.4f}, Loss: {avg_loss:.4f}")

            # --- Validation & Early Stopping ---
            if (epoch + 1) % validation_interval == 0 and val_dataloader:
                val_loss, val_reward = self.validate(val_dataloader, k)
                print(f"    Val Reward: {val_reward:.4f}")

                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    patience_counter = 0
                    self._save_checkpoint(epoch + 1, is_best=True)
                else:
                    patience_counter += 1
                    self._save_checkpoint(epoch + 1, is_best=False)

                if patience_counter >= early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break
            else:
                # Save every epoch even without validation
                self._save_checkpoint(epoch + 1, is_best=False)

        print(f"\n  Training complete. Best val reward: {self.best_val_reward:.4f}")

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Save checkpoint using model.save_pretrained (component-level state dicts)."""
        tag = f"checkpoint_best_epoch_{epoch}" if is_best else f"checkpoint_epoch_{epoch}"
        save_dir = os.path.join(self.checkpoint_dir, tag)
        self.path_ranker.save_pretrained(save_dir)
        print(f"    {'Best c' if is_best else 'C'}heckpoint saved: {save_dir}")
