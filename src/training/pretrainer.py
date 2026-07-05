"""
Pretraining logic for PathRankingModel.

This module implements the Pretrainer class which handles the pretraining phase
with cosine similarity objective. The pretraining phase uses fixed configuration:
5 epochs, n=500.

Aligned with the working implementation in ablation-2/model_ablation.py:
- Loss: 0.5 * MSE + 0.5 * margin_ranking_loss (100 sampled pairs)
- Optimizer: AdamW + ReduceLROnPlateau
- Targets: exponential positional decay (cosine_targets)
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Pretrainer:
    """
    Handles pretraining phase with cosine similarity objective.
    Fixed configuration: 5 epochs, n=500.

    Matches the proven CosinePretrainer from ablation-2/model_ablation.py:
    - 0.5*MSE + 0.5*MarginRankingLoss (100 sampled pairs)
    - AdamW optimizer with ReduceLROnPlateau scheduler
    - Gradient accumulation (default 8)
    """

    def __init__(
        self,
        path_ranker,
        checkpoint_dir: str,
        device: str = "cuda"
    ):
        """
        Initialize Pretrainer.

        Args:
            path_ranker: PathRankingModel instance to train
            checkpoint_dir: Directory path where checkpoints will be saved
            device: Device to use for training (default: "cuda")
        """
        self.path_ranker = path_ranker
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _compute_ranking_loss(self, predicted_scores, target_scores, sample_pairs=100):
        """
        Margin ranking loss over sampled pairs.

        Samples up to `sample_pairs` random pairs and computes margin ranking loss
        where the target ordering defines the correct direction.
        """
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
        """
        Single training step.

        Expects batch with keys:
            - question_embedding: [embed_dim] tensor
            - path_embeddings (or topk_linearized_triplet_embeddings): [n, embed_dim]
            - rel_embeddings (or topK_rel_embeddings): [n, embed_dim]
            - cosine_targets: [n] exponential decay targets
            - graph_features: [n, 2] PPR features

        Returns:
            Loss tensor or None if batch is invalid.
        """
        if batch is None:
            return None

        question_embed = batch["question_embedding"].to(self.device)

        # Support both field naming conventions
        if "path_embeddings" in batch:
            path_embeds = batch["path_embeddings"].to(self.device)
        else:
            path_embeds = batch["topk_linearized_triplet_embeddings"].to(self.device)

        if "rel_embeddings" in batch:
            rel_embeds = batch["rel_embeddings"].to(self.device)
        else:
            rel_embeds = batch["topK_rel_embeddings"].to(self.device)

        cosine_targets = batch["cosine_targets"].to(self.device)
        graph_features = batch["graph_features"].to(self.device)

        # Forward pass (unsqueeze question to add batch dim)
        predicted_scores, _ = self.path_ranker(
            question_embed.unsqueeze(0), path_embeds, rel_embeds, graph_features
        )

        # Loss: 0.5 * MSE + 0.5 * MarginRanking (100 sampled pairs)
        mse = self.mse_loss(predicted_scores, cosine_targets)
        ranking = self._compute_ranking_loss(predicted_scores, cosine_targets)
        return 0.5 * mse + 0.5 * ranking

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        **kwargs  # Accept extra kwargs for backward compat (validation_interval, save_best, etc.)
    ):
        """
        Run pretraining.

        Args:
            train_dataloader: DataLoader for training data (batch_size=1, collate filters None)
            val_dataloader: Optional DataLoader for validation
            num_epochs: Number of epochs (default 5)
            learning_rate: Learning rate (default 1e-4)
            weight_decay: Weight decay (default 1e-5)
            gradient_accumulation_steps: Gradient accumulation steps (default 8)
        """
        optimizer = torch.optim.AdamW(
            self.path_ranker.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"\n  Pretrain Epoch {epoch + 1}/{num_epochs}")

            # --- Training ---
            self.path_ranker.train()
            epoch_loss = 0.0
            valid_batches = 0
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

            # Flush remaining gradients
            if valid_batches % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = epoch_loss / max(valid_batches, 1)
            print(f"    Train Loss: {avg_train_loss:.4f}")

            # --- Validation ---
            if val_dataloader is not None:
                self.path_ranker.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        loss = self.train_step(batch)
                        if loss is None:
                            continue
                        val_loss += loss.item()
                        val_count += 1

                avg_val_loss = val_loss / max(val_count, 1)
                print(f"    Val Loss: {avg_val_loss:.4f}")
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint(epoch + 1)
            else:
                scheduler.step(avg_train_loss)

        # Always save final checkpoint
        self._save_checkpoint(num_epochs)
        print(f"\n  Pretraining complete! Checkpoints in: {self.checkpoint_dir}")

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"best_pretrained_model-{epoch}.pt")
        torch.save(
            {"epoch": epoch, "model_state_dict": self.path_ranker.state_dict()},
            path,
        )
        print(f"    Checkpoint saved: {path}")
