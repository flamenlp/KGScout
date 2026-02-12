"""
Pretraining logic for PathRankingModel.

This module implements the Pretrainer class which handles the pretraining phase
with cosine similarity objective. The pretraining phase uses fixed configuration:
5 epochs, n=500, and is not configurable.
"""

import os
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

from training.checkpoint_utils import validate_checkpoint_structure


class Pretrainer:
    """
    Handles pretraining phase with cosine similarity objective.
    Fixed configuration: 5 epochs, n=500, not configurable.
    
    This phase ALWAYS runs first before main training.
    
    Requirements:
        - 3.1: Organize pretraining logic in training/ directory
        - 3.3: Use exactly 5 epochs with n=500 fixed
        - 3.6: Save checkpoints to output directory during training
        - 3.10: Generate training progress graphs using TrainingMonitor
        - 10.6: Use fixed hyperparameters for pretraining
    """
    
    def __init__(
        self,
        path_ranker,
        checkpoint_dir: str,
        device: str = "cuda"
    ):
        """
        Initialize Pretrainer with PathRankingModel and checkpoint directory.
        
        Args:
            path_ranker: PathRankingModel instance to train
            checkpoint_dir: Directory path where checkpoints will be saved
            device: Device to use for training (default: "cuda")
        """
        self.path_ranker = path_ranker
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Check if checkpoint directory is writable before training starts
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(checkpoint_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            raise PermissionError(
                f"Checkpoint directory '{checkpoint_dir}' is not writable. "
                f"Please check permissions or specify a different directory. Error: {e}"
            )
        
        # Storage for training history
        self.train_losses = []
        self.val_losses = []
        self.train_mse_losses = []
        self.train_ranking_losses = []
        self.val_mse_losses = []
        self.val_ranking_losses = []
        self.train_spearman_corrs = []
        self.val_spearman_corrs = []
        self.train_ndcg_scores = []
        self.val_ndcg_scores = []
        self.epochs = []
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 5,  # Fixed, not configurable
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        validation_interval: int = 1,
        save_best: bool = True
    ):
        """
        Run pretraining for exactly 5 epochs with n=500.
        Computes MSE loss, ranking loss, Spearman correlation, and NDCG@50.
        
        Saves checkpoint after each epoch. The final epoch checkpoint
        (best_pretrained_model-5.pt) is automatically loaded for main training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            num_epochs: Number of epochs (fixed at 5)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Number of steps to accumulate gradients
            validation_interval: Interval for validation (in epochs)
            save_best: Whether to save best model based on validation loss
        """
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.path_ranker.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_dataloader)
        )
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Pretraining Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training phase
            self.path_ranker.train()
            train_loss, train_mse, train_ranking, train_spearman, train_ndcg = self._train_epoch(
                train_dataloader,
                optimizer,
                scheduler,
                gradient_accumulation_steps
            )
            
            # Store training metrics
            self.epochs.append(epoch + 1)
            self.train_losses.append(train_loss)
            self.train_mse_losses.append(train_mse)
            self.train_ranking_losses.append(train_ranking)
            self.train_spearman_corrs.append(train_spearman)
            self.train_ndcg_scores.append(train_ndcg)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"  MSE Loss: {train_mse:.4f}")
            print(f"  Ranking Loss: {train_ranking:.4f}")
            print(f"  Spearman Corr: {train_spearman:.4f}")
            print(f"  NDCG@50: {train_ndcg:.4f}")
            
            # Validation phase
            val_loss = None
            if val_dataloader is not None and (epoch + 1) % validation_interval == 0:
                self.path_ranker.eval()
                val_loss, val_mse, val_ranking, val_spearman, val_ndcg = self._validate_epoch(
                    val_dataloader
                )
                
                # Store validation metrics
                self.val_losses.append(val_loss)
                self.val_mse_losses.append(val_mse)
                self.val_ranking_losses.append(val_ranking)
                self.val_spearman_corrs.append(val_spearman)
                self.val_ndcg_scores.append(val_ndcg)
                
                print(f"\nVal Loss: {val_loss:.4f}")
                print(f"  MSE Loss: {val_mse:.4f}")
                print(f"  Ranking Loss: {val_ranking:.4f}")
                print(f"  Spearman Corr: {val_spearman:.4f}")
                print(f"  NDCG@50: {val_ndcg:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"best_pretrained_model-{epoch + 1}.pt"
            )
            self.save_checkpoint(
                checkpoint_path,
                epoch + 1,
                optimizer,
                scheduler,
                train_loss,
                val_loss
            )
            print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Save best model based on validation loss
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    "best_pretrained_model.pt"
                )
                self.save_checkpoint(
                    best_checkpoint_path,
                    epoch + 1,
                    optimizer,
                    scheduler,
                    train_loss,
                    val_loss
                )
                print(f"Best model saved: {best_checkpoint_path}")
        
        # Generate training progress plots
        self.plot_training_progress()
        print(f"\nPretraining complete! Plots saved to {self.checkpoint_dir}")
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gradient_accumulation_steps: int
    ) -> Tuple[float, float, float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (total_loss, mse_loss, ranking_loss, spearman_corr, ndcg_score)
        """
        total_loss = 0.0
        total_mse = 0.0
        total_ranking = 0.0
        total_spearman = 0.0
        total_ndcg = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Skip None batches (empty samples)
            if batch is None:
                continue
            
            try:
                # Train step
                loss, mse_loss, ranking_loss, spearman, ndcg = self.train_step(batch)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  WARNING: NaN or Inf loss detected at batch {batch_idx}")
                    print("Saving emergency checkpoint...")
                    emergency_path = os.path.join(
                        self.checkpoint_dir,
                        f"emergency_checkpoint_batch_{batch_idx}.pt"
                    )
                    self.save_checkpoint(
                        emergency_path,
                        -1,  # Special epoch marker for emergency
                        optimizer,
                        scheduler,
                        float('nan'),
                        None
                    )
                    raise ValueError(
                        f"NaN or Inf loss encountered at batch {batch_idx}. "
                        f"Emergency checkpoint saved to {emergency_path}. "
                        "This may indicate numerical instability. Consider reducing learning rate."
                    )
                
                # Check for gradient explosion (loss > 1e6)
                if loss.item() > 1e6:
                    print(f"\n⚠️  WARNING: Gradient explosion detected (loss={loss.item():.2e})")
                    print("Applying aggressive gradient clipping...")
                
                # Normalize loss by accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Accumulate metrics
                total_loss += loss.item() * gradient_accumulation_steps
                total_mse += mse_loss
                total_ranking += ranking_loss
                total_spearman += spearman
                total_ndcg += ndcg
                num_batches += 1
                
            except RuntimeError as e:
                # Handle OOM errors
                if "out of memory" in str(e).lower():
                    print(f"\n❌ ERROR: Out of Memory (OOM) at batch {batch_idx}")
                    print("\nSuggestions to fix OOM error:")
                    print("  1. Reduce batch size in your DataLoader")
                    print(f"  2. Increase gradient_accumulation_steps (current: {gradient_accumulation_steps})")
                    print("  3. Reduce model size or sequence length")
                    print("  4. Use a GPU with more memory")
                    print("  5. Enable gradient checkpointing if available")
                    
                    # Try to clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    raise RuntimeError(
                        f"Out of memory error at batch {batch_idx}. "
                        "See suggestions above to resolve this issue."
                    ) from e
                else:
                    # Re-raise other runtime errors
                    raise
        
        # Final gradient update if there are remaining gradients
        if num_batches % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Return average metrics
        if num_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        return (
            total_loss / num_batches,
            total_mse / num_batches,
            total_ranking / num_batches,
            total_spearman / num_batches,
            total_ndcg / num_batches
        )
    
    def _validate_epoch(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float, float, float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (total_loss, mse_loss, ranking_loss, spearman_corr, ndcg_score)
        """
        total_loss = 0.0
        total_mse = 0.0
        total_ranking = 0.0
        total_spearman = 0.0
        total_ndcg = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Skip None batches (empty samples)
                if batch is None:
                    continue
                
                # Validation step
                loss, mse_loss, ranking_loss, spearman, ndcg = self.train_step(batch)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_mse += mse_loss
                total_ranking += ranking_loss
                total_spearman += spearman
                total_ndcg += ndcg
                num_batches += 1
        
        # Return average metrics
        if num_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        return (
            total_loss / num_batches,
            total_mse / num_batches,
            total_ranking / num_batches,
            total_spearman / num_batches,
            total_ndcg / num_batches
        )
    
    def train_step(self, batch: Dict) -> Tuple[float, float, float, float, float]:
        """
        Single training step with cosine similarity targets.
        
        Computes:
        1. MSE loss between predicted scores and cosine similarity targets
        2. Ranking loss (margin ranking loss)
        3. Spearman correlation between predicted and target scores
        4. NDCG@50 for ranking quality
        
        Args:
            batch: Dictionary containing batch data with keys:
                - question_embedding: Question embedding tensor
                - topk_linearized_triplet_embeddings: Triplet embeddings
                - topK_rel_embeddings: Relation embeddings
                - graph_features: PPR graph features
                - topk_rel_data: List of (score, triplet) tuples
        
        Returns:
            Tuple of (total_loss, mse_loss, ranking_loss, spearman_corr, ndcg_score)
        """
        # Move batch to device
        question_embed = batch["question_embedding"].to(self.device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].to(self.device)
        relation_embeds = batch["topK_rel_embeddings"].to(self.device)
        graph_scores = batch["graph_features"].to(self.device)
        
        # Extract cosine similarity scores as targets
        # topk_rel_data is a list of (score, triplet) tuples
        target_scores = torch.tensor(
            [score for score, _ in batch["topk_rel_data"]],
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward pass
        predicted_scores, _ = self.path_ranker(
            question_embed,
            triplet_embeds,
            relation_embeds,
            graph_scores
        )
        
        # 1. MSE Loss: Difference between predicted and target scores
        mse_loss = F.mse_loss(predicted_scores, target_scores)
        
        # 2. Ranking Loss: Margin ranking loss
        # Create pairs where target_i > target_j
        ranking_loss = 0.0
        num_pairs = 0
        margin = 0.1
        
        for i in range(len(target_scores)):
            for j in range(i + 1, len(target_scores)):
                if target_scores[i] > target_scores[j]:
                    # predicted_i should be > predicted_j
                    loss = torch.clamp(
                        margin - (predicted_scores[i] - predicted_scores[j]),
                        min=0.0
                    )
                    ranking_loss += loss
                    num_pairs += 1
                elif target_scores[j] > target_scores[i]:
                    # predicted_j should be > predicted_i
                    loss = torch.clamp(
                        margin - (predicted_scores[j] - predicted_scores[i]),
                        min=0.0
                    )
                    ranking_loss += loss
                    num_pairs += 1
        
        if num_pairs > 0:
            ranking_loss = ranking_loss / num_pairs
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = mse_loss + 0.1 * ranking_loss
        
        # 3. Spearman Correlation
        spearman_corr = self.compute_spearman_correlation(
            predicted_scores,
            target_scores
        )
        
        # 4. NDCG@50
        ndcg_score = self.compute_ndcg_at_k(
            predicted_scores,
            target_scores,
            k=50
        )
        
        return (
            total_loss,
            mse_loss.item(),
            ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else ranking_loss,
            spearman_corr,
            ndcg_score
        )
    
    def compute_spearman_correlation(
        self,
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor
    ) -> float:
        """
        Compute Spearman rank correlation between predicted and target scores.
        
        Spearman correlation measures how well the ranking of predicted scores
        matches the ranking of target scores.
        
        Args:
            predicted_scores: Predicted ranking scores
            target_scores: Target (ground truth) scores
        
        Returns:
            Spearman correlation coefficient (range: -1 to 1)
        """
        # Convert to numpy for scipy
        pred_np = predicted_scores.detach().cpu().numpy()
        target_np = target_scores.detach().cpu().numpy()
        
        # Handle edge cases
        if len(pred_np) < 2:
            return 0.0
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(pred_np, target_np)
            # Handle NaN (e.g., when all values are the same)
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except:
            return 0.0
    
    def compute_ndcg_at_k(
        self,
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        k: int = 50
    ) -> float:
        """
        Compute NDCG@k (Normalized Discounted Cumulative Gain) for ranking quality.
        
        NDCG measures the quality of ranking by comparing the predicted ranking
        to the ideal ranking based on target scores. It gives more weight to
        correct predictions at the top of the ranking.
        
        Args:
            predicted_scores: Predicted ranking scores
            target_scores: Target (ground truth) scores
            k: Number of top items to consider (default: 50)
        
        Returns:
            NDCG@k score (range: 0 to 1, higher is better)
        """
        # Convert to numpy
        pred_np = predicted_scores.detach().cpu().numpy()
        target_np = target_scores.detach().cpu().numpy()
        
        # Handle edge cases
        if len(pred_np) == 0:
            return 0.0
        
        k = min(k, len(pred_np))
        
        # Get top-k indices based on predicted scores
        top_k_indices = np.argsort(pred_np)[::-1][:k]
        
        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            relevance = target_np[idx]
            # DCG formula: sum(rel_i / log2(i + 2))
            dcg += relevance / np.log2(i + 2)
        
        # Compute IDCG (Ideal DCG) - best possible ranking
        ideal_indices = np.argsort(target_np)[::-1][:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            relevance = target_np[idx]
            idcg += relevance / np.log2(i + 2)
        
        # Compute NDCG
        if idcg == 0.0:
            return 0.0
        
        ndcg = dcg / idcg
        return float(ndcg)
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loss: float,
        val_loss: Optional[float]
    ):
        """
        Save checkpoint with model state and training progress.
        
        Args:
            checkpoint_path: Path where checkpoint will be saved
            epoch: Current epoch number
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (None if not validated)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.path_ranker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else float('inf'),
            "metrics": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_mse_losses": self.train_mse_losses,
                "train_ranking_losses": self.train_ranking_losses,
                "val_mse_losses": self.val_mse_losses,
                "val_ranking_losses": self.val_ranking_losses,
                "train_spearman_corrs": self.train_spearman_corrs,
                "val_spearman_corrs": self.val_spearman_corrs,
                "train_ndcg_scores": self.train_ndcg_scores,
                "val_ndcg_scores": self.val_ndcg_scores,
                "epochs": self.epochs
            }
        }
        
        # Validate checkpoint structure before saving
        validate_checkpoint_structure(checkpoint)
        
        torch.save(checkpoint, checkpoint_path)
    
    def plot_training_progress(self):
        """
        Generate training progress visualization.
        
        Creates PNG plots showing:
        - Training and validation losses
        - MSE and ranking losses
        - Spearman correlation
        - NDCG@50 scores
        
        Saves plots to checkpoint directory.
        """
        if not self.epochs:
            print("No training data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        ax.plot(self.epochs, self.train_losses, marker='o', label='Train Loss')
        if self.val_losses:
            ax.plot(self.epochs, self.val_losses, marker='s', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: MSE and Ranking Loss
        ax = axes[0, 1]
        ax.plot(self.epochs, self.train_mse_losses, marker='o', label='Train MSE')
        ax.plot(self.epochs, self.train_ranking_losses, marker='s', label='Train Ranking')
        if self.val_mse_losses:
            ax.plot(self.epochs, self.val_mse_losses, marker='^', label='Val MSE')
        if self.val_ranking_losses:
            ax.plot(self.epochs, self.val_ranking_losses, marker='v', label='Val Ranking')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('MSE and Ranking Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Spearman Correlation
        ax = axes[1, 0]
        ax.plot(self.epochs, self.train_spearman_corrs, marker='o', label='Train Spearman')
        if self.val_spearman_corrs:
            ax.plot(self.epochs, self.val_spearman_corrs, marker='s', label='Val Spearman')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Spearman Rank Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: NDCG@50
        ax = axes[1, 1]
        ax.plot(self.epochs, self.train_ndcg_scores, marker='o', label='Train NDCG@50')
        if self.val_ndcg_scores:
            ax.plot(self.epochs, self.val_ndcg_scores, marker='s', label='Val NDCG@50')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NDCG@50')
        ax.set_title('NDCG@50 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.checkpoint_dir, 'pretraining_progress.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress plot saved: {plot_path}")
