"""
Main training logic for PathRankingModel using REINFORCE algorithm.

This module implements the Trainer class which handles the main training phase
with REINFORCE algorithm and configurable k parameter for sampling.
"""

import os
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.rewards import compute_reward_v8
from training.monitor import TrainingMonitor
from training.checkpoint_utils import validate_checkpoint_structure


class Trainer:
    """
    Handles main training phase with REINFORCE algorithm.
    Configurable k parameter for number of triplets to sample.
    
    This phase ALWAYS runs after pretraining and loads pretrained weights.
    
    Requirements:
        - 3.2: Organize main training logic in training/ directory
        - 3.4: Accept configurable k parameter
        - 3.6: Save checkpoints to output directory during training
        - 3.7: Compute rewards using compute_reward_v8 function
        - 3.8: Implement REINFORCE algorithm for policy gradient updates
        - 3.9: Support gradient accumulation during training
        - 10.1: Support configurable learning rate
        - 10.2: Support configurable number of epochs
        - 10.3: Support configurable gradient accumulation steps
        - 10.4: Support configurable validation intervals
        - 10.5: Support configurable early stopping patience
    """
    
    def __init__(
        self,
        path_ranker,
        checkpoint_dir: str,
        device: str = "cuda"
    ):
        """
        Initialize trainer with a PathRankingModel that has been pretrained.
        The model should already have weights loaded from pretraining phase.
        
        Args:
            path_ranker: PathRankingModel instance (should have pretrained weights)
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
        
        # Move model to device
        self.path_ranker.to(device)
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        monitor: TrainingMonitor,
        num_epochs: int,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        gradient_accumulation_steps: int,
        validation_interval: int,
        early_stopping_patience: int,
        k: int  # Configurable: 30, 50, 100, or 150
    ):
        """
        Run main training with REINFORCE algorithm.
        Uses configurable k parameter for sampling.
        
        IMPORTANT: This method assumes the path_ranker already has
        pretrained weights loaded. It does NOT load them automatically.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            monitor: TrainingMonitor instance for logging
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Number of steps to accumulate gradients
            validation_interval: Interval for validation (in epochs)
            early_stopping_patience: Number of epochs to wait before early stopping
            k: Number of triplets to sample per question
        """
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.path_ranker.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler with warmup
        total_steps = num_epochs * len(train_dataloader)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Main Training Epoch {epoch + 1}/{num_epochs} (k={k})")
            print(f"{'='*60}")
            
            # Training phase
            self.path_ranker.train()
            train_loss, train_reward = self._train_epoch(
                train_dataloader,
                optimizer,
                scheduler,
                monitor,
                gradient_accumulation_steps,
                k,
                global_step
            )
            
            global_step += len(train_dataloader)
            
            # Log epoch metrics
            epoch_metrics = {
                'train_loss': train_loss,
                'train_reward': train_reward,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Reward: {train_reward:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Validation phase
            val_loss = None
            val_reward = None
            if val_dataloader is not None and (epoch + 1) % validation_interval == 0:
                self.path_ranker.eval()
                val_loss, val_reward = self._validate_epoch(
                    val_dataloader,
                    k
                )
                
                epoch_metrics['val_loss'] = val_loss
                epoch_metrics['val_reward'] = val_reward
                
                print(f"\nVal Loss: {val_loss:.4f}")
                print(f"Val Reward: {val_reward:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"best_model_k{k}.pt"
                    )
                    self.save_checkpoint(
                        best_checkpoint_path,
                        epoch + 1,
                        optimizer,
                        scheduler,
                        train_loss,
                        val_loss,
                        k
                    )
                    print(f"Best model saved: {best_checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
            
            # Log epoch metrics to monitor
            monitor.log_epoch(epoch_metrics, epoch + 1)
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}_k{k}.pt"
            )
            self.save_checkpoint(
                checkpoint_path,
                epoch + 1,
                optimizer,
                scheduler,
                train_loss,
                val_loss,
                k
            )
            print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Generate training progress plots
        monitor.plot_metrics()
        print(f"\nMain training complete! Plots saved to {monitor.log_dir}")
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: TrainingMonitor,
        gradient_accumulation_steps: int,
        k: int,
        global_step: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, average_reward)
        """
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Skip None batches (empty samples)
            if batch is None:
                continue
            
            try:
                # Train step
                loss, reward = self.train_step(batch, k)
                
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
                        None,
                        k
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
                
                # Log step metrics
                step_metrics = {
                    'loss': loss.item() * gradient_accumulation_steps,
                    'reward': reward
                }
                monitor.log_step(step_metrics, global_step + batch_idx)
                
                # Accumulate metrics
                total_loss += loss.item() * gradient_accumulation_steps
                total_reward += reward
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
            return 0.0, 0.0
        
        return total_loss / num_batches, total_reward / num_batches
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        k: int
    ) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, average_reward)
        """
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Skip None batches (empty samples)
                if batch is None:
                    continue
                
                # Validation step (same as train step but without gradients)
                loss, reward = self.train_step(batch, k)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_reward += reward
                num_batches += 1
        
        # Return average metrics
        if num_batches == 0:
            return 0.0, 0.0
        
        return total_loss / num_batches, total_reward / num_batches
    
    def train_step(
        self,
        batch: Dict,
        k: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Single training step with REINFORCE.
        
        Process:
        1. Forward pass to get probabilities
        2. Sample k paths using categorical sampling
        3. Compute reward using compute_reward_v8
        4. Compute policy gradient loss
        5. Update baseline
        
        Args:
            batch: Dictionary containing batch data with keys:
                - question_embedding: Question embedding tensor
                - topk_linearized_triplet_embeddings: Triplet embeddings
                - topK_rel_embeddings: Relation embeddings
                - graph_features: PPR graph features
                - topk_linearized_triplets: List of linearized triplet strings
                - topk_rel_data: List of (score, triplet) tuples
                - q_entity: List of question entities
                - a_entity: List of answer entities
            k: Number of paths to sample
        
        Returns:
            Tuple of (loss, reward)
        """
        # Move batch to device
        question_embed = batch["question_embedding"].to(self.device)
        triplet_embeds = batch["topk_linearized_triplet_embeddings"].to(self.device)
        relation_embeds = batch["topK_rel_embeddings"].to(self.device)
        graph_scores = batch["graph_features"].to(self.device)
        
        # Get paths and triplets
        paths = batch["topk_linearized_triplets"]
        triplets = [triplet for _, triplet in batch["topk_rel_data"]]
        q_entities = batch["q_entity"]
        a_entities = batch["a_entity"]
        
        # Forward pass to get probabilities
        ranking_scores, path_probs = self.path_ranker(
            question_embed,
            triplet_embeds,
            relation_embeds,
            graph_scores
        )
        
        # Sample k paths using categorical sampling
        selected_paths, selected_probs, selected_scores, log_probs = self.path_ranker.sample_paths(
            path_probs,
            paths,
            k,
            ranking_scores
        )
        
        # Get selected triplets
        selected_indices = [paths.index(path) for path in selected_paths]
        selected_triplets = [triplets[i] for i in selected_indices]
        
        # Compute reward using compute_reward_v8
        reward = compute_reward_v8(
            triplets=selected_triplets,
            q_entities=q_entities,
            a_entities=a_entities
        )
        
        # REINFORCE loss: -log_prob * (reward - baseline)
        # The baseline is a learnable parameter that reduces variance
        baseline = self.path_ranker.baseline
        advantage = reward - baseline.item()
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantage).mean()
        
        # Baseline update loss (MSE between baseline and reward)
        baseline_loss = (baseline - reward) ** 2
        
        # Total loss
        total_loss = policy_loss + 0.1 * baseline_loss
        
        return total_loss, reward
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loss: float,
        val_loss: Optional[float],
        k: int
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
            k: K parameter used for training
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.path_ranker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else float('inf'),
            "k": k,
            "metrics": {
                "k": k,
                "epoch": epoch
            }
        }
        
        # Validate checkpoint structure before saving
        validate_checkpoint_structure(checkpoint)
        
        torch.save(checkpoint, checkpoint_path)
