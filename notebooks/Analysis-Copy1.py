#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import string
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 


# In[2]:


import transformers
transformers.logging.set_verbosity_error()

torch.manual_seed(100)
np.random.seed(100)


# In[3]:


class JointTrainingDatasetv3PPR(Dataset):
    def __init__(self, train_data, device='cuda'):
        self.device = device
        # This dataset is prepared using  JointTrainingDatasetv3 dataset. Only graph features are computed in this dataset
        global count_dirty_data
        self.precomputed_data = []
        for entry in tqdm(train_data, total=len(train_data)):
            q_entity = [e.lower() for e in entry["q_entity"]]
            triplets=  [t[1] for t in entry["topk_rel_data"]]
            try:
                if len(q_entity)==0 or len(triplets[0])!=3:
                    count_dirty_data+=1
                if len(triplets)==0 or len(q_entity)==0:
                    graph_feats = torch.zeros((1, 2))
                else:
                    G = nx.DiGraph()
                    for (s, r, o) in triplets:
                        G.add_edge(s.lower(), o.lower(), relation=r.lower())
                    personalization = {n: (1.0 if n in q_entity else 0.0) for n in G.nodes()}
                    ppr_scores = nx.pagerank(
                        G,
                        alpha=0.85,
                        personalization=personalization,
                        max_iter=100,
                        tol=1e-05
                    )
                    graph_feats = []
                    for (s, r, o) in triplets:
                        s_, o_ = s.lower(), o.lower()
                        ppr_s = ppr_scores.get(s_, 0.0)
                        ppr_o = ppr_scores.get(o_, 0.0)
                        graph_feats.append([ppr_s, ppr_o])

                    graph_feats = torch.tensor(graph_feats, dtype=torch.float32)
            except Exception as e:
                print(q_entity)
                print("Triplets: ", triplets)
                print("="*20)
                    
            self.precomputed_data.append({
                "question": entry["question"],
                "is_empty": entry["is_empty"],
                "q_entity": entry["q_entity"],
                "a_entity":entry["a_entity"],
                "answer": entry["answer"],
                "question_embedding": entry["question_embedding"],
                "topk_linearized_triplets": entry["topk_linearized_triplets"],
                "topk_linearized_triplet_embeddings": entry["topk_linearized_triplet_embeddings"],
                "topk_rel_data":entry["topk_rel_data"],
                "topK_rel_embeddings": entry["topK_rel_embeddings"],
                "graph_features": graph_feats.to(self.device)
            })
            
    def __len__(self):
        return len(self.precomputed_data)
    
    def __getitem__(self, idx):
        return self.precomputed_data[idx]
    
    
 


# In[4]:


class PathRankingModel(nn.Module):
    def __init__(self, hidden_size=384, device="cuda"):
        super().__init__()
        self.device=device
        self.hidden_size = hidden_size
        self.question_triplet_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1
        )
        self.question_relation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1
        )
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        #Triplet centic scorer
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3+2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        #  Relation-Centric Scorer
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3+2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Combiner Network
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2),  # Input: [tower_A_score, tower_B_score, graph_feats, attention_delta]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)   # Final score adjustment
        )

        # Temperature and baseline (unchanged)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, question_embed, triplet_embeds, relation_embeds, graph_scores):
        num_triplets = triplet_embeds.size(0)
        question_embed = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed

        # Existing attention computations (unchanged)
        triplet_attended, triplet_weights = self.question_triplet_attention(
            triplet_embeds, question_embed, question_embed
        )
        relation_attended, relation_weights = self.question_relation_attention(
            relation_embeds, question_embed, question_embed
        )
        triplet_weights = triplet_weights.squeeze(0).squeeze(1)
        relation_weights = relation_weights.squeeze(0).squeeze(1)

        # Gate computation (unchanged)
        question_expanded = question_embed.expand(num_triplets, -1)
        gate_input = torch.cat([question_expanded, triplet_embeds, relation_embeds], dim=-1)
        path_gates = self.gate_network(gate_input).squeeze(-1)
        #avg_ppr_scores = graph_scores.mean(dim=1)
        # Two-Tower Scoring
        # Tower A: Triplet-centric score
        triplet_centric_input = torch.cat([
            triplet_embeds,
            triplet_attended,
            question_expanded,
            graph_scores
        ], dim=-1)
        tower_A_scores = self.triplet_mlp(triplet_centric_input).squeeze(-1)
        #print("tower_A_scores: ",tower_A_scores.shape)
        # Tower B: Relation-centric score
        relation_centric_input = torch.cat([
            relation_embeds,
            relation_attended,
            question_expanded,
            graph_scores
        ], dim=-1)
        tower_B_scores = self.relation_mlp(relation_centric_input).squeeze(-1)
        #print("tower_B_scores: ",tower_B_scores.shape)
        combiner_input = torch.stack([
            tower_A_scores,
            tower_B_scores,
            path_gates,
        ], dim=-1)
        combined_scores = self.combiner_mlp(combiner_input).squeeze(-1)
        #print("combined_scores: ",combined_scores.shape)
        # Final scoring with temperature
        temp = self.temperature.clamp(min=0.1, max=5.0)
        path_probs = F.softmax(combined_scores / temp, dim=0)

        return combined_scores, path_probs

    
    def sample_paths(self, probabilities: torch.Tensor, paths: List[str], k: int, ranking_scores) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k paths using categorical sampling for REINFORCE"""
        # Handle the case where we have fewer paths than k
        if len(paths) <= k:
            # For the case where len(paths) <= k, we need to ensure log_probs has gradients
            log_probs = torch.log(probabilities + 1e-10)  # Add small epsilon to avoid log(0)
            indices = torch.arange(len(paths), device=probabilities.device)
            return paths, probabilities, ranking_scores, log_probs
        dist = torch.distributions.Categorical(probs=probabilities)
        # Sample without replacement
        selected_indices = []
        log_probs_list = []
        remaining_indices = torch.ones(len(probabilities), dtype=torch.bool, device=probabilities.device)
        
        for _ in range(min(k, len(paths))):
            # Create masked probabilities
            masked_probs = probabilities * remaining_indices.float()
            # Re-normalize
            masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
            # Create distribution and sample
            masked_dist = torch.distributions.Categorical(probs=masked_probs)
            idx = masked_dist.sample()
            # Store log probability with gradient
            log_prob = masked_dist.log_prob(idx)
            # Update tracking
            selected_indices.append(idx.item())
            log_probs_list.append(log_prob)
            # Mark as used
            remaining_indices[idx] = False
        
        # Convert indices to tensor
        selected_indices_tensor = torch.tensor(selected_indices, device=probabilities.device)
        
        # Stack log probabilities
        log_probs = torch.stack(log_probs_list)
        
        # Get selected paths
        selected_paths = [paths[i] for i in selected_indices]
        selected_probs = probabilities[selected_indices_tensor]
        selected_ranking_scores = ranking_scores[selected_indices_tensor]
        
        return selected_paths, selected_probs, selected_ranking_scores, log_probs
    
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        path_state = {
            'question_triplet_attention': self.question_triplet_attention.state_dict(),
            'question_relation_attention': self.question_relation_attention.state_dict(),
            "gate_network": self.gate_network.state_dict(),
            "triplet_mlp": self.triplet_mlp.state_dict(),
            "relation_mlp": self.relation_mlp.state_dict(),
            "combiner_mlp": self.combiner_mlp.state_dict(),
            'temperature': self.temperature.detach().cpu(),
            'baseline': self.baseline.detach().cpu()
        }
        torch.save(path_state, os.path.join(save_directory, "path_ranker.pt"))
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model using HuggingFace from_pretrained"""
        model = cls()
        extra_state = torch.load(os.path.join(load_directory, "path_ranker.pt"))
        model.question_triplet_attention.load_state_dict(extra_state['question_triplet_attention'])
        model.question_relation_attention.load_state_dict(extra_state['question_relation_attention'])
        model.gate_network.load_state_dict(extra_state['gate_network'])
        model.triplet_mlp.load_state_dict(extra_state['triplet_mlp'])
        model.relation_mlp.load_state_dict(extra_state['relation_mlp'])
        model.combiner_mlp.load_state_dict(extra_state['combiner_mlp'])
        model.temperature.data = extra_state['temperature'].to(model.device)
        model.baseline.data = extra_state['baseline'].to(model.device)
        return model


# In[5]:


# training monitor
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
import json
import os

class TrainingMonitor:
    def __init__(self, save_dir="training_logs"): 
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics storage
        self.metrics = {
            'train_loss': [],
            'train_reward': [],
            'train_reinforce_loss': [],
            'val_loss': [],
            'val_reward': [],
            'val_reinforce_loss': [],
            'learning_rate': [],
            'baseline_value': [],
            'temperature': [],
            'selection_entropy': [],
            'gradient_norm': [],
            'epochs': []
        }
        
        # Detailed per-batch metrics (for smoothing)
        self.batch_metrics = {
            'batch_rewards': [],
            'batch_losses': [],
            'batch_numbers': []
        }
    
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics, optimizer, model):
        """Log metrics for an epoch"""
        self.metrics['epochs'].append(epoch)
        
        # Training metrics
        self.metrics['train_loss'].append(train_metrics.get('loss', 0))
        self.metrics['train_reward'].append(train_metrics.get('reward', 0))
        self.metrics['train_reinforce_loss'].append(train_metrics.get('reinforce_loss', 0))
        
        # Validation metrics
        self.metrics['val_loss'].append(val_metrics.get('loss', 0))
        self.metrics['val_reward'].append(val_metrics.get('reward', 0))
        self.metrics['val_reinforce_loss'].append(val_metrics.get('reinforce_loss', 0))
        
        # Model metrics
        self.metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        self.metrics['baseline_value'].append(model.baseline.item())
        self.metrics['temperature'].append(model.temperature.item())
        
        # Calculate selection entropy (measure of exploration)
        if hasattr(model, 'last_path_probs'):
            entropy = -torch.sum(model.last_path_probs * torch.log(model.last_path_probs + 1e-10))
            self.metrics['selection_entropy'].append(entropy.item())
        else:
            self.metrics['selection_entropy'].append(0)
    
    def log_batch_metrics(self, batch_num, reward, loss):
        """Log per-batch metrics for detailed analysis"""
        self.batch_metrics['batch_numbers'].append(batch_num)
        self.batch_metrics['batch_rewards'].append(reward)
        self.batch_metrics['batch_losses'].append(loss)
    
    def log_gradient_norm(self, model):
        """Log gradient norms for debugging"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.metrics['gradient_norm'].append(total_norm)
    
    def plot_training_progress(self, save_plots=True, show_plots=True):
        """Generate comprehensive training progress plots""" 
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress Monitoring', fontsize=16, fontweight='bold')
        
        epochs = self.metrics['epochs']
        
        # 1. Loss curves
        ax1 = axes[0, 0]
        if self.metrics['train_loss']:
            ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.metrics['val_loss']:
            ax1.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward curves (MOST IMPORTANT)
        ax2 = axes[0, 1]
        if self.metrics['train_reward']:
            ax2.plot(epochs, self.metrics['train_reward'], 'b-', label='Train Reward', linewidth=2)
        if self.metrics['val_reward']:
            ax2.plot(epochs, self.metrics['val_reward'], 'r-', label='Val Reward', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Answer Coverage (Reward)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        #ax2.set_ylim(0, 1.0)  # Rewards are 0-1
        
        # 3. REINFORCE loss
        ax3 = axes[0, 2]
        if self.metrics['train_reinforce_loss']:
            ax3.plot(epochs, self.metrics['train_reinforce_loss'], 'g-', label='REINFORCE Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('REINFORCE Loss')
        ax3.set_title('REINFORCE Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning rate and baseline
        ax4 = axes[1, 0]
        ax4_twin = ax4.twinx()
        
        if self.metrics['learning_rate']:
            line1 = ax4.plot(epochs, self.metrics['learning_rate'], 'purple', label='Learning Rate', linewidth=2)
        if self.metrics['baseline_value']:
            line2 = ax4_twin.plot(epochs, self.metrics['baseline_value'], 'orange', label='Baseline', linewidth=2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate', color='purple')
        ax4_twin.set_ylabel('Baseline Value', color='orange')
        ax4.set_title('Learning Rate & Baseline')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Temperature and entropy
        ax5 = axes[1, 1]
        ax5_twin = ax5.twinx()
        
        if self.metrics['temperature']:
            line1 = ax5.plot(epochs, self.metrics['temperature'], 'red', label='Temperature', linewidth=2)
        if self.metrics['selection_entropy']:
            line2 = ax5_twin.plot(epochs, self.metrics['selection_entropy'], 'blue', label='Selection Entropy', linewidth=2)
        
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Temperature', color='red')
        ax5_twin.set_ylabel('Selection Entropy', color='blue')
        ax5.set_title('Temperature & Exploration')
        
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # 6. Gradient norms
        ax6 = axes[1, 2]
        if self.metrics['gradient_norm']:
            ax6.plot(epochs, self.metrics['gradient_norm'], 'brown', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Gradient Norm')
        ax6.set_title('Gradient Magnitude')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')  # Log scale for gradient norms
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.save_dir}/training_progress.png', dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {self.save_dir}/training_progress.png")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_batch_level_analysis(self, window_size=100):
        """Plot detailed batch-level metrics with smoothing"""
        if not self.batch_metrics['batch_rewards']:
            print("No batch-level metrics to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        batch_nums = self.batch_metrics['batch_numbers']
        rewards = self.batch_metrics['batch_rewards']
        losses = self.batch_metrics['batch_losses']
        
        # Smooth the metrics using moving average
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(rewards) > window_size:
            smooth_rewards = moving_average(rewards, window_size)
            smooth_losses = moving_average(losses, window_size)
            smooth_batches = batch_nums[window_size-1:]
            
            ax1.plot(batch_nums, rewards, alpha=0.3, color='blue', label='Raw Rewards')
            ax1.plot(smooth_batches, smooth_rewards, color='blue', linewidth=2, label=f'Smoothed (window={window_size})')
        else:
            ax1.plot(batch_nums, rewards, color='blue', linewidth=2, label='Batch Rewards')
        
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Reward')
        ax1.set_title('Batch-Level Reward Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Loss plot
        if len(losses) > window_size:
            ax2.plot(batch_nums, losses, alpha=0.3, color='red', label='Raw Losses')
            ax2.plot(smooth_batches, smooth_losses, color='red', linewidth=2, label=f'Smoothed (window={window_size})')
        else:
            ax2.plot(batch_nums, losses, color='red', linewidth=2, label='Batch Losses')
        
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Loss')
        ax2.set_title('Batch-Level Loss Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/batch_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_metrics(self):
        """Save metrics to JSON for later analysis"""
        with open(f'{self.save_dir}/training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        with open(f'{self.save_dir}/batch_metrics.json', 'w') as f:
            json.dump(self.batch_metrics, f, indent=2)
    
    def print_training_summary(self):
        """Print a summary of training progress"""
        if not self.metrics['epochs']:
            print("No training metrics available")
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        last_epoch = self.metrics['epochs'][-1]
        print(f"Epochs Trained: {last_epoch}")
        
        if self.metrics['train_reward']:
            print(f"Final Train Reward: {self.metrics['train_reward'][-1]:.4f}")
            print(f"Best Train Reward: {max(self.metrics['train_reward']):.4f}")
        
        if self.metrics['val_reward']:
            print(f"Final Val Reward: {self.metrics['val_reward'][-1]:.4f}")
            print(f"Best Val Reward: {max(self.metrics['val_reward']):.4f}")
        
        if self.metrics['baseline_value']:
            print(f"Final Baseline: {self.metrics['baseline_value'][-1]:.4f}")
        
        if self.metrics['temperature']:
            print(f"Final Temperature: {self.metrics['temperature'][-1]:.4f}")
        
        # Check for potential issues
        print("\nTraining Health Check:")
        if self.metrics['val_reward'] and max(self.metrics['val_reward']) < 0.3:
            print("⚠️  Low validation rewards - consider longer training or check data")
        
        if self.metrics['gradient_norm'] and self.metrics['gradient_norm'][-1] < 1e-6:
            print("⚠️  Very small gradients - possible vanishing gradient problem")
        
        if self.metrics['gradient_norm'] and self.metrics['gradient_norm'][-1] > 10:
            print("⚠️  Large gradients - consider gradient clipping")
        
        print("="*60)


# In[6]:


from tqdm import tqdm
import unicodedata
from difflib import SequenceMatcher
import Levenshtein

reward_error_list= []
count_warning = 0
count_error = 0
class JointTrainer:
    def __init__(
        self,
        path_ranker: PathRankingModel,
        reward_func,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 16,
        checkpoint_dir: str = "checkpoints",
        gamma=0.99,
        baseline_decay=0.9
    ):
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
        # Add reward buffer for accumulation
        self.reward_buffer = []
        # Track best validation loss
        self.best_val_reward = float('-inf')
        self.device = device
        
    def compute_reinforce_loss(self, log_probs, rewards, baseline):
        """
        Compute REINFORCE loss with baseline for variance reduction
        
        Args:
            log_probs: Log probabilities of selected actions (paths)
            rewards: Rewards for selected actions (negative LLM loss)
            baseline: Optional baseline value for variance reduction
            
        Returns:
            REINFORCE loss
        """
        # If no explicit baseline is provided, use the running average
        if baseline is None:
            baseline = self.path_ranker.baseline.detach()
            print("baseline was none: ", baseline)
        # Calculate advantages
        advantages = rewards - baseline
        # REINFORCE loss is negative expected reward
        # We're maximizing expected reward, so we negate it for gradient descent
        reinforce_loss = -(log_probs * advantages.detach()).mean()
        return reinforce_loss

    
    def update_baseline_with_buffer(self):
        if len(self.reward_buffer) > 0:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)

            # Much more conservative baseline updates
            if self.running_baseline == 0:
                self.running_baseline = avg_reward * 0.8  # Start below actual rewards
            else:
                # Only update if we're significantly off, and do it slowly
                error = avg_reward - self.running_baseline
                if abs(error) > 0.5:  # Only update for significant differences
                    self.running_baseline += 0.1 * error  # Very slow updates

            # Cap baseline to prevent overshoot
            max_reasonable_baseline = avg_reward * 0.9
            self.running_baseline = min(self.running_baseline, max_reasonable_baseline)

            self.path_ranker.baseline.data = torch.tensor([self.running_baseline], device=self.device)
            self.reward_buffer = []
    
        
        
    def train_step( self, batch: Dict[str, torch.Tensor], k: int = 10 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single training step"""
        post = batch['question']
        global count_warning, count_error
        paths = [p[0] for p in batch['topk_linearized_triplets']]
        answer = [p[0] for p in batch["answer"]]
        ques_embed = batch["question_embedding"].to(self.device)
        linearized_triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(self.device)
        relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(self.device)
        q_entity = [p[0] for p in batch['q_entity']]
        a_entity = [p[0] for p in batch['a_entity']]
        triplets = [(data[1][0][0], data[1][1][0], data[1][2][0]) for data in batch["topk_rel_data"]]
        graph_features = batch["graph_features"].squeeze(0).to(self.device)

        #print(ques_embed.shape,linearized_triplet_embeds.shape,relation_embeds.shape)
        if len(q_entity)==0:
            return None,None.to(self.device)
        ranking_scores, path_probs = self.path_ranker(ques_embed, linearized_triplet_embeds, relation_embeds, graph_features)
        #selected_paths, selected_probs, selected_ranking_scores, log_probs = self.path_ranker.sample_paths(path_probs, paths, k, ranking_scores)
        selected_triplets, selected_probs, selected_ranking_scores, log_probs = self.path_ranker.sample_paths(path_probs, triplets, k, ranking_scores)
        answer_reward = self.reward_func(selected_triplets, q_entity, a_entity)
        
        #answer_reward = compute_reward_v2(post, selected_triplets, q_entity, a_entity)
        if answer_reward is None:
            return None, None
        reward = torch.tensor([answer_reward], device=self.device)
        # Add to reward buffer for baseline update
        self.reward_buffer.append(reward.item())
        #path_importance = -lama_loss * selected_probs
        reinforcement_loss = self.compute_reinforce_loss(log_probs, reward.expand(log_probs.size(0)), torch.tensor([self.running_baseline], device=reward.device))
        # Total loss is LLM loss + REINFORCE loss
        #total_loss = reinforcement_loss
#         print("reward ",reward)
#         print("reinforcement_loss ", reinforcement_loss)
        return reinforcement_loss, reward
    
    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader, k: int = 10) -> float:
        """Run validation loop"""
        self.path_ranker.eval()
        total_loss = 0
        total_ranking_loss = 0
        total_reward = 0
        valid_samples = 0

        for batch in tqdm(val_dataloader, desc="Validation"):
            loss, reward = self.train_step(batch, k)
            if loss is None:
                continue
            total_loss += loss.item()
            total_reward += reward.item()
            valid_samples += 1

        avg_loss = total_loss / valid_samples if valid_samples > 0 else 0
        avg_reward = total_reward / valid_samples if valid_samples > 0 else 0

        return avg_loss, avg_reward
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        print("saving model in epoch:{} and is best:{}".format(str(epoch),str(is_best)))
        if is_best:
            print("Best model with validation scores: ", val_loss)
            save_dir = os.path.join(self.checkpoint_dir, f"checkpoint_best_epoch_{epoch}")
        else:
            save_dir = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        self.path_ranker.save_pretrained(save_dir)
        training_state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        print("Saved!!!")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str):
        """Load checkpoint using HuggingFace methods"""
        path_ranker = PathRankingModel.from_pretrained(checkpoint_dir)
        model = cls(path_ranker,reward_func=None)
        training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"))
        model.best_val_loss = training_state['best_val_loss']
        
        print(training_state['epoch'], training_state['val_loss'])
        return model
    
    def train(
        self,
        monitor:TrainingMonitor,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 3,
        k: int = 20,
        learning_rate: float = 1e-5,
        warmup_steps: int = 1000,
        scheduler_type: str = 'linear',
        validation_interval: int = 1,
        early_stopping_patience: int = 3,
    ):
        optimizer = torch.optim.AdamW([
            {'params': self.path_ranker.parameters(), 'lr': learning_rate}
        ])
        total_steps = (len(train_dataloader) * num_epochs) // self.gradient_accumulation_steps
        scheduler_func = get_cosine_schedule_with_warmup if scheduler_type == 'cosine' else get_linear_schedule_with_warmup
        scheduler = scheduler_func(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.best_val_loss = float("inf")
        #self.model.eval()
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Training...")
            self.path_ranker.train()
            
            
            epoch_rewards = []
            epoch_losses = []
            epoch_reinforce_losses = []
            epoch_advantages = []  # Fixed variable name
            epoch_baselines = []   # Track baseline values too
            
            total_train_loss = 0
            optimizer.zero_grad()
            valid_batch_count =0
            
            with tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", unit="batch") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    loss, reward = self.train_step(batch, k)
                    if loss is None and reward is None:
                        continue
                    #print(reward)
                    if math.isnan(reward):
                        print("--------------------->", reward)
                    
                    # Get current baseline BEFORE updating it
                    current_baseline = self.path_ranker.baseline.detach().item() if hasattr(self.path_ranker, 'baseline') else self.running_baseline
                    advantage = reward.item() - current_baseline

                    # Store metrics
                    epoch_rewards.append(reward.item())
                    epoch_losses.append(loss.item())
                    epoch_reinforce_losses.append(loss.item())
                    epoch_advantages.append(advantage)
                    epoch_baselines.append(current_baseline)
                    
                    # Scale loss for gradient accumulation
                    valid_batch_count += 1
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    if valid_batch_count % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                        # Clip gradients before optimizer step
                        self.update_baseline_with_buffer()
                        torch.nn.utils.clip_grad_norm_(self.path_ranker.parameters(), self.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        monitor.log_batch_metrics(
                            epoch * len(train_dataloader) + batch_idx, 
                            reward.item(), 
                            loss.item()
                        )
                        
                    total_train_loss += loss.item()
                    if (batch_idx + 1) % (self.gradient_accumulation_steps * 10) == 0:
                        avg_loss = total_train_loss / (batch_idx + 1)
                        current_lr = scheduler.get_last_lr()[0]
                        pbar.set_postfix({
                                'train_loss': f"{avg_loss:.2f}",
                                'lr': f"{current_lr:.2e}"
                            })
                
            train_metrics = {
                'loss': np.mean(epoch_losses),
                'reward': np.mean(epoch_rewards),
                'reinforce_loss': np.mean(epoch_reinforce_losses)
            }

            if (epoch + 1) % validation_interval == 0 and val_dataloader is not None:
                val_loss, val_reward= self.validate(val_dataloader, k)
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Validation Loss: {val_loss:.4f}, "
                      f"Validation reward: {val_reward:.4f}")
                
                val_metrics = {
                    'loss': val_loss,
                    'reward': val_reward,
                    'reinforce_loss': val_loss
                }
                monitor.log_epoch_metrics(epoch+1, train_metrics, val_metrics, optimizer, self.path_ranker)
                monitor.log_gradient_norm(self.path_ranker)


                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    patience_counter = 0
                    self.save_checkpoint(epoch + 1, val_loss, is_best=True)
                else:
                    self.save_checkpoint(epoch + 1, val_loss, is_best=False)
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Average Reward: {np.mean(epoch_rewards):.4f}")
            print(f"  Average Advantage: {np.mean(epoch_advantages):.4f}")
            print(f"  Advantage Std: {np.std(epoch_advantages):.4f}")
            print(f"  Final Baseline: {self.running_baseline:.4f}")
            global reward_error, reward_error_list
            print("error count", reward_error)
            reward_error_list.append(reward_error)
            reward_error=0
            plt.figure(figsize=(15, 5))

            # Plot 1: Advantage distribution
            plt.subplot(1, 3, 1)
            plt.hist(epoch_advantages, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(epoch_advantages), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(epoch_advantages):.3f}')
            plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
            plt.title(f"Epoch {epoch+1} — Advantage Distribution")
            plt.xlabel("Advantage (Reward - Baseline)")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2: Reward vs Baseline over time
            plt.subplot(1, 3, 2)
            batch_indices = range(len(epoch_rewards))
            plt.plot(batch_indices, epoch_rewards, alpha=0.6, label='Rewards')
            plt.plot(batch_indices, epoch_baselines, alpha=0.8, label='Baseline')
            plt.title(f"Epoch {epoch+1} — Rewards vs Baseline")
            plt.xlabel("Batch Index")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 3: Advantage over time
            plt.subplot(1, 3, 3)
            plt.plot(batch_indices, epoch_advantages, alpha=0.7, color='green')
            plt.axhline(0, color='black', linestyle='-', alpha=0.5)
            plt.title(f"Epoch {epoch+1} — Advantage Over Time")
            plt.xlabel("Batch Index")
            plt.ylabel("Advantage")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Additional statistics
            if len(epoch_advantages) > 0:
                print(f"  Advantage Range: [{np.min(epoch_advantages):.4f}, {np.max(epoch_advantages):.4f}]")
                print(f"  Positive Advantages: {np.sum(np.array(epoch_advantages) > 0)} / {len(epoch_advantages)}")

        monitor.plot_training_progress(save_plots=True, show_plots=True)
        monitor.plot_batch_level_analysis()
        monitor.save_metrics()
        monitor.print_training_summary()


# # New Analysis

# In[23]:


import json
import networkx as nx
import torch
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class SampledJointTrainingDataset(Dataset):
    """Dataset class for loading test data"""
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if data["topk_linearized_triplet_embeddings"].shape[0] <= self.k:
            use_k = data["topk_linearized_triplet_embeddings"].shape[0]
        else:
            use_k = self.k
        
        return {
            "question": data["question"],
            "is_empty": data["is_empty"],
            "q_entity": data["q_entity"],
            "a_entity": data["a_entity"],
            "answer": data["answer"],
            "question_embedding": data["question_embedding"],
            "topk_linearized_triplets": data["topk_linearized_triplets"][:use_k],
            "topk_linearized_triplet_embeddings": data["topk_linearized_triplet_embeddings"][:use_k],
            "topk_rel_data": data["topk_rel_data"][:use_k],
            "topK_rel_embeddings": data["topK_rel_embeddings"][:use_k],
            "graph_features": data["graph_features"][:use_k]
        }
def linearize_triplet(triplet):
    """Linearize a triplet for display"""
    try:
        sub = f"{triplet[0]}"
        rel = f"{triplet[1]}"
        obj = f"{triplet[2]}"
        relation_split = " ".join(rel.split("."))
        relation_split2 = " ".join(relation_split.split("_"))
        return f"{sub} {relation_split2} {obj}"
    except Exception as e:
        print(f"Error in forming triplet: {triplet}")
        return str(triplet)

def create_graph(triplets, directed=True):
    """Create a graph from triplets"""
    G = nx.DiGraph() if directed else nx.Graph()
    for (s, p, o) in triplets:
        G.add_edge(s.lower(), o.lower(), relation=p.lower())
    return G

def check_answer_entity_presence(triplets, answer_entities):
    """Check if any answer entity is present in the triplets"""
    # answer_entities are already lowercased
    return any(
        ent in {s.lower(), o.lower()} 
        for ent in answer_entities 
        for (s, _, o) in triplets
    )

def check_reasoning_path_v1(graph, question_entities, answer_entities):
    """Find all shortest paths between all q_entity and a_entity pairs"""
    try:
        all_paths = []

        for q_ent in question_entities:
            if q_ent not in graph.nodes:
                continue

            for a_ent in answer_entities:
                if a_ent not in graph.nodes:
                    continue

                try:
                    if nx.has_path(graph, q_ent, a_ent):
                        path = nx.shortest_path(graph, q_ent, a_ent)
                        all_paths.append(path)
                except:
                    continue
        has_path = len(all_paths) > 0
        return has_path, all_paths
    except Exception as e:
        print("error in checking reasoning path", e)
        
def check_reasoning_path(graph, question_entities, answer_entities):
    try:
        f_all_paths=[]
        b_all_paths=[]
        f_has_path, f_all_paths = check_reasoning_path_v1(graph, question_entities, answer_entities)
        b_has_path, b_all_paths = check_reasoning_path_v1(graph, answer_entities, question_entities)
        #print("is f_all_paths None", f_all_paths)
        #print("is b_all_paths None", b_all_paths)
        f_all_paths.extend(b_all_paths)
        return  f_has_path or b_has_path, f_all_paths
    except Exception as e:
        print("Error in processing reasoning bidir path",e)
        return False , []


def get_path_triplets(graph, path):
    """Extract triplets from a path"""
    try:
        if not path or len(path) < 2:
            return []

        triplets = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            if graph.has_edge(source, target):
                relation = graph[source][target].get('relation', 'unknown')
                triplets.append((source, relation, target))
        return triplets
    except Exception as e:
        print("Error in gettin path triplets",e)

def calculate_triplet_overlap(triplets1, triplets2):
    """Calculate overlap between two sets of triplets"""
    # Normalize triplets to tuples
    try:
        set1 = set()
        for t in triplets1:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                set1.add((str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()))

        set2 = set()
        for t in triplets2:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                set2.add((str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()))

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if len(union) == 0:
            return 0.0, 0, 0

        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio, len(intersection), len(union)
    except Exception as e:
        print("error in computing overlap", e)

def extract_triplets_from_batch(batch, top_k, select_all=False):
    """Extract triplets from batch data"""
    try:
        triplets = [(data[1][0][0], data[1][1][0], data[1][2][0]) for data in batch["topk_rel_data"]]
        if select_all:
            return triplets
        return triplets[:top_k] if len(triplets) >= top_k else triplets
    except Exception as e:
        print(f"Error extracting triplets: {e}")
        return []

def analyze_comparison_with_model(
    test_dataloader,
    jointtrainer,
    top_k=30,
    batch_size=1,
    output_dir="analysis_results",
    device="cuda"
):
    """
    Main comparison analysis function using PyTorch dataset and KGScout model
    
    Args:
        test_dataset_path: Path to the saved PyTorch dataset (.pt file)
        jointtrainer: The KGScout model (JointTrainer instance)
        top_k: Number of top triplets to consider
        batch_size: Batch size for DataLoader (recommended: 1 for analysis)
        output_dir: Directory to save results
        device: Device to run model on ('cuda' or 'cpu')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Set model to eval mode
    torch.manual_seed(100)
    np.random.seed(100)
    jointtrainer.path_ranker.eval()
    
    # Initialize result containers
    case_1_1 = []  # Cosine (no relevant) vs KGScout (some relevant)
    case_1_2 = []  # Cosine (some relevant, no path) vs KGScout (reasoning path)
    case_2_1_non_overlapping = []  # Both relevant, non-overlapping paths
    case_2_1_overlapping = []  # Both relevant, overlapping paths
    case_5_cosine_better = []  # Cosine better than KGScout
    case_6_both_fail = []  # Both fail to find answer entities
    
    # Statistics
    stats = {
        'total_questions': 0,
        'case_1_1_count': 0,
        'case_1_2_count': 0,
        'case_2_1_non_overlapping_count': 0,
        'case_2_1_overlapping_count': 0,
        'case_5_cosine_better_count': 0,
        'case_6_both_fail_count': 0,
        'both_no_answer': 0,
        'only_cosine_has_answer': 0,
        'only_kgscout_has_answer': 0,
        'both_have_answer': 0,
        'processing_errors': 0
    }
    
    print("\n" + "="*70)
    print("ANALYZING TRIPLETS")
    print("="*70)
    
    # Process each batch
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Processing batches")):
            try:
               # Extract triplets first to check if valid
                    #print("1")
                    cosine_triplets = extract_triplets_from_batch(batch, top_k)
                    if len(cosine_triplets) == 0:
                        continue
                    #print("2")
                    # Extract basic info
                    #print( batch["question"])
                    #print(batch["answer"])
                    question = batch["question"][0] if isinstance(batch["question"], list) else batch["question"]
                    if len(batch["answer"]) ==0:
                        continue
                    answer = batch["answer"][0] if isinstance(batch["answer"], list) else batch["answer"]
                    #print("3")
                    # Extract entities exactly like reference code
                    ques_ents = [p[0].lower() for p in batch["q_entity"]]
                    ans_ents = [p[0].lower() for p in batch["a_entity"]]
                    #print("4")
                    # Skip if no entities
                    if len(ques_ents) == 0 or len(ans_ents) == 0:
                        continue

                    # Create cosine graph
                    G_cosine = create_graph(cosine_triplets, directed=True)

                    # Check answer entity presence in cosine triplets
                    cosine_has_answer = check_answer_entity_presence(cosine_triplets, ans_ents)

                    # Check reasoning path in cosine graph
                    cosine_has_path, cosine_path = check_reasoning_path(G_cosine, ques_ents, ans_ents)
                    # stats_list1.append(cosine_path)
                    #print("5")
                    # ================= MODEL-SELECTED TRIPLETS ====================
                    # Get embeddings
                    ques_embed = batch["question_embedding"].to(device)
                    triplet_embeds = batch["topk_linearized_triplet_embeddings"].squeeze(0).to(device)
                    relation_embeds = batch["topK_rel_embeddings"].squeeze(0).to(device)
                    graph_features = batch["graph_features"].squeeze(0).to(device)
                    #print("6")
                    # Get model predictions
                    ranking_scores, path_probs = jointtrainer.path_ranker(
                        ques_embed, triplet_embeds, relation_embeds, graph_features
                    )
                    #print("7")
                    # Sample paths using the model
                    # Extract ALL available triplets for the model to choose from
                    all_triplets = extract_triplets_from_batch(batch, 1000, select_all=True)
                    #print("8")
                    selected_triplets, selected_probs, selected_ranking_scores, log_probs = \
                        jointtrainer.path_ranker.sample_paths(
                            path_probs, all_triplets, top_k, ranking_scores
                        )
                    #print("9")
                    # Create KGScout graph
                    G_kgscout = create_graph(selected_triplets, directed=True)

                    # Check answer entity presence in KGScout triplets
                    kgscout_has_answer = check_answer_entity_presence(selected_triplets, ans_ents)

                    # Check reasoning path in KGScout graph
                    kgscout_has_path, kgscout_path = check_reasoning_path(G_kgscout, ques_ents, ans_ents)
#                     stats_list2.append(kgscout_path)
#                     count-=1
#                     if count<=0:
#                         break

                    # Update statistics
                    stats['total_questions'] += 1

                    if not cosine_has_answer and not kgscout_has_answer:
                        stats['both_no_answer'] += 1
                    elif cosine_has_answer and not kgscout_has_answer:
                        stats['only_cosine_has_answer'] += 1
                    elif not cosine_has_answer and kgscout_has_answer:
                        stats['only_kgscout_has_answer'] += 1
                    else:
                        stats['both_have_answer'] += 1

                    # ================= CASE CLASSIFICATION ====================
                    # Case 1.1: Cosine (no relevant) vs KGScout (some relevant)
                    
                    if not cosine_has_answer and not kgscout_has_answer:
                        case_6_both_fail.append({
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'kgscout_triplets': [linearize_triplet(t) for t in selected_triplets]
                        })
                        stats['case_6_both_fail_count'] += 1
                    
                    elif not cosine_has_answer and kgscout_has_answer:
                    #    print("B")
                        case_1_1.append({
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'kgscout_triplets': [linearize_triplet(t) for t in selected_triplets],
                            'kgscout_has_path': kgscout_has_path,
                            'kgscout_paths': [list(path) for path in kgscout_path] if kgscout_has_path else []
                        })
                     #   print("C")
                        stats['case_1_1_count'] += 1

                    # Case 1.2: Cosine (some relevant, no path) vs KGScout (reasoning path)
                    elif cosine_has_answer and not cosine_has_path and kgscout_has_path:
                        # Get all unique triplets from all KGScout paths
                     #   print("D")
                        kgscout_all_triplets = set()
                        for path in kgscout_path:
                            path_triplets = get_path_triplets(G_kgscout, path)
                            for t in path_triplets:
                                kgscout_all_triplets.add((str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()))

                        overlap_ratio, overlap_count, union_count = calculate_triplet_overlap(
                            cosine_triplets, list(kgscout_all_triplets)
                        )
                      #  print("E")
                        case_1_2.append({
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'kgscout_triplets': [linearize_triplet(t) for t in selected_triplets],
                            'cosine_has_answer_entity': True,
                            'cosine_has_path': False,
                            'kgscout_has_path': True,
                            'kgscout_paths': [list(path) for path in kgscout_path],
                            'kgscout_path_triplets': [linearize_triplet(t) for t in kgscout_all_triplets],
                            'triplet_overlap_ratio': overlap_ratio
                        })
                       # print("F")
                        stats['case_1_2_count'] += 1

                    # Case 2: Both have reasoning paths
                    elif cosine_has_path and kgscout_has_path:
                        # Get all triplets from all paths for both methods
#                         print("G")
#                         print(cosine_path)
                        cosine_all_triplets = set()
                        for path in cosine_path:
                            path_triplets = get_path_triplets(G_cosine, path)
                            #b_var = path_triplets is None
                            #print("-------CHEK-----", b_var)
                            for t in path_triplets:
                                # Normalize triplet to tuple for set operations
                                cosine_all_triplets.add((str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()))
                        #print("H")
                        kgscout_all_triplets = set()
                        for path in kgscout_path:
                            path_triplets = get_path_triplets(G_kgscout, path)
                            for t in path_triplets:
                                # Normalize triplet to tuple for set operations
                                kgscout_all_triplets.add((str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()))
                        #print("I")
                        # Calculate Jaccard similarity on triplets
                        intersection = cosine_all_triplets & kgscout_all_triplets
                        union = cosine_all_triplets | kgscout_all_triplets
                        overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0.0
                        overlap_count = len(intersection)
                        union_count = len(union)
                        #print("J")
                        # Convert all paths to serializable format
                        cosine_paths_serializable = [list(path) for path in cosine_path]
                        kgscout_paths_serializable = [list(path) for path in kgscout_path]

                        # Get linearized triplets from all paths
                        cosine_path_triplets_all = [linearize_triplet(t) for t in cosine_all_triplets]
                        kgscout_path_triplets_all = [linearize_triplet(t) for t in kgscout_all_triplets]
                        #print("K")
                        result_item = {
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_paths': cosine_paths_serializable,  # All paths
                            'kgscout_paths': kgscout_paths_serializable,  # All paths
                            'cosine_path_triplets': cosine_path_triplets_all,  # All unique triplets from all paths
                            'kgscout_path_triplets': kgscout_path_triplets_all,  # All unique triplets from all paths
                            'path_overlap_ratio': overlap_ratio,
                            'overlapping_triplets_count': overlap_count,
                            'union_triplets_count': union_count,
                            'all_cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'all_kgscout_triplets': [linearize_triplet(t) for t in selected_triplets]
                        }

                        # Case 2.1: Non-overlapping or partial overlap (< 0.8 overlap ratio)
                        if overlap_ratio <=0.3:
                            case_2_1_non_overlapping.append(result_item)
                            stats['case_2_1_non_overlapping_count'] += 1
                        # Case 2.2: High overlap (>= 0.8 overlap ratio)
                        elif overlap_ratio >= 0.7:
                            case_2_1_overlapping.append(result_item)
                            stats['case_2_1_overlapping_count'] += 1
                        #print("L")
                        
                    # Case 5: Cosine better than KGScout
                    # Sub-case 5a: Cosine has answer entity, KGScout doesn't
                    elif cosine_has_answer and not kgscout_has_answer:
                        case_5_cosine_better.append({
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'kgscout_triplets': [linearize_triplet(t) for t in selected_triplets],
                            'cosine_has_path': cosine_has_path,
                            'cosine_paths': [list(path) for path in cosine_path] if cosine_has_path else [],
                            'reason': 'cosine_has_answer_kgscout_does_not'
                        })
                        stats['case_5a_cosine_better_count'] += 1

                    # Sub-case 5b: Both have answer, but only Cosine has path
                    elif cosine_has_answer and kgscout_has_answer and cosine_has_path and not kgscout_has_path:
                        case_5_cosine_better.append({
                            'question': question,
                            'answer': answer,
                            'answer_entities': ans_ents,
                            'q_entity': ques_ents,
                            'cosine_triplets': [linearize_triplet(t) for t in cosine_triplets],
                            'kgscout_triplets': [linearize_triplet(t) for t in selected_triplets],
                            'cosine_has_path': True,
                            'cosine_paths': [list(path) for path in cosine_path],
                            'kgscout_has_answer': True,
                            'kgscout_has_path': False,
                            'reason': 'cosine_has_path_kgscout_does_not'
                        })
                        stats['case_5b_cosine_better_count'] += 1
                
            except Exception as e:
                print(f"\nError processing batch {i}: {e}")
                stats['processing_errors'] += 1
                continue
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    with open(os.path.join(output_dir, 'case_1_1_cosine_no_relevant_kgscout_some_relevant.json'), 'w', encoding='utf-8') as f:
        json.dump(case_1_1, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 1.1: {len(case_1_1)} instances")
    
    with open(os.path.join(output_dir, 'case_1_2_cosine_answer_only_kgscout_path.json'), 'w', encoding='utf-8') as f:
        json.dump(case_1_2, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 1.2: {len(case_1_2)} instances")
    
    with open(os.path.join(output_dir, 'case_2_1_both_paths_non_overlapping.json'), 'w', encoding='utf-8') as f:
        json.dump(case_2_1_non_overlapping, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 2.1 (non-overlapping): {len(case_2_1_non_overlapping)} instances")
    
    with open(os.path.join(output_dir, 'case_2_2_both_paths_overlapping.json'), 'w', encoding='utf-8') as f:
        json.dump(case_2_1_overlapping, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 2.2 (overlapping): {len(case_2_1_overlapping)} instances")
    
    with open(os.path.join(output_dir, 'case_5_cosine_better.json'), 'w', encoding='utf-8') as f:
        json.dump(case_5_cosine_better, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 5 (cosine better): {len(case_5_cosine_better)} instances")
    
    with open(os.path.join(output_dir, 'case_6_both_fail.json'), 'w', encoding='utf-8') as f:
        json.dump(case_6_both_fail, f, indent=2, ensure_ascii=False)
    print(f"Saved Case 6 (both fail): {len(case_6_both_fail)} instances")
    
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics")
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total questions analyzed: {stats['total_questions']}")
    print(f"Processing errors: {stats['processing_errors']}")
    print(f"\nAnswer Entity Presence:")
    print(f"  - Both have no answer entity: {stats['both_no_answer']}")
    print(f"  - Only Cosine has answer entity: {stats['only_cosine_has_answer']}")
    print(f"  - Only KGScout has answer entity: {stats['only_kgscout_has_answer']}")
    print(f"  - Both have answer entity: {stats['both_have_answer']}")
    print(f"\nCase Distribution:")
    print(f"  - Case 1.1 (Cosine: no relevant, KGScout: some relevant): {stats['case_1_1_count']}")
    print(f"  - Case 1.2 (Cosine: answer only, KGScout: path): {stats['case_1_2_count']}")
    print(f"  - Case 2.1 (Both paths, non-overlapping): {stats['case_2_1_non_overlapping_count']}")
    print(f"  - Case 2.2 (Both paths, overlapping): {stats['case_2_1_overlapping_count']}")
    print(f"  - Case 5 (Cosine better): {stats['case_5a_cosine_better_count']}")
    print(f"  - Case 5 (Cosine better): {stats['case_5b_cosine_better_count']}")
    print(f"  - Case 6 (Both fail): {stats['case_6_both_fail_count']}")
    print(f"\nVerification:")
    total_cases = (stats['case_1_1_count'] + stats['case_1_2_count'] + 
                   stats['case_2_1_non_overlapping_count'] + stats['case_2_1_overlapping_count'] +
                   stats['case_5_cosine_better_count'] + stats['case_6_both_fail_count'])
    print(f"  - Sum of all cases: {total_cases}")
    print(f"  - Total questions: {stats['total_questions']}")
    print(f"  - Match: {'✓' if total_cases == stats['total_questions'] else '✗ MISMATCH!'}")
    print("\n" + "="*70)
    print(f"Results saved to '{output_dir}/' directory")
    print("="*70)
    
    return stats


# In[24]:


cosine_pred_path = "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/results/architecture-v8/k30_cosine/predictions.jsonl"
kg_scout_pred_path = "/mnt/LS226/LS25/sourav23099/126vm-data/results/webqsp/v7-rv8-n1000-e30-k30_cosine/predictions.jsonl"
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30-k30_cosine/checkpoint_epoch_30")
analyze_comparison_with_model(tst_dataloader,
    joint_trainer_best,
    top_k=30,
    batch_size=1,
    output_dir="/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k30-bidir-analysis_results_v2",
    device="cuda")


# In[25]:


topk=50
cosine_pred_path = "/mnt/LS226/LS25/sourav23099/126vm-data/results/webqsp/k50-cosine/predictions.jsonl"
kg_scout_pred_path = "/mnt/LS226/LS25/sourav23099/126vm-data/results/webqsp/v7-rv8-n1000-e30-k50_cosine/predictions.jsonl"
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30-k50_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k50-bidir-analysis_results_v2"
analyze_comparison_with_model(tst_dataloader,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[26]:


topk=100
#cosine_pred_path = "/mnt/LS226/LS25/sourav23099/126vm-data/results/webqsp/k50-cosine/predictions.jsonl"
#kg_scout_pred_path = "/mnt/LS226/LS25/sourav23099/126vm-data/results/webqsp/v7-rv8-n1000-e30-k50_cosine/predictions.jsonl"
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k100-bidir-analysis_results_v2"
analyze_comparison_with_model(tst_dataloader,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[27]:


topk=150
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30-k150_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k150-analysis_results_v2"
analyze_comparison_with_model(tst_dataloader,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# ### CWQ

# In[29]:


topk=30
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/model/architecture-v8/v7-rv8-n1000-e30-k30_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k30-analysis_cwq_v2"
analyze_comparison_with_model(tst_dataloader_cwq,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[30]:


topk=50
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/model/architecture-v8/v7-rv8-n1000-e30-k50_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k50-analysis_cwq_v2"
analyze_comparison_with_model(tst_dataloader_cwq,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[31]:


topk=100
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/model/architecture-v8/v7-rv8-n1000-e30_cosine/checkpoint_best_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k100-analysis_cwq_v2"
analyze_comparison_with_model(tst_dataloader_cwq,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[32]:


topk=150
joint_trainer_best = JointTrainer.load_checkpoint("/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/model/architecture-v8/v7-rv8-n1000-e30-k150_cosine/checkpoint_epoch_30")
path = "/mnt/LS226/LS25/sourav23099/126vm-data/rv8-k150-analysis_cwq_v2"
analyze_comparison_with_model(tst_dataloader_cwq,
    joint_trainer_best,
    top_k=topk,
    batch_size=1,
    output_dir=path,
    device="cuda")


# In[ ]:




