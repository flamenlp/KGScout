"""
PathRankingModel for knowledge graph triplet ranking using attention mechanisms.
"""
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathRankingModel(nn.Module):
    """
    Neural model for ranking knowledge graph triplets using attention mechanisms.
    
    Architecture:
    - Question-triplet attention (8 heads, hidden_size=384)
    - Question-relation attention (8 heads, hidden_size=384)
    - Gate network (3-layer MLP with sigmoid output)
    - Triplet-centric scorer (3-layer MLP)
    - Relation-centric scorer (3-layer MLP)
    - Combiner network (2-layer MLP)
    - Learnable temperature and baseline parameters
    """
    
    def __init__(self, hidden_size: int = 384, device: str = "cuda"):
        """Initialize model with specified hidden size and device."""
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        
        # Question-triplet attention
        self.question_triplet_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            batch_first=True, 
            dropout=0.1
        )
        
        # Question-relation attention
        self.question_relation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            batch_first=True, 
            dropout=0.1
        )
        
        # Gate network (3-layer MLP with sigmoid output)
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
        
        # Triplet-centric scorer (3-layer MLP)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Relation-centric scorer (3-layer MLP)
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Combiner network (2-layer MLP)
        self.combiner_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Temperature and baseline as learnable parameters
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.baseline = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        question_embed: torch.Tensor,      # Shape: (1, hidden_size) or (hidden_size,)
        triplet_embeds: torch.Tensor,      # Shape: (num_triplets, hidden_size)
        relation_embeds: torch.Tensor,     # Shape: (num_triplets, hidden_size)
        graph_scores: torch.Tensor         # Shape: (num_triplets, 2) - PPR scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute ranking scores and probabilities.
        
        Returns:
            combined_scores: Raw ranking scores for each triplet
            path_probs: Softmax probabilities over triplets
        """
        num_triplets = triplet_embeds.size(0)
        
        # Ensure question_embed has batch dimension
        question_embed = question_embed.unsqueeze(0) if question_embed.dim() == 1 else question_embed
        
        # Compute attention between question and triplets
        triplet_attended, triplet_weights = self.question_triplet_attention(
            triplet_embeds, question_embed, question_embed
        )
        
        # Compute attention between question and relations
        relation_attended, relation_weights = self.question_relation_attention(
            relation_embeds, question_embed, question_embed
        )
        
        # Squeeze attention weights
        triplet_weights = triplet_weights.squeeze(0).squeeze(1)
        relation_weights = relation_weights.squeeze(0).squeeze(1)
        
        # Expand question to match number of triplets
        question_expanded = question_embed.expand(num_triplets, -1)
        
        # Compute gate values
        gate_input = torch.cat([question_expanded, triplet_embeds, relation_embeds], dim=-1)
        path_gates = self.gate_network(gate_input).squeeze(-1)
        
        # Tower A: Triplet-centric score
        triplet_centric_input = torch.cat([
            triplet_embeds,
            triplet_attended,
            question_expanded,
            graph_scores
        ], dim=-1)
        tower_A_scores = self.triplet_mlp(triplet_centric_input).squeeze(-1)
        
        # Tower B: Relation-centric score
        relation_centric_input = torch.cat([
            relation_embeds,
            relation_attended,
            question_expanded,
            graph_scores
        ], dim=-1)
        tower_B_scores = self.relation_mlp(relation_centric_input).squeeze(-1)
        
        # Combine scores using combiner MLP
        combiner_input = torch.stack([
            tower_A_scores,
            tower_B_scores,
            path_gates,
        ], dim=-1)
        combined_scores = self.combiner_mlp(combiner_input).squeeze(-1)
        
        # Apply temperature clamping and compute softmax probabilities
        temp = self.temperature.clamp(min=0.1, max=5.0)
        path_probs = F.softmax(combined_scores / temp, dim=0)
        
        return combined_scores, path_probs
    
    def sample_paths(
        self,
        probabilities: torch.Tensor,
        paths: List[str],
        k: int,
        ranking_scores: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample k paths using categorical sampling for REINFORCE.
        Implements sampling without replacement.
        
        Returns:
            selected_paths: List of k selected path strings
            selected_probs: Probabilities of selected paths
            selected_scores: Ranking scores of selected paths
            log_probs: Log probabilities for gradient computation
        """
        # Handle the case where we have fewer paths than k
        if len(paths) <= k:
            # For the case where len(paths) <= k, we need to ensure log_probs has gradients
            log_probs = torch.log(probabilities + 1e-10)  # Add small epsilon to avoid log(0)
            indices = torch.arange(len(paths), device=probabilities.device)
            return paths, probabilities, ranking_scores, log_probs
        
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
        """Save model state to directory."""
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
    def from_pretrained(cls, load_directory: str, device: str = "cuda") -> "PathRankingModel":
        """
        Load model from directory with error handling.
        
        Args:
            load_directory: Directory containing saved model
            device: Target device ('cuda' or 'cpu')
            
        Returns:
            Loaded PathRankingModel instance
            
        Raises:
            FileNotFoundError: If checkpoint file is missing
            ValueError: If model architecture mismatch during loading
            
        Error Handling:
            - Missing checkpoint files: Raises FileNotFoundError with descriptive message
            - Architecture mismatches: Raises ValueError with details about incompatible components
            - Device mismatches: Automatically maps to available device
        """
        checkpoint_path = os.path.join(load_directory, "path_ranker.pt")
        
        # Handle missing checkpoint file
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint file not found at: {checkpoint_path}\n"
                f"Please ensure the model was saved correctly and the path is correct."
            )
        
        # Handle device mismatch - automatically map to available device
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Mapping model to CPU.")
            device = "cpu"
        
        # Load checkpoint with automatic device mapping
        try:
            map_location = device if device == "cpu" else f"cuda:{torch.cuda.current_device()}"
            extra_state = torch.load(checkpoint_path, map_location=map_location)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load checkpoint from {checkpoint_path}. "
                f"The file may be corrupted or in an incompatible format.\n"
                f"Error: {str(e)}"
            )
        
        # Verify all required components are present
        required_components = [
            'question_triplet_attention',
            'question_relation_attention',
            'gate_network',
            'triplet_mlp',
            'relation_mlp',
            'combiner_mlp',
            'temperature',
            'baseline'
        ]
        
        missing_components = [comp for comp in required_components if comp not in extra_state]
        if missing_components:
            raise ValueError(
                f"Model architecture mismatch: checkpoint is missing required components: {missing_components}\n"
                f"This may indicate the checkpoint was saved with a different model version.\n"
                f"Expected components: {required_components}"
            )
        
        # Initialize model
        model = cls(device=device)
        
        # Load state dictionaries with error handling for architecture mismatches
        try:
            model.question_triplet_attention.load_state_dict(extra_state['question_triplet_attention'])
            model.question_relation_attention.load_state_dict(extra_state['question_relation_attention'])
            model.gate_network.load_state_dict(extra_state['gate_network'])
            model.triplet_mlp.load_state_dict(extra_state['triplet_mlp'])
            model.relation_mlp.load_state_dict(extra_state['relation_mlp'])
            model.combiner_mlp.load_state_dict(extra_state['combiner_mlp'])
            model.temperature.data = extra_state['temperature'].to(model.device)
            model.baseline.data = extra_state['baseline'].to(model.device)
        except RuntimeError as e:
            # This catches shape mismatches and other architecture incompatibilities
            raise ValueError(
                f"Model architecture mismatch: failed to load state dictionaries.\n"
                f"The checkpoint may have been saved with a different hidden_size or model architecture.\n"
                f"Current model hidden_size: {model.hidden_size}\n"
                f"Error details: {str(e)}"
            )
        except KeyError as e:
            raise ValueError(
                f"Model architecture mismatch: checkpoint has unexpected structure.\n"
                f"Missing key in state dictionary: {str(e)}"
            )
        
        return model
