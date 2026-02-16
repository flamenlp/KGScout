"""
Evaluation service for model evaluation.

This service handles loading trained models and computing evaluation metrics
on test data.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from model.path_ranker import PathRankingModel
from preprocess.joint_dataset import JointTrainingDatasetv3PPR
from testing.evaluator import Evaluator
from training.trainer import Trainer


class EvaluateService:
    """
    Service for evaluating model performance.
    
    Handles:
    - Loading trained model checkpoints
    - Processing test data
    - Computing evaluation metrics
    """
    
    def __init__(self, device: str = None):
        """
        Initialize evaluation service.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def _create_collate_fn():
        """Create collate function to filter None samples."""
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None
            return batch[0] if len(batch) == 1 else batch[0]
        return collate_fn
    
    def load_model(self, checkpoint_path: str) -> PathRankingModel:
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        
        Returns:
            Loaded PathRankingModel
        """
        print(f"Loading model checkpoint from {checkpoint_path}...")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}\n"
                f"Please ensure the model was trained and saved correctly."
            )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load checkpoint from {checkpoint_path}.\n"
                f"The file may be corrupted or in an incompatible format.\n"
                f"Error: {str(e)}"
            )
        
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint is missing 'model_state_dict' key.\n"
                f"The checkpoint may have been saved incorrectly or is corrupted.\n"
                f"Available keys: {list(checkpoint.keys())}"
            )
        
        # Create model and load weights
        path_ranker = PathRankingModel(hidden_size=384, device=self.device)
        
        try:
            path_ranker.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise ValueError(
                f"Model architecture mismatch when loading checkpoint.\n"
                f"The checkpoint may have been saved with a different model architecture or hidden_size.\n"
                f"Error details: {str(e)}"
            )
        
        path_ranker.to(self.device)
        print("Model loaded successfully!")
        
        return path_ranker
    
    def load_test_data(self, test_data_path: str):
        """
        Load and preprocess test data.
        
        Args:
            test_data_path: Path to test data file
        
        Returns:
            DataLoader for test data
        """
        print(f"\nLoading test data from {test_data_path}...")
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Create test dataset with PPR features
        print("\nCreating test dataset with PPR features...")
        test_dataset = JointTrainingDatasetv3PPR(test_data, device=self.device)
        
        # Create dataloader
        collate_fn = self._create_collate_fn()
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        return test_loader
    
    def evaluate(
        self,
        model_path: str,
        test_data_path: str,
        top_k: int
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model_path: Path to trained model checkpoint
            test_data_path: Path to test data file
            top_k: Number of top triplets to evaluate
        
        Returns:
            Dictionary with evaluation metrics:
                - answer_coverage: Answer coverage metric
                - path_coverage: Path coverage metric
                - average_reward: Average reward metric
        """
        print(f"Using device: {self.device}")
        
        # Load model
        path_ranker = self.load_model(model_path)
        
        # Load test data
        test_loader = self.load_test_data(test_data_path)
        
        # Create trainer (needed by evaluator)
        trainer = Trainer(
            path_ranker=path_ranker,
            checkpoint_dir="",
            device=self.device
        )
        
        # Create evaluator
        print("\nInitializing evaluator...")
        evaluator = Evaluator(device=self.device)
        
        # Run evaluation
        print(f"\nRunning evaluation with top-{top_k} triplets...")
        metrics = evaluator.evaluate_answer_and_path_coverage(
            test_dataloader=test_loader,
            trainer=trainer,
            top_k=top_k
        )
        
        return metrics
