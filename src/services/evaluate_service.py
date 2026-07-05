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

from src.model.path_ranker import PathRankingModel
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.testing.evaluator import Evaluator


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
        
        Supports both formats:
        - model_state_dict format (from old Trainer)
        - save_pretrained format (component-level state dicts from JointTrainer)
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
        
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
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load checkpoint from {checkpoint_path}.\n"
                f"The file may be corrupted or in an incompatible format.\n"
                f"Error: {str(e)}"
            )
        
        # Create model
        path_ranker = PathRankingModel(hidden_size=384, device=self.device)
        
        if "model_state_dict" in checkpoint:
            # Standard format with full state_dict
            path_ranker.load_state_dict(checkpoint["model_state_dict"])
        else:
            # save_pretrained format (component-level state dicts)
            for key, val in checkpoint.items():
                if key in ('temperature', 'baseline'):
                    getattr(path_ranker, key).data = val.to(self.device)
                elif hasattr(path_ranker, key):
                    getattr(path_ranker, key).load_state_dict(val)
        
        path_ranker.to(self.device)
        path_ranker.eval()
        print("Model loaded successfully!")
        
        return path_ranker
    
    def load_test_data(self, test_data_path: str):
        """
        Load and preprocess test data.
        
        Args:
            test_data_path: Path to test data file (.pt)
        
        Returns:
            DataLoader for test data
        """
        print(f"\nLoading test data from {test_data_path}...")

        import __main__
        __main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

        test_data = torch.load(test_data_path, weights_only=False, map_location="cpu")
        
        print(f"Loaded {len(test_data)} test samples")
        
        # If already a Dataset, use directly; otherwise wrap
        from torch.utils.data import Dataset
        if isinstance(test_data, Dataset):
            test_dataset = test_data
        else:
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
        
        # Create a lightweight wrapper — Evaluator only needs trainer.path_ranker
        class _ModelHolder:
            pass
        trainer = _ModelHolder()
        trainer.path_ranker = path_ranker
        
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
