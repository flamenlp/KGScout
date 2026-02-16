"""
Inference service for model inference.

This service handles loading trained models and running inference
to select top-k triplets from test data.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from model.path_ranker import PathRankingModel
from preprocess.joint_dataset import JointTrainingDatasetv3PPR
from inference.predictor import Predictor


class InferenceService:
    """
    Service for running inference on test data.
    
    Handles:
    - Loading trained model checkpoints
    - Processing test data
    - Selecting top-k triplets
    - Saving results
    """
    
    def __init__(self, device: str = None):
        """
        Initialize inference service.
        
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
    
    def run_inference(
        self,
        model_path: str,
        test_data_path: str,
        output_dir: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Run inference on test data.
        
        Args:
            model_path: Path to trained model checkpoint
            test_data_path: Path to test data file
            output_dir: Directory to save results
            top_k: Number of top triplets to select
        
        Returns:
            Dictionary with inference results:
                - total_samples: Number of samples processed
                - average_reward: Average reward across samples
                - output_file: Path to saved results
        """
        print(f"Using device: {self.device}")
        
        # Load model
        path_ranker = self.load_model(model_path)
        
        # Load test data
        test_loader = self.load_test_data(test_data_path)
        
        # Create predictor
        print("\nInitializing predictor...")
        predictor = Predictor(model=path_ranker, device=self.device)
        
        # Run inference
        print(f"\nRunning inference to select top-{top_k} triplets...")
        results = predictor.predict(
            test_dataloader=test_loader,
            top_k=top_k,
            output_dir=output_dir
        )
        
        # Compute statistics
        avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0
        
        return {
            'total_samples': len(results),
            'average_reward': avg_reward,
            'output_file': os.path.join(output_dir, "inference_results.json")
        }
