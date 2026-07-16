"""
Preprocessing service for dataset preparation.

This service handles the preprocessing of datasets with PPR features.
"""

import os
import pickle
import torch
from typing import Dict, Any

from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR


class PreprocessService:
    """
    Service for preprocessing datasets with PPR features.
    
    Encapsulates the logic for loading raw data, computing PPR features,
    and saving preprocessed data.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize preprocessing service.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def preprocess(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Preprocess dataset with PPR features.
        
        Args:
            input_path: Path to input data file (pickle format)
            output_dir: Directory to save preprocessed data
        
        Returns:
            Dictionary with preprocessing results:
                - total_samples: Number of samples processed
                - skipped_samples: Number of samples skipped
                - output_path: Path to saved preprocessed data
        """
        print(f"Using device: {self.device}")
        
        # Load input data
        print(f"\nLoading input data from {input_path}...")
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        
        print(f"Loaded {len(input_data)} samples")
        
        # Create dataset with PPR features
        print("\nComputing PPR features...")
        dataset = JointTrainingDatasetv3PPR(input_data, device=self.device)
        
        print(f"Preprocessing complete! Processed {len(dataset)} samples")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessed data
        output_path = os.path.join(output_dir, "preprocessed_data.pkl")
        print(f"\nSaving preprocessed data to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset.precomputed_data, f)
        
        return {
            'total_samples': len(dataset),
            'skipped_samples': dataset.skipped_samples,
            'output_path': output_path
        }
