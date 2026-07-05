"""
Training service for model training pipeline.

This service handles the complete training pipeline including pretraining
and main training phases.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.model.path_ranker import PathRankingModel
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
from src.preprocess.pretrain_dataset import CosinePretrainingDataset
from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.training.pretrainer import Pretrainer
from src.training.trainer import Trainer


class TrainService:
    """
    Service for training PathRankingModel with two-phase pipeline.
    
    Handles:
    1. Pretraining phase (5 epochs, n=500, fixed)
    2. Loading pretrained weights
    3. Main training phase (configurable k parameter)
    """
    
    def __init__(self, device: str = None):
        """
        Initialize training service.
        
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
    
    def load_data(self, train_path: str, val_path: str) -> tuple:
        """
        Load training and validation data.
        
        Args:
            train_path: Path to training data file
            val_path: Path to validation data file
        
        Returns:
            Tuple of (train_data, val_data)
        """
        import __main__
        from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR
        __main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

        print("\nLoading training data...")
        train_data = torch.load(train_path, weights_only=False, map_location="cpu")
        
        print("Loading validation data...")
        val_data = torch.load(val_path, weights_only=False, map_location="cpu")
        
        return train_data, val_data
    
    def create_base_datasets(self, train_data, val_data):
        """
        Create base datasets with PPR features, or return as-is if already datasets.
        
        Args:
            train_data: Training data (raw list of dicts or JointTrainingDatasetv3PPR)
            val_data: Validation data (raw list of dicts or JointTrainingDatasetv3PPR)
        
        Returns:
            Tuple of (train_base_dataset, val_base_dataset)
        """
        from torch.utils.data import Dataset
        
        # If data is already a Dataset (loaded from .pt), use directly
        if isinstance(train_data, Dataset):
            print(f"\nTraining data is already a Dataset ({type(train_data).__name__}, {len(train_data)} samples)")
            train_base_dataset = train_data
        else:
            print("\nCreating base datasets with PPR features...")
            train_base_dataset = JointTrainingDatasetv3PPR(train_data, device=self.device)
        
        if isinstance(val_data, Dataset):
            print(f"Validation data is already a Dataset ({type(val_data).__name__}, {len(val_data)} samples)")
            val_base_dataset = val_data
        else:
            val_base_dataset = JointTrainingDatasetv3PPR(val_data, device=self.device)
        
        return train_base_dataset, val_base_dataset
    
    def run_pretraining(
        self,
        train_base_dataset,
        val_base_dataset,
        checkpoint_dir: str,
        learning_rate: float,
        weight_decay: float,
        gradient_accumulation_steps: int,
        validation_interval: int
    ) -> str:
        """
        Run pretraining phase.
        
        Args:
            train_base_dataset: Base training dataset
            val_base_dataset: Base validation dataset
            checkpoint_dir: Directory to save checkpoints
            learning_rate: Learning rate
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation steps
            validation_interval: Validation interval
        
        Returns:
            Path to best pretrained model checkpoint
        """
        print("\n" + "=" * 60)
        print("PHASE 1: PRETRAINING")
        print("=" * 60)
        print("Configuration: 5 epochs, n=500 (fixed)")
        
        # Create pretraining datasets
        pretrain_train_dataset = CosinePretrainingDataset(train_base_dataset, k=500)
        pretrain_val_dataset = CosinePretrainingDataset(val_base_dataset, k=500)
        
        # Create dataloaders
        collate_fn = self._create_collate_fn()
        pretrain_train_loader = DataLoader(
            pretrain_train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        )
        pretrain_val_loader = DataLoader(
            pretrain_val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize model
        print("\nInitializing PathRankingModel...")
        path_ranker = PathRankingModel(hidden_size=384, device=self.device)
        path_ranker.to(self.device)
        
        # Create checkpoint directory
        pretrain_checkpoint_dir = os.path.join(checkpoint_dir, "pretraining")
        os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
        
        # Initialize pretrainer
        pretrainer = Pretrainer(
            path_ranker=path_ranker,
            checkpoint_dir=pretrain_checkpoint_dir,
            device=self.device
        )
        
        # Run pretraining
        print("\nStarting pretraining phase...")
        pretrainer.train(
            train_dataloader=pretrain_train_loader,
            val_dataloader=pretrain_val_loader,
            num_epochs=5,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            validation_interval=validation_interval,
            save_best=True
        )
        
        print("\nPretraining phase complete!")
        
        # Return path to best checkpoint
        pretrained_model_path = os.path.join(pretrain_checkpoint_dir, "best_pretrained_model-5.pt")
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrain_checkpoint_dir, "best_pretrained_model.pt")
        
        return pretrained_model_path
    
    def load_pretrained_model(self, checkpoint_path: str) -> PathRankingModel:
        """
        Load pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
        
        Returns:
            PathRankingModel with pretrained weights
        """
        print("\n" + "=" * 60)
        print("PHASE 2: LOADING PRETRAINED MODEL")
        print("=" * 60)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Pretrained model checkpoint not found.\n"
                f"Expected location: {checkpoint_path}\n"
                f"Please ensure pretraining completed successfully."
            )
        
        print(f"Loading pretrained model from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load pretrained checkpoint from {checkpoint_path}.\n"
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
                f"Model architecture mismatch when loading pretrained weights.\n"
                f"The checkpoint may have been saved with a different model architecture.\n"
                f"Error details: {str(e)}"
            )
        
        path_ranker.to(self.device)
        print("Pretrained weights loaded successfully!")
        
        return path_ranker
    
    def run_main_training(
        self,
        path_ranker: PathRankingModel,
        train_base_dataset,
        val_base_dataset,
        checkpoint_dir: str,
        k: int,
        num_epochs: int,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        gradient_accumulation_steps: int,
        validation_interval: int,
        early_stopping_patience: int,
        sample_k: int = 1000
    ) -> Dict[str, str]:
        """
        Run main training phase.
        
        Args:
            path_ranker: PathRankingModel with pretrained weights
            train_base_dataset: Base training dataset
            val_base_dataset: Base validation dataset
            checkpoint_dir: Directory to save checkpoints
            k: Number of top triplets to select per question (selection size)
            num_epochs: Number of epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation steps
            validation_interval: Validation interval
            early_stopping_patience: Early stopping patience
            sample_k: Number of triplets to prefilter per question (pool size).
                      The model sees sample_k triplets and selects the top-k from them.
                      Default: 1000 (consistent with all ablation studies).
        
        Returns:
            Dictionary with paths to checkpoints and logs
        """
        print("\n" + "=" * 60)
        print("PHASE 3: MAIN TRAINING")
        print("=" * 60)
        print(f"Configuration: N={sample_k} (pool), k={k} (selection), {num_epochs} epochs")
        
        # Create main training datasets
        # sample_k controls the prefiltered pool size (N=1000 by default)
        # k controls how many triplets the model selects from the pool during REINFORCE
        main_train_dataset = SampledJointTrainingDataset(train_base_dataset, k=sample_k)
        main_val_dataset = SampledJointTrainingDataset(val_base_dataset, k=sample_k)
        
        # Create dataloaders (batch_size=1, default collate — same as model_ablation.py)
        main_train_loader = DataLoader(
            main_train_dataset,
            batch_size=1,
            shuffle=True,
        )
        main_val_loader = DataLoader(
            main_val_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        # Create checkpoint directory
        main_checkpoint_dir = os.path.join(checkpoint_dir, f"main_training_k{k}")
        os.makedirs(main_checkpoint_dir, exist_ok=True)
        
        # Initialize trainer (matches JointTrainer from ablation-2/model_ablation.py)
        trainer = Trainer(
            path_ranker=path_ranker,
            checkpoint_dir=main_checkpoint_dir,
            device=self.device,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # Run training
        print("\nStarting main training phase...")
        trainer.train(
            train_dataloader=main_train_loader,
            val_dataloader=main_val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            validation_interval=validation_interval,
            early_stopping_patience=early_stopping_patience,
            k=k,
        )
        
        print("\nMain training phase complete!")
        
        return {
            'checkpoint_dir': main_checkpoint_dir,
            'log_dir': main_checkpoint_dir
        }
    
    def train(
        self,
        train_data_path: str,
        val_data_path: str,
        checkpoint_dir: str,
        k: int,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        validation_interval: int = 1,
        early_stopping_patience: int = 10,
        sample_k: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            checkpoint_dir: Directory to save checkpoints
            k: K value for top-k selection during REINFORCE training
            num_epochs: Number of epochs for main training
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation steps
            validation_interval: Validation interval
            early_stopping_patience: Early stopping patience
            sample_k: Pool size (N) — number of prefiltered triplets per question.
                      The model sees sample_k triplets and selects top-k from them.
                      Default: 1000 (consistent with all ablation studies).
        
        Returns:
            Dictionary with training results and paths
        """
        print(f"Using device: {self.device}")
        
        # Load data
        train_data, val_data = self.load_data(train_data_path, val_data_path)
        
        # Create base datasets
        train_base_dataset, val_base_dataset = self.create_base_datasets(train_data, val_data)
        
        # Run pretraining
        pretrained_model_path = self.run_pretraining(
            train_base_dataset,
            val_base_dataset,
            checkpoint_dir,
            learning_rate,
            weight_decay,
            gradient_accumulation_steps,
            validation_interval
        )
        
        # Load pretrained model
        path_ranker = self.load_pretrained_model(pretrained_model_path)
        
        # Run main training
        training_results = self.run_main_training(
            path_ranker,
            train_base_dataset,
            val_base_dataset,
            checkpoint_dir,
            k,
            num_epochs,
            learning_rate,
            warmup_steps,
            weight_decay,
            gradient_accumulation_steps,
            validation_interval,
            early_stopping_patience,
            sample_k
        )
        
        return {
            'pretrain_checkpoint': os.path.join(checkpoint_dir, "pretraining"),
            'main_checkpoint': training_results['checkpoint_dir'],
            'log_dir': training_results['log_dir']
        }
