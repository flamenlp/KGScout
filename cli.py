#!/usr/bin/env python3
"""
Command-line interface for ML project pipeline.

Provides commands for:
- preprocess_dataset: Prepare datasets with PPR features
- train: Run complete training pipeline (pretraining → main training)
- inference: Run inference on test data
- evaluate: Compute evaluation metrics
"""

import argparse
import os
import sys


def validate_file_exists(filepath: str, description: str) -> None:
    """
    Validate that a required input file exists.
    
    Args:
        filepath: Path to the file to validate
        description: Human-readable description of the file for error messages
    
    Raises:
        SystemExit: If file does not exist
    """
    if not os.path.exists(filepath):
        print(f"Error: {description} not found: {filepath}", file=sys.stderr)
        sys.exit(1)


def validate_directory_writable(dirpath: str, description: str) -> None:
    """
    Validate that a directory is writable (or can be created).
    
    Args:
        dirpath: Path to the directory to validate
        description: Human-readable description of the directory for error messages
    
    Raises:
        SystemExit: If directory is not writable
    """
    # Try to create the directory if it doesn't exist
    try:
        os.makedirs(dirpath, exist_ok=True)
    except PermissionError:
        print(f"Error: {description} cannot be created (permission denied): {dirpath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to create {description}: {dirpath}", file=sys.stderr)
        print(f"Reason: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Test if directory is writable
    if not os.access(dirpath, os.W_OK):
        print(f"Error: {description} is not writable: {dirpath}", file=sys.stderr)
        sys.exit(1)


def validate_train_arguments(args) -> None:
    """
    Validate training arguments for logical conflicts.
    
    Args:
        args: Parsed command-line arguments for train command
    
    Raises:
        SystemExit: If conflicting arguments are detected
    """
    # Check if early stopping patience is greater than or equal to num_epochs
    if args.early_stopping_patience >= args.num_epochs:
        print(f"Warning: Early stopping patience ({args.early_stopping_patience}) is >= num_epochs ({args.num_epochs}).", file=sys.stderr)
        print("Early stopping will have no effect. Consider reducing patience or increasing epochs.", file=sys.stderr)
    
    # Check if validation interval is greater than num_epochs
    if args.validation_interval > args.num_epochs:
        print(f"Error: Validation interval ({args.validation_interval}) cannot be greater than num_epochs ({args.num_epochs}).", file=sys.stderr)
        sys.exit(1)
    
    # Check if learning rate is reasonable
    if args.learning_rate <= 0:
        print(f"Error: Learning rate must be positive, got {args.learning_rate}", file=sys.stderr)
        sys.exit(1)
    
    if args.learning_rate > 1.0:
        print(f"Warning: Learning rate ({args.learning_rate}) is unusually high. Typical values are 1e-5 to 1e-3.", file=sys.stderr)
    
    # Check if weight decay is reasonable
    if args.weight_decay < 0:
        print(f"Error: Weight decay must be non-negative, got {args.weight_decay}", file=sys.stderr)
        sys.exit(1)
    
    # Check if gradient accumulation steps is positive
    if args.gradient_accumulation_steps <= 0:
        print(f"Error: Gradient accumulation steps must be positive, got {args.gradient_accumulation_steps}", file=sys.stderr)
        sys.exit(1)
    
    # Check if warmup steps is non-negative
    if args.warmup_steps < 0:
        print(f"Error: Warmup steps must be non-negative, got {args.warmup_steps}", file=sys.stderr)
        sys.exit(1)


def validate_inference_arguments(args) -> None:
    """
    Validate inference arguments for logical conflicts.
    
    Args:
        args: Parsed command-line arguments for inference command
    
    Raises:
        SystemExit: If conflicting arguments are detected
    """
    # Check if top_k is positive
    if args.top_k <= 0:
        print(f"Error: top_k must be positive, got {args.top_k}", file=sys.stderr)
        sys.exit(1)


def validate_evaluate_arguments(args) -> None:
    """
    Validate evaluation arguments for logical conflicts.
    
    Args:
        args: Parsed command-line arguments for evaluate command
    
    Raises:
        SystemExit: If conflicting arguments are detected
    """
    # Check if top_k is positive
    if args.top_k <= 0:
        print(f"Error: top_k must be positive, got {args.top_k}", file=sys.stderr)
        sys.exit(1)


def run_preprocess_command(args):
    """
    Execute preprocessing command to prepare datasets with PPR features.
    
    Args:
        args: Parsed command-line arguments
    """
    import torch
    import pickle
    from preprocess.joint_dataset import JointTrainingDatasetv3PPR
    
    # Validate input file exists
    validate_file_exists(args.input, "Input data file")
    
    # Validate output directory is writable
    validate_directory_writable(args.output, "Output directory")
    
    print("=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)
    print(f"Input data: {args.input}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load input data
    print(f"\nLoading input data from {args.input}...")
    with open(args.input, 'rb') as f:
        input_data = pickle.load(f)
    
    print(f"Loaded {len(input_data)} samples")
    
    # Create JointTrainingDatasetv3PPR with PPR features
    print("\nComputing PPR features...")
    dataset = JointTrainingDatasetv3PPR(input_data, device=device)
    
    print(f"Preprocessing complete! Processed {len(dataset)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save preprocessed data to output directory
    output_path = os.path.join(args.output, "preprocessed_data.pkl")
    print(f"\nSaving preprocessed data to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset.precomputed_data, f)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Preprocessed data saved to: {output_path}")
    print(f"Total samples: {len(dataset)}")
    if dataset.skipped_samples > 0:
        print(f"Skipped samples: {dataset.skipped_samples}")
    print("=" * 60)


def run_train_command(args):
    """
    Execute complete training pipeline:
    1. Run pretraining (5 epochs, n=500, fixed)
    2. Load best pretrained model checkpoint
    3. Run main training with specified k parameter
    
    Args:
        args: Parsed command-line arguments
    """
    import torch
    import pickle
    from torch.utils.data import DataLoader
    from model.path_ranker import PathRankingModel
    from preprocess.joint_dataset import JointTrainingDatasetv3PPR
    from preprocess.pretrain_dataset import CosinePretrainingDataset
    from preprocess.sampled_dataset import SampledJointTrainingDataset
    from training.pretrainer import Pretrainer
    from training.trainer import Trainer
    from training.monitor import TrainingMonitor
    
    # Validate input files exist
    validate_file_exists(args.train_data, "Training data file")
    validate_file_exists(args.val_data, "Validation data file")
    
    # Validate checkpoint directory is writable
    validate_directory_writable(args.checkpoint_dir, "Checkpoint directory")
    
    # Validate training arguments for conflicts
    validate_train_arguments(args)
    
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Main training k parameter: {args.k}")
    print(f"Main training epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load training and validation data
    print("\nLoading training data...")
    with open(args.train_data, 'rb') as f:
        train_data = pickle.load(f)
    
    print("Loading validation data...")
    with open(args.val_data, 'rb') as f:
        val_data = pickle.load(f)
    
    # Create base datasets with PPR features
    print("\nCreating base datasets with PPR features...")
    train_base_dataset = JointTrainingDatasetv3PPR(train_data, device=device)
    val_base_dataset = JointTrainingDatasetv3PPR(val_data, device=device)
    
    # ========================================================================
    # PHASE 1: PRETRAINING (5 epochs, n=500, fixed)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: PRETRAINING")
    print("=" * 60)
    print("Configuration: 5 epochs, n=500 (fixed)")
    
    # Create pretraining datasets with fixed k=500
    pretrain_train_dataset = CosinePretrainingDataset(train_base_dataset, k=500)
    pretrain_val_dataset = CosinePretrainingDataset(val_base_dataset, k=500)
    
    # Create dataloaders for pretraining
    # Use collate_fn to filter out None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return batch[0] if len(batch) == 1 else batch[0]
    
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
    
    # Initialize model for pretraining
    print("\nInitializing PathRankingModel...")
    path_ranker = PathRankingModel(hidden_size=384, device=device)
    path_ranker.to(device)
    
    # Create pretraining checkpoint directory
    pretrain_checkpoint_dir = os.path.join(args.checkpoint_dir, "pretraining")
    os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
    
    # Initialize pretrainer
    pretrainer = Pretrainer(
        path_ranker=path_ranker,
        checkpoint_dir=pretrain_checkpoint_dir,
        device=device
    )
    
    # Run pretraining
    print("\nStarting pretraining phase...")
    pretrainer.train(
        train_dataloader=pretrain_train_loader,
        val_dataloader=pretrain_val_loader,
        num_epochs=5,  # Fixed at 5 epochs
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_interval=args.validation_interval,
        save_best=True
    )
    
    print("\nPretraining phase complete!")
    
    # ========================================================================
    # PHASE 2: LOAD PRETRAINED MODEL
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: LOADING PRETRAINED MODEL")
    print("=" * 60)
    
    # Load best pretrained model checkpoint from epoch 5
    pretrained_model_path = os.path.join(pretrain_checkpoint_dir, "best_pretrained_model-5.pt")
    
    if not os.path.exists(pretrained_model_path):
        print(f"Warning: Checkpoint {pretrained_model_path} not found.")
        print("Trying to load best_pretrained_model.pt instead...")
        pretrained_model_path = os.path.join(pretrain_checkpoint_dir, "best_pretrained_model.pt")
    
    # Validate file exists with descriptive error
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(
            f"Pretrained model checkpoint not found.\n"
            f"Expected location: {pretrained_model_path}\n"
            f"Please ensure pretraining completed successfully."
        )
    
    print(f"Loading pretrained model from: {pretrained_model_path}")
    
    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(pretrained_model_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load pretrained checkpoint from {pretrained_model_path}.\n"
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {str(e)}"
        )
    
    # Verify checkpoint has model_state_dict
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint is missing 'model_state_dict' key.\n"
            f"The checkpoint may have been saved incorrectly or is corrupted.\n"
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Create new model and load pretrained weights with error handling
    path_ranker = PathRankingModel(hidden_size=384, device=device)
    
    try:
        path_ranker.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(
            f"Model architecture mismatch when loading pretrained weights.\n"
            f"The checkpoint may have been saved with a different model architecture.\n"
            f"Error details: {str(e)}"
        )
    
    path_ranker.to(device)
    
    print("Pretrained weights loaded successfully!")
    
    # ========================================================================
    # PHASE 3: MAIN TRAINING (with specified k parameter)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: MAIN TRAINING")
    print("=" * 60)
    print(f"Configuration: k={args.k}, {args.num_epochs} epochs")
    
    # Create main training datasets with configurable k
    main_train_dataset = SampledJointTrainingDataset(train_base_dataset, k=args.k)
    main_val_dataset = SampledJointTrainingDataset(val_base_dataset, k=args.k)
    
    # Create dataloaders for main training
    main_train_loader = DataLoader(
        main_train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    main_val_loader = DataLoader(
        main_val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create main training checkpoint directory
    main_checkpoint_dir = os.path.join(args.checkpoint_dir, f"main_training_k{args.k}")
    os.makedirs(main_checkpoint_dir, exist_ok=True)
    
    # Create training monitor
    monitor_log_dir = os.path.join(main_checkpoint_dir, "training_logs")
    monitor = TrainingMonitor(log_dir=monitor_log_dir)
    
    # Initialize trainer with pretrained model
    trainer = Trainer(
        path_ranker=path_ranker,
        checkpoint_dir=main_checkpoint_dir,
        device=device
    )
    
    # Run main training
    print("\nStarting main training phase...")
    trainer.train(
        train_dataloader=main_train_loader,
        val_dataloader=main_val_loader,
        monitor=monitor,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        k=args.k
    )
    
    print("\nMain training phase complete!")
    
    # ========================================================================
    # TRAINING PIPELINE COMPLETE
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Pretraining checkpoints: {pretrain_checkpoint_dir}")
    print(f"Main training checkpoints: {main_checkpoint_dir}")
    print(f"Training logs: {monitor_log_dir}")
    print("=" * 60)


def run_inference_command(args):
    """
    Execute inference to select top-k triplets from test data.
    
    Args:
        args: Parsed command-line arguments
    """
    import torch
    import pickle
    from torch.utils.data import DataLoader
    from model.path_ranker import PathRankingModel
    from preprocess.joint_dataset import JointTrainingDatasetv3PPR
    from inference.predictor import Predictor
    
    # Validate input files exist
    validate_file_exists(args.model_path, "Model checkpoint")
    validate_file_exists(args.test_data, "Test data file")
    
    # Validate output directory is writable
    validate_directory_writable(args.output_dir, "Output directory")
    
    # Validate inference arguments for conflicts
    validate_inference_arguments(args)
    
    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)
    print(f"Model checkpoint: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Create test dataset with PPR features
    print("\nCreating test dataset with PPR features...")
    test_dataset = JointTrainingDatasetv3PPR(test_data, device=device)
    
    # Create dataloader
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return batch[0] if len(batch) == 1 else batch[0]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model checkpoint
    print(f"\nLoading model checkpoint from {args.model_path}...")
    
    # Handle missing checkpoint file
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {args.model_path}\n"
            f"Please ensure the model was trained and saved correctly."
        )
    
    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load checkpoint from {args.model_path}.\n"
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {str(e)}"
        )
    
    # Verify checkpoint has model_state_dict
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint is missing 'model_state_dict' key.\n"
            f"The checkpoint may have been saved incorrectly or is corrupted.\n"
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Create model and load weights with error handling
    path_ranker = PathRankingModel(hidden_size=384, device=device)
    
    try:
        path_ranker.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(
            f"Model architecture mismatch when loading checkpoint.\n"
            f"The checkpoint may have been saved with a different model architecture or hidden_size.\n"
            f"Error details: {str(e)}"
        )
    
    path_ranker.to(device)
    
    print("Model loaded successfully!")
    
    # Create Predictor instance
    print("\nInitializing predictor...")
    predictor = Predictor(model=path_ranker, device=device)
    
    # Run inference on test data
    print(f"\nRunning inference to select top-{args.top_k} triplets...")
    results = predictor.predict(
        test_dataloader=test_loader,
        top_k=args.top_k,
        output_dir=args.output_dir
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {args.output_dir}")
    
    # Compute average reward
    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0
    print(f"Average reward: {avg_reward:.4f}")
    print("=" * 60)


def run_evaluate_command(args):
    """
    Execute evaluation to compute metrics on test data.
    
    Args:
        args: Parsed command-line arguments
    """
    import torch
    import pickle
    from torch.utils.data import DataLoader
    from model.path_ranker import PathRankingModel
    from preprocess.joint_dataset import JointTrainingDatasetv3PPR
    from testing.evaluator import Evaluator
    from training.trainer import Trainer
    
    # Validate input files exist
    validate_file_exists(args.model_path, "Model checkpoint")
    validate_file_exists(args.test_data, "Test data file")
    
    # Validate evaluation arguments for conflicts
    validate_evaluate_arguments(args)
    
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Model checkpoint: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Create test dataset with PPR features
    print("\nCreating test dataset with PPR features...")
    test_dataset = JointTrainingDatasetv3PPR(test_data, device=device)
    
    # Create dataloader
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return batch[0] if len(batch) == 1 else batch[0]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model checkpoint
    print(f"\nLoading model checkpoint from {args.model_path}...")
    
    # Handle missing checkpoint file
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {args.model_path}\n"
            f"Please ensure the model was trained and saved correctly."
        )
    
    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load checkpoint from {args.model_path}.\n"
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {str(e)}"
        )
    
    # Verify checkpoint has model_state_dict
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint is missing 'model_state_dict' key.\n"
            f"The checkpoint may have been saved incorrectly or is corrupted.\n"
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Create model and load weights with error handling
    path_ranker = PathRankingModel(hidden_size=384, device=device)
    
    try:
        path_ranker.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(
            f"Model architecture mismatch when loading checkpoint.\n"
            f"The checkpoint may have been saved with a different model architecture or hidden_size.\n"
            f"Error details: {str(e)}"
        )
    
    path_ranker.to(device)
    
    print("Model loaded successfully!")
    
    # Create Trainer instance (needed by Evaluator)
    # Note: We don't need checkpoint_dir for evaluation, just pass empty string
    trainer = Trainer(
        path_ranker=path_ranker,
        checkpoint_dir="",
        device=device
    )
    
    # Create Evaluator instance
    print("\nInitializing evaluator...")
    evaluator = Evaluator(device=device)
    
    # Run evaluation on test data
    print(f"\nRunning evaluation with top-{args.top_k} triplets...")
    metrics = evaluator.evaluate_answer_and_path_coverage(
        test_dataloader=test_loader,
        trainer=trainer,
        top_k=args.top_k
    )
    
    # Print metrics to console
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Answer Coverage:  {metrics['answer_coverage']:.4f}")
    print(f"Path Coverage:    {metrics['path_coverage']:.4f}")
    print(f"Average Reward:   {metrics['average_reward']:.4f}")
    print("=" * 60)


def main():
    """
    CLI entry point with subcommands for different pipeline stages.
    """
    parser = argparse.ArgumentParser(
        description="ML Project Pipeline - Knowledge Graph Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess dataset
  python cli.py preprocess_dataset --input data/train.json --output data/preprocessed/

  # Train model with k=50
  python cli.py train --train-data data/train.json --val-data data/val.json \\
                      --checkpoint-dir checkpoints/ --k 50

  # Run inference
  python cli.py inference --model-path checkpoints/best_model.pt \\
                          --test-data data/test.json --output-dir results/

  # Evaluate model
  python cli.py evaluate --model-path checkpoints/best_model.pt \\
                         --test-data data/test.json --top-k 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ========================================================================
    # preprocess_dataset command
    # ========================================================================
    preprocess_parser = subparsers.add_parser(
        'preprocess_dataset',
        help='Prepare datasets with PPR (Personalized PageRank) features'
    )
    preprocess_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file (JSON format)'
    )
    preprocess_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output directory for preprocessed data'
    )
    
    # ========================================================================
    # train command - Runs BOTH pretraining and main training
    # ========================================================================
    train_parser = subparsers.add_parser(
        'train',
        help='Run complete training pipeline (pretraining → main training)'
    )
    train_parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data file'
    )
    train_parser.add_argument(
        '--val-data',
        type=str,
        required=True,
        help='Path to validation data file'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory to save model checkpoints'
    )
    train_parser.add_argument(
        '--k',
        type=int,
        required=True,
        choices=[30, 50, 100, 150],
        help='K value for main training phase (number of triplets to sample per question)'
    )
    train_parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Number of epochs for main training phase (default: 50)'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for training (default: 1e-4)'
    )
    train_parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay for optimizer (default: 1e-5)'
    )
    train_parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=8,
        help='Number of gradient accumulation steps (default: 8)'
    )
    train_parser.add_argument(
        '--validation-interval',
        type=int,
        default=1,
        help='Validate every N epochs (default: 1)'
    )
    train_parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience in epochs (default: 10)'
    )
    train_parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps for learning rate scheduler (default: 100)'
    )
    
    # ========================================================================
    # inference command
    # ========================================================================
    inference_parser = subparsers.add_parser(
        'inference',
        help='Run inference to select top-k triplets from test data'
    )
    inference_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    inference_parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data file'
    )
    inference_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save inference results'
    )
    inference_parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top triplets to select (default: 100)'
    )
    
    # ========================================================================
    # evaluate command
    # ========================================================================
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Compute evaluation metrics (answer coverage, path coverage)'
    )
    evaluate_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    evaluate_parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data file'
    )
    evaluate_parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top triplets to evaluate (default: 100)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Dispatch to appropriate command handler
    if args.command == 'preprocess_dataset':
        run_preprocess_command(args)
    elif args.command == 'train':
        run_train_command(args)
    elif args.command == 'inference':
        run_inference_command(args)
    elif args.command == 'evaluate':
        run_evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
