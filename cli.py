#!/usr/bin/env python3
"""
Command-line interface for KGScout training pipeline.

Provides the 'train' command used by the justfile to run the complete
training pipeline (pretraining → main REINFORCE training).

Other commands (inference, evaluation, coverage, statistical analysis)
are invoked directly via justfile recipes and standalone scripts.
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
    try:
        os.makedirs(dirpath, exist_ok=True)
    except PermissionError:
        print(f"Error: {description} cannot be created (permission denied): {dirpath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to create {description}: {dirpath}", file=sys.stderr)
        print(f"Reason: {str(e)}", file=sys.stderr)
        sys.exit(1)

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
    if args.early_stopping_patience >= args.num_epochs:
        print(
            f"Warning: Early stopping patience ({args.early_stopping_patience}) "
            f"is >= num_epochs ({args.num_epochs}). Early stopping will have no effect.",
            file=sys.stderr,
        )

    if args.validation_interval > args.num_epochs:
        print(
            f"Error: Validation interval ({args.validation_interval}) "
            f"cannot be greater than num_epochs ({args.num_epochs}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.learning_rate <= 0:
        print(f"Error: Learning rate must be positive, got {args.learning_rate}", file=sys.stderr)
        sys.exit(1)

    if args.learning_rate > 1.0:
        print(
            f"Warning: Learning rate ({args.learning_rate}) is unusually high. "
            f"Typical values are 1e-5 to 1e-3.",
            file=sys.stderr,
        )

    if args.weight_decay < 0:
        print(f"Error: Weight decay must be non-negative, got {args.weight_decay}", file=sys.stderr)
        sys.exit(1)

    if args.gradient_accumulation_steps <= 0:
        print(
            f"Error: Gradient accumulation steps must be positive, "
            f"got {args.gradient_accumulation_steps}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.warmup_steps < 0:
        print(f"Error: Warmup steps must be non-negative, got {args.warmup_steps}", file=sys.stderr)
        sys.exit(1)


def run_train_command(args):
    """
    Execute complete training pipeline:
    1. Run pretraining (5 epochs, n=500, fixed)
    2. Load best pretrained model checkpoint
    3. Run main training with specified k parameter

    Args:
        args: Parsed command-line arguments
    """
    from src.services.train_service import TrainService
    from src.model import get_model_class
    from src.training.rewards import get_reward_function

    # Validate input files exist
    validate_file_exists(args.train_data, "Training data file")
    validate_file_exists(args.val_data, "Validation data file")

    # Validate checkpoint directory is writable
    validate_directory_writable(args.checkpoint_dir, "Checkpoint directory")

    # Validate training arguments for conflicts
    validate_train_arguments(args)

    # Resolve model class and reward function
    model_class = get_model_class(args.model_class)
    reward_function = get_reward_function(args.reward_function)

    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Main training k parameter: {args.k}")
    print(f"Main training epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    if args.model_class:
        print(f"Model class: {args.model_class} ({model_class.__name__})")
    if args.reward_function:
        print(f"Reward function: {args.reward_function}")
    print("=" * 60)

    # Create service and run training
    service = TrainService()
    results = service.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        k=args.k,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        sample_k=args.sample_k,
        model_class=model_class,
        reward_function=reward_function,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Pretraining checkpoints: {results['pretrain_checkpoint']}")
    print(f"Main training checkpoints: {results['main_checkpoint']}")
    print(f"Training logs: {results['log_dir']}")
    print("=" * 60)


def main():
    """CLI entry point. Only the 'train' command is supported here."""
    parser = argparse.ArgumentParser(
        description="KGScout Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python cli.py train --train-data data/train.pt --val-data data/val.pt \\
                      --checkpoint-dir checkpoints/ --k 100

  # Train with model ablation variant
  python cli.py train --train-data data/train.pt --val-data data/val.pt \\
                      --checkpoint-dir checkpoints/ --k 100 --model-class no-gate

  # Train with reward ablation variant
  python cli.py train --train-data data/train.pt --val-data data/val.pt \\
                      --checkpoint-dir checkpoints/ --k 100 --reward-function only_presence
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # train command
    # ========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Run complete training pipeline (pretraining → main training)",
    )
    train_parser.add_argument(
        "--train-data", type=str, required=True, help="Path to training data file"
    )
    train_parser.add_argument(
        "--val-data", type=str, required=True, help="Path to validation data file"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    train_parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="K value for main training (number of top triplets to select per question)",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs for main training (default: 30)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer (default: 1e-5)",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps (default: 32)",
    )
    train_parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for LR scheduler (default: 100)",
    )
    train_parser.add_argument(
        "--sample-k",
        type=int,
        default=1000,
        help="Pool size N: number of prefiltered triplets per question (default: 1000)",
    )
    train_parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        choices=["no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"],
        help="Model architecture variant for ablation (default: PathRankingModel)",
    )
    train_parser.add_argument(
        "--reward-function",
        type=str,
        default=None,
        choices=["only_presence", "only_connection"],
        help="Reward function variant for ablation (default: compute_reward_v8)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "train":
        run_train_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
