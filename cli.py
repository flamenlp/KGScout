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
    from src.services.preprocess_service import PreprocessService
    
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
    
    # Create service and run preprocessing
    service = PreprocessService()
    results = service.preprocess(args.input, args.output)
    
    # Print summary
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Preprocessed data saved to: {results['output_path']}")
    print(f"Total samples: {results['total_samples']}")
    if results['skipped_samples'] > 0:
        print(f"Skipped samples: {results['skipped_samples']}")
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
    from src.services.train_service import TrainService
    
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
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Pretraining checkpoints: {results['pretrain_checkpoint']}")
    print(f"Main training checkpoints: {results['main_checkpoint']}")
    print(f"Training logs: {results['log_dir']}")
    print("=" * 60)


def run_inference_command(args):
    """
    Execute inference to select top-k triplets from test data.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.inference_service import InferenceService
    
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
    
    # Create service and run inference
    service = InferenceService()
    results = service.run_inference(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        top_k=args.top_k
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total samples processed: {results['total_samples']}")
    print(f"Results saved to: {results['output_file']}")
    print(f"Average reward: {results['average_reward']:.4f}")
    print("=" * 60)


def run_evaluate_command(args):
    """
    Execute evaluation to compute metrics on test data.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.evaluate_service import EvaluateService
    
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
    
    # Create service and run evaluation
    service = EvaluateService()
    metrics = service.evaluate(
        model_path=args.model_path,
        test_data_path=args.test_data,
        top_k=args.top_k
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Answer Coverage:  {metrics['answer_coverage']:.4f}")
    print(f"Path Coverage:    {metrics['path_coverage']:.4f}")
    print(f"Average Reward:   {metrics['average_reward']:.4f}")
    print("=" * 60)


def run_llm_comparison_command(args):
    """
    Execute LLM comparison analysis.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.llm_comparison_service import LLMComparisonService
    
    # Validate dataset parameter
    if args.dataset not in ['webqsp', 'cwq']:
        print(f"Error: Invalid dataset '{args.dataset}'. Must be 'webqsp' or 'cwq'.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model-path is provided when using kgscout retriever
    if args.retriever_type == 'kgscout' and not args.model_path:
        print("Error: --model-path is required when using kgscout retriever.", file=sys.stderr)
        print("Suggestion: Provide the path to your trained KGscout model checkpoint.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model checkpoint exists if provided
    if args.model_path:
        validate_file_exists(args.model_path, "Model checkpoint")
    
    # Validate output directory is writable
    validate_directory_writable(args.output_dir, "Output directory")
    
    # Display header
    print("=" * 60)
    print("LLM COMPARISON ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Retriever: {args.retriever_type}")
    print(f"Top-k: {args.k}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Execute service
    try:
        service = LLMComparisonService()
        results = service.run_comparison(
            dataset=args.dataset,
            llm_model=args.llm_model,
            retriever_type=args.retriever_type,
            k=args.k,
            model_path=args.model_path,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Unexpected error during analysis: {str(e)}", file=sys.stderr)
        print(f"Suggestion: Check logs for more details and verify all inputs are correct.", file=sys.stderr)
        sys.exit(1)
    
    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total Questions: {results['total_questions']}")
    print(f"\nMetrics:")
    print(f"  Hit Score:       {results['hit']:.4f}")
    print(f"  Hit@1 Score:     {results['hit_at_1']:.4f}")
    print(f"  Macro F1:        {results['macro_f1']:.4f}")
    print(f"  Macro Precision: {results['macro_precision']:.4f}")
    print(f"  Macro Recall:    {results['macro_recall']:.4f}")
    print(f"  Exact Match:     {results['exact_match']:.4f}")
    print(f"\nOutput Files:")
    print(f"  Predictions: {results['predictions_file']}")
    print(f"  Results:     {results['results_file']}")
    print("=" * 60)


def run_k_ablation_command(args):
    """
    Execute k-value ablation study with Llama-3.1-8b.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.k_ablation_service import KAblationService
    
    # Validate dataset parameter
    if args.dataset not in ['webqsp', 'cwq']:
        print(f"Error: Invalid dataset '{args.dataset}'. Must be 'webqsp' or 'cwq'.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model-path is provided when using kgscout retriever
    if args.retriever_type == 'kgscout' and not args.model_path:
        print("Error: --model-path is required when using kgscout retriever.", file=sys.stderr)
        print("Suggestion: Provide the path to your trained KGscout model checkpoint.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model checkpoint exists if provided
    if args.model_path:
        validate_file_exists(args.model_path, "Model checkpoint")
    
    # Validate output directory is writable
    validate_directory_writable(args.output_dir, "Output directory")
    
    # Parse k-values if provided
    k_values = None
    if args.k_values:
        try:
            k_values = [int(k.strip()) for k in args.k_values.split(',')]
            # Validate all k-values are positive
            if any(k <= 0 for k in k_values):
                print("Error: All k-values must be positive integers.", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid k-values format '{args.k_values}'. Expected comma-separated integers (e.g., '30,50,100').", file=sys.stderr)
            sys.exit(1)
    
    # Validate k if provided
    if args.k is not None and args.k <= 0:
        print(f"Error: k must be positive, got {args.k}", file=sys.stderr)
        sys.exit(1)
    
    # Execute service
    try:
        service = KAblationService()
        results = service.run_ablation(
            dataset=args.dataset,
            retriever_type=args.retriever_type,
            k_values=k_values,
            k=args.k,
            model_path=args.model_path,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Unexpected error during ablation study: {str(e)}", file=sys.stderr)
        print(f"Suggestion: Check logs for more details and verify all inputs are correct.", file=sys.stderr)
        sys.exit(1)


def run_coverage_analysis_command(args):
    """
    Execute coverage analysis to measure answer and path coverage.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.coverage_analysis_service import CoverageAnalysisService
    
    # Validate dataset parameter
    if args.dataset not in ['webqsp', 'cwq']:
        print(f"Error: Invalid dataset '{args.dataset}'. Must be 'webqsp' or 'cwq'.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model checkpoint exists
    validate_file_exists(args.model_path, "Model checkpoint")
    
    # Validate output directory is writable
    validate_directory_writable(args.output_dir, "Output directory")
    
    # Parse k-values if provided, otherwise use defaults
    k_values = [30, 50, 100, 150]  # Default values
    if args.k_values:
        try:
            k_values = [int(k.strip()) for k in args.k_values.split(',')]
            # Validate all k-values are positive
            if any(k <= 0 for k in k_values):
                print("Error: All k-values must be positive integers.", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid k-values format '{args.k_values}'. Expected comma-separated integers (e.g., '30,50,100').", file=sys.stderr)
            sys.exit(1)
    
    # Display header
    print("=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"K-values: {k_values}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Execute service
    try:
        service = CoverageAnalysisService()
        results = service.run_coverage_analysis(
            dataset=args.dataset,
            model_path=args.model_path,
            k_values=k_values,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Unexpected error during coverage analysis: {str(e)}", file=sys.stderr)
        print(f"Suggestion: Check logs for more details and verify all inputs are correct.", file=sys.stderr)
        sys.exit(1)
    
    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results['output_file']}")
    print("=" * 60)


def run_statistical_analysis_command(args):
    """
    Execute statistical comparison analysis with case categorization.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.services.statistical_analysis_service import StatisticalAnalysisService
    
    # Validate dataset parameter
    if args.dataset not in ['webqsp', 'cwq']:
        print(f"Error: Invalid dataset '{args.dataset}'. Must be 'webqsp' or 'cwq'.", file=sys.stderr)
        sys.exit(1)
    
    # Validate model checkpoint exists
    validate_file_exists(args.model_path, "Model checkpoint")
    
    # Validate output directory is writable
    validate_directory_writable(args.output_dir, "Output directory")
    
    # Validate k is positive
    if args.k <= 0:
        print(f"Error: k must be positive, got {args.k}", file=sys.stderr)
        sys.exit(1)
    
    # Display header
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"Top-k: {args.k}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Execute service
    try:
        service = StatisticalAnalysisService()
        results = service.run_statistical_analysis(
            dataset=args.dataset,
            model_path=args.model_path,
            k=args.k,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Unexpected error during statistical analysis: {str(e)}", file=sys.stderr)
        print(f"Suggestion: Check logs for more details and verify all inputs are correct.", file=sys.stderr)
        sys.exit(1)
    
    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results['output_file']}")
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

  # LLM comparison with KGscout retriever
  python cli.py llm-comparison --dataset webqsp --llm-model llama \\
                               --retriever-type kgscout --k 100 \\
                               --model-path checkpoints/best_model.pt \\
                               --output-dir results/

  # LLM comparison with cosine retriever
  python cli.py llm-comparison --dataset cwq --llm-model qwen \\
                               --retriever-type cosine --k 50 \\
                               --output-dir results/

  # K-value ablation with KGscout retriever
  python cli.py k-ablation --dataset webqsp --retriever-type kgscout \\
                           --k-values 30,50,100,150 \\
                           --model-path checkpoints/best_model.pt \\
                           --output-dir results/

  # K-value ablation with single k value
  python cli.py k-ablation --dataset cwq --retriever-type cosine \\
                           --k 100 --output-dir results/

  # Coverage analysis with default k-values [30, 50, 100, 150]
  python cli.py coverage-analysis --dataset webqsp \\
                                  --model-path checkpoints/best_model.pt \\
                                  --output-dir results/

  # Coverage analysis with custom k-values
  python cli.py coverage-analysis --dataset cwq \\
                                  --model-path checkpoints/best_model.pt \\
                                  --k-values 50,100,200 \\
                                  --output-dir results/

  # Statistical analysis comparing retrievers
  python cli.py statistical-analysis --dataset webqsp \\
                                     --model-path checkpoints/best_model.pt \\
                                     --k 100 \\
                                     --output-dir results/
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
    
    # ========================================================================
    # llm-comparison command
    # ========================================================================
    llm_comparison_parser = subparsers.add_parser(
        'llm-comparison',
        help='Compare LLM models using the same retriever configuration'
    )
    llm_comparison_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help='Dataset to use for evaluation (webqsp or cwq)'
    )
    llm_comparison_parser.add_argument(
        '--llm-model',
        type=str,
        required=True,
        choices=['llama', 'qwen', 'deepseek'],
        help='LLM model to use for answer generation (llama, qwen, or deepseek)'
    )
    llm_comparison_parser.add_argument(
        '--retriever-type',
        type=str,
        required=True,
        choices=['kgscout', 'cosine'],
        help='Retriever type to use for triplet selection (kgscout or cosine)'
    )
    llm_comparison_parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of top triplets to select'
    )
    llm_comparison_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained KGscout model checkpoint (required if retriever-type is kgscout)'
    )
    llm_comparison_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )
    
    # ========================================================================
    # k-ablation command
    # ========================================================================
    k_ablation_parser = subparsers.add_parser(
        'k-ablation',
        help='Run k-value ablation study with Llama-3.1-8b to evaluate different k values'
    )
    k_ablation_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help='Dataset to use for evaluation (webqsp or cwq)'
    )
    k_ablation_parser.add_argument(
        '--retriever-type',
        type=str,
        required=True,
        choices=['kgscout', 'cosine'],
        help='Retriever type to use for triplet selection (kgscout or cosine)'
    )
    k_ablation_parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated list of k values to test (e.g., "30,50,100,150"). Defaults to [30, 50, 100, 150] if not provided'
    )
    k_ablation_parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Single k value to test (overrides --k-values if provided)'
    )
    k_ablation_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained KGscout model checkpoint (required if retriever-type is kgscout)'
    )
    k_ablation_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )
    
    # ========================================================================
    # coverage-analysis command
    # ========================================================================
    coverage_analysis_parser = subparsers.add_parser(
        'coverage-analysis',
        help='Analyze answer and path coverage for different k values comparing KGscout vs Cosine retrievers'
    )
    coverage_analysis_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help='Dataset to use for evaluation (webqsp or cwq)'
    )
    coverage_analysis_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained KGscout model checkpoint'
    )
    coverage_analysis_parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated list of k values to test (e.g., "30,50,100,150"). Defaults to [30, 50, 100, 150] if not provided'
    )
    coverage_analysis_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )
    
    # ========================================================================
    # statistical-analysis command
    # ========================================================================
    statistical_analysis_parser = subparsers.add_parser(
        'statistical-analysis',
        help='Perform statistical comparison between cosine and KGscout retrievers with case categorization'
    )
    statistical_analysis_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help='Dataset to use for evaluation (webqsp or cwq)'
    )
    statistical_analysis_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained KGscout model checkpoint'
    )
    statistical_analysis_parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of top triplets to select'
    )
    statistical_analysis_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
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
    elif args.command == 'llm-comparison':
        run_llm_comparison_command(args)
    elif args.command == 'k-ablation':
        run_k_ablation_command(args)
    elif args.command == 'coverage-analysis':
        run_coverage_analysis_command(args)
    elif args.command == 'statistical-analysis':
        run_statistical_analysis_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
