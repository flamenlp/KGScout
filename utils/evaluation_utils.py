"""
Common evaluation utility functions for dataset loading, model loading, and result saving.

This module provides shared functionality used across all evaluation services:
- Dataset loading with validation
- Model checkpoint loading with error handling
- Result saving with timestamps
- Dataset field validation
"""

import os
import json
import torch
from datetime import datetime
from typing import List, Dict, Any
from model.path_ranker import PathRankingModel


def load_dataset(dataset_name: str) -> List[Dict]:
    """
    Load preprocessed dataset by name.
    
    Args:
        dataset_name: Dataset identifier ('webqsp' or 'cwq')
    
    Returns:
        List of dataset samples as dictionaries
    
    Raises:
        FileNotFoundError: If dataset file does not exist
        ValueError: If dataset name is invalid or file is corrupted
    
    Requirements:
        - 5.1: Load webqsp dataset from dataset/processed-dataset/webqsp-tst.pt
        - 5.2: Load cwq dataset from dataset/processed-dataset/cwq-tst.pt
        - 5.3: Display error message with expected file path when file does not exist
        - 5.4: Display error message when dataset file is corrupted or invalid
        - 5.5: Validate that loaded dataset contains required fields
    """
    # Map dataset names to file paths
    dataset_paths = {
        'webqsp': 'dataset/processed-dataset/webqsp-tst.pt',
        'cwq': 'dataset/processed-dataset/cwq-tst.pt'
    }
    
    # Validate dataset name
    if dataset_name not in dataset_paths:
        raise ValueError(
            f"Invalid dataset name: '{dataset_name}'. "
            f"Expected one of: {list(dataset_paths.keys())}"
        )
    
    dataset_path = dataset_paths[dataset_name]
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Please ensure the dataset has been preprocessed and saved to the expected location."
        )
    
    # Load dataset with error handling
    try:
        data = torch.load(dataset_path)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset from {dataset_path}. "
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {str(e)}"
        )
    
    # Validate dataset is a list
    if not isinstance(data, list):
        raise ValueError(
            f"Invalid dataset format: expected list, got {type(data)}. "
            f"The dataset file may be corrupted."
        )
    
    # Validate dataset is not empty
    if len(data) == 0:
        raise ValueError(
            f"Dataset is empty: {dataset_path}. "
            f"Please check the preprocessing step."
        )
    
    # Validate required fields in first sample
    if not validate_dataset_fields(data):
        raise ValueError(
            f"Dataset validation failed: missing required fields in {dataset_path}. "
            f"Please check the preprocessing step."
        )
    
    return data


def load_model_checkpoint(model_path: str, device: str) -> PathRankingModel:
    """
    Load trained model checkpoint with comprehensive error handling.
    
    Args:
        model_path: Path to model checkpoint directory
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        Loaded PathRankingModel instance
    
    Raises:
        FileNotFoundError: If checkpoint file is missing
        ValueError: If model architecture mismatch during loading
    
    Requirements:
        - 6.1: Validate the file exists before attempting to load
        - 6.2: Verify the checkpoint contains model_state_dict key
        - 6.3: Display error message with architecture details when checkpoint architecture mismatches
        - 6.4: Display confirmation with model configuration details when checkpoint loads successfully
        - 6.5: Load the model to the appropriate device (CUDA if available, otherwise CPU)
    """
    # Validate model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint directory not found: {model_path}\n"
            f"Please ensure the model was trained and saved correctly."
        )
    
    # Check if it's a directory or file
    if os.path.isdir(model_path):
        checkpoint_file = os.path.join(model_path, "path_ranker.pt")
    else:
        checkpoint_file = model_path
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Model checkpoint file not found: {checkpoint_file}\n"
            f"Please ensure the model was saved correctly."
        )
    
    # Handle device availability
    if device == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Using CPU instead.")
        device = "cpu"
    
    # Load model using PathRankingModel.from_pretrained
    try:
        model = PathRankingModel.from_pretrained(model_path, device=device)
        
        # Display confirmation
        print(f"Model loaded successfully from: {model_path}")
        print(f"Model configuration:")
        print(f"  - Hidden size: {model.hidden_size}")
        print(f"  - Device: {device}")
        print(f"  - Temperature: {model.temperature.item():.4f}")
        print(f"  - Baseline: {model.baseline.item():.4f}")
        
        return model
        
    except FileNotFoundError as e:
        # Re-raise with original message
        raise e
    except ValueError as e:
        # Re-raise with original message (architecture mismatch)
        raise e
    except Exception as e:
        raise ValueError(
            f"Failed to load model checkpoint from {model_path}.\n"
            f"Error: {str(e)}"
        )


def save_results_with_timestamp(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str
) -> str:
    """
    Save results to JSON file with timestamp in filename.
    
    Args:
        results: Dictionary containing results to save
        output_dir: Directory to save results
        prefix: Filename prefix (e.g., 'llm_comparison', 'coverage_analysis')
    
    Returns:
        Path to saved results file
    
    Raises:
        PermissionError: If output directory is not writable
        IOError: If file write operation fails
    
    Requirements:
        - 8.1: Save results to the specified output directory
        - 8.2: Include timestamps in all output filenames to prevent overwriting
        - 8.3: Save detailed per-question results in JSON format
        - 8.5: Create output directory automatically if it does not exist
        - 8.6: Display error message and exit when output directory is not writable
        - 8.7: Validate that all output files were written successfully
    """
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Cannot create output directory (permission denied): {output_dir}\n"
            f"Please check directory permissions."
        )
    except Exception as e:
        raise IOError(
            f"Failed to create output directory: {output_dir}\n"
            f"Error: {str(e)}"
        )
    
    # Check if directory is writable
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(
            f"Output directory is not writable: {output_dir}\n"
            f"Please check directory permissions."
        )
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results to JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(
            f"Failed to write results to file: {filepath}\n"
            f"Error: {str(e)}"
        )
    
    # Validate file was written successfully
    if not os.path.exists(filepath):
        raise IOError(
            f"Results file was not created successfully: {filepath}\n"
            f"Please check disk space and permissions."
        )
    
    # Validate file is not empty
    if os.path.getsize(filepath) == 0:
        raise IOError(
            f"Results file is empty: {filepath}\n"
            f"File write may have failed."
        )
    
    return filepath


def validate_dataset_fields(data: List[Dict]) -> bool:
    """
    Validate that dataset contains all required fields.
    
    Args:
        data: List of dataset samples
    
    Returns:
        True if all required fields are present, False otherwise
    
    Requirements:
        - 5.5: Validate that loaded dataset contains required fields before proceeding
    """
    if not data or len(data) == 0:
        return False
    
    # Required fields for evaluation
    required_fields = [
        "question",
        "q_entity",
        "a_entity",
        "answer",
        "question_embedding",
        "topk_linearized_triplets",
        "topk_linearized_triplet_embeddings",
        "topk_rel_data",
        "topK_rel_embeddings"
    ]
    
    # Check first sample for required fields
    sample = data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"Warning: Dataset is missing required fields: {missing_fields}")
        return False
    
    return True
