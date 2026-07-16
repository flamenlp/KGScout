"""
Checkpoint utilities for validation and management.

This module provides utilities for validating checkpoint structure
and ensuring checkpoints contain all required keys.
"""

import os
from typing import Dict, List, Any
import torch


def validate_checkpoint_structure(checkpoint: Dict[str, Any]) -> bool:
    """
    Validate that checkpoint contains all required keys.
    
    Required keys according to Requirement 6.3:
    - epoch: Current epoch number
    - model_state_dict: Model state dictionary
    - optimizer_state_dict: Optimizer state dictionary
    - scheduler_state_dict: Scheduler state dictionary
    - train_loss: Training loss value
    - val_loss: Validation loss value
    - metrics: Dictionary of additional metrics
    
    Args:
        checkpoint: Checkpoint dictionary to validate
    
    Returns:
        True if checkpoint contains all required keys, False otherwise
    
    Raises:
        ValueError: If checkpoint is missing required keys (with details)
    
    Requirements:
        - 6.3: Checkpoint should include model state and training progress
    """
    required_keys = [
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "train_loss",
        "val_loss",
        "metrics"
    ]
    
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            f"Required keys are: {required_keys}"
        )
    
    return True


def load_checkpoint(checkpoint_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load checkpoint from file and optionally validate its structure.
    
    Args:
        checkpoint_path: Path to checkpoint file
        validate: Whether to validate checkpoint structure (default: True)
    
    Returns:
        Checkpoint dictionary
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist or is corrupted
        ValueError: If checkpoint is missing required keys (when validate=True)
        
    Error Handling:
        - Missing checkpoint files: Raises FileNotFoundError with descriptive message
        - Corrupted files: Raises FileNotFoundError with error details
        - Device mismatches: Automatically maps to CPU for compatibility
    """
    # Handle missing checkpoint file
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path}\n"
            f"Please ensure the checkpoint was saved correctly and the path is correct."
        )
    
    # Load checkpoint with automatic device mapping to CPU for compatibility
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {str(e)}"
        )
    
    if validate:
        validate_checkpoint_structure(checkpoint)
    
    return checkpoint


def get_checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract summary information from a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
    
    Returns:
        Dictionary with checkpoint summary information
    """
    # Validate first
    validate_checkpoint_structure(checkpoint)
    
    return {
        "epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "has_metrics": "metrics" in checkpoint and bool(checkpoint["metrics"])
    }
