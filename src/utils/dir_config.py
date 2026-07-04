"""
Centralized directory and path configuration loader.

Loads dir_mapping.yml from the project root and provides helper functions
to access dataset paths, result directories, and experiment settings.

Usage:
    from src.utils.dir_config import load_config, get_dataset_paths, get_results_dir

    config = load_config()
    train, val, test = get_dataset_paths("cwq")
    model_ablation_dir = get_results_dir("ablation2", "model_ablation")
"""

import os
import yaml
from typing import Dict, Any, Tuple, List, Optional


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _find_project_root() -> str:
    """Find the project root by looking for dir_mapping.yml starting from this file."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):  # walk up at most 10 levels
        candidate = os.path.join(current, "dir_mapping.yml")
        if os.path.exists(candidate):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError(
        "Could not find dir_mapping.yml in any parent directory. "
        "Ensure it exists at the project root."
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load dir_mapping.yml configuration.

    Args:
        config_path: Optional explicit path to dir_mapping.yml.
                     If None, auto-discovers from project root.

    Returns:
        Parsed YAML configuration dict.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        root = _find_project_root()
        config_path = os.path.join(root, "dir_mapping.yml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config_path is None or config_path == os.path.join(_find_project_root(), "dir_mapping.yml"):
        _CONFIG_CACHE = config

    return config


def get_dataset_paths(dataset: str, config: Optional[Dict] = None) -> Tuple[str, str, str]:
    """
    Get train, val, test paths for a dataset.

    Args:
        dataset: Dataset name ('cwq' or 'webqsp')
        config: Optional pre-loaded config dict

    Returns:
        Tuple of (train_path, val_path, test_path)

    Raises:
        ValueError: If dataset name is invalid
    """
    if config is None:
        config = load_config()

    datasets = config.get("datasets", {})
    if dataset not in datasets:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(datasets.keys())}"
        )

    ds = datasets[dataset]
    return ds["train"], ds["val"], ds["test"]


def get_results_dir(experiment_type: str, sub_type: Optional[str] = None, config: Optional[Dict] = None) -> str:
    """
    Get results directory path.

    Args:
        experiment_type: One of 'ablation2', 'k_ablation', 'generalization', 'base'
        sub_type: Sub-type within experiment (e.g., 'model_ablation', 'reward_ablation')
        config: Optional pre-loaded config dict

    Returns:
        Directory path string

    Examples:
        get_results_dir("ablation2", "model_ablation")  -> "./results/ablation-2/model-ablation"
        get_results_dir("k_ablation")                   -> "./results/k-ablation"
    """
    if config is None:
        config = load_config()

    results = config.get("results", {})

    if experiment_type == "base":
        return results["base"]
    elif sub_type:
        return results[experiment_type][sub_type]
    else:
        return results[experiment_type]


def get_k_values(config: Optional[Dict] = None) -> List[int]:
    """Get the list of k values for k-ablation experiments."""
    if config is None:
        config = load_config()
    return config["experiments"]["k_ablation"]["k_values"]


def get_model_variants(config: Optional[Dict] = None) -> List[str]:
    """Get model ablation variant names."""
    if config is None:
        config = load_config()
    return config["experiments"]["model_variants"]


def get_reward_variants(config: Optional[Dict] = None) -> List[str]:
    """Get reward ablation variant names."""
    if config is None:
        config = load_config()
    return config["experiments"]["reward_variants"]


def get_defaults(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get default hyperparameters."""
    if config is None:
        config = load_config()
    return config["defaults"]


def get_llm_model_id(model_name: str, config: Optional[Dict] = None) -> str:
    """
    Get HuggingFace model ID for an LLM shorthand name.

    Args:
        model_name: One of 'llama', 'qwen', 'deepseek'

    Returns:
        HuggingFace model identifier string
    """
    if config is None:
        config = load_config()
    models = config.get("llm_models", {})
    if model_name not in models:
        raise ValueError(f"Unknown LLM model '{model_name}'. Available: {list(models.keys())}")
    return models[model_name]


def get_log_path(log_name: str, config: Optional[Dict] = None) -> str:
    """
    Get log file path by name.

    Args:
        log_name: One of 'k_ablation', 'ablation2', 'inference', 'vllm_inference', 'triplet_analysis'
    """
    if config is None:
        config = load_config()
    logs = config.get("logs", {})
    if log_name not in logs:
        raise ValueError(f"Unknown log '{log_name}'. Available: {list(logs.keys())}")
    return logs[log_name]
