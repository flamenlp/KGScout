"""
Model registry for PathRankingModel and ablation variants.
"""

from typing import Optional, Type
import torch.nn as nn

from src.model.path_ranker import PathRankingModel


# Valid model class names (ablation variants)
VALID_MODEL_CLASSES = [
    "no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"
]


def get_model_class(name: Optional[str] = None) -> Type[nn.Module]:
    """
    Get model class by name.

    Args:
        name: Model variant name. If None, returns PathRankingModel (default).
              Valid names: "no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"

    Returns:
        Model class (not an instance)

    Raises:
        ValueError: If name is not a valid model class name
    """
    if name is None:
        return PathRankingModel

    import importlib
    model_ablation = importlib.import_module("ablation-2.model_ablation")

    MODEL_REGISTRY = {
        "no-ppr": model_ablation.ReversedNoPPR,
        "no-rt": model_ablation.ReversedNoRT,
        "no-tt": model_ablation.ReversedNoTT,
        "no-gate": model_ablation.ReversedNoGate,
        "no-ra": model_ablation.ReversedNoRA,
        "no-ta": model_ablation.ReversedNoTA,
    }

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: '{name}'. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[name]
