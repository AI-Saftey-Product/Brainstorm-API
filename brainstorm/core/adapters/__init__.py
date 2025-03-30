"""Model adapter initialization."""
from typing import Dict, Any

from brainstorm.core.adapters.nlp_adapter import get_nlp_adapter
from brainstorm.core.adapters.base_adapter import ModelAdapter


def get_model_adapter(model_config: Dict[str, Any]):
    """Get the appropriate adapter for a model based on its configuration."""
    model_type = model_config.get("type", "").lower()
    
    if model_type == "nlp":
        return get_nlp_adapter(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

__all__ = ['get_model_adapter', 'ModelAdapter']