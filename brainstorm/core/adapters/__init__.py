"""Model adapter initialization."""
from functools import lru_cache
from typing import Dict, Any

from brainstorm.core.adapters.nlp_adapter import get_nlp_adapter
from brainstorm.core.adapters.base_adapter import ModelAdapter
from brainstorm.db.models.model import ModelDefinition


@lru_cache
def get_model_adapter(model_definition: ModelDefinition) -> ModelAdapter:
    """
    Get the appropriate adapter for a model based on its configuration.
    Currently only NLP is implemented.
    """
    return get_nlp_adapter(model_definition)
