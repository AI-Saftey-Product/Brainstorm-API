"""Registry for managing NLP models and their configurations."""
import logging
from typing import Dict, Any, Optional, Union, List
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from brainstorm.testing.modalities.nlp.adversarial.utils import get_use_model, get_bert_score, get_detoxify

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing NLP models and their configurations."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        """Get singleton instance of ModelRegistry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the model registry."""
        if ModelRegistry._instance is not None:
            raise Exception("ModelRegistry is a singleton! Use get_instance() instead.")
            
        self._models = {}
        self._tokenizers = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def register_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_config: Configuration parameters for the model
        """
        if model_name in self._models:
            logger.warning(f"Model {model_name} already exists in registry. Overwriting...")
            
        try:
            logger.info(f"Initializing model: {model_name}")
            
            if model_name == "use_encoder":
                self._models[model_name] = get_use_model()
            elif model_name == "bert_score":
                self._models[model_name] = get_bert_score()
            elif model_name == "toxicity":
                self._models[model_name] = get_detoxify()
            else:
                if "transformers" in model_config.get("backend", ""):
                    model = AutoModel.from_pretrained(model_config["model_id"])
                    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
                    self._models[model_name] = model.to(self._device)
                    self._tokenizers[model_name] = tokenizer
                elif "sentence_transformers" in model_config.get("backend", ""):
                    model = SentenceTransformer(model_config["model_id"])
                    self._models[model_name] = model
                else:
                    raise ValueError(f"Unknown model backend: {model_config.get('backend')}")
                    
            logger.info(f"Successfully registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {str(e)}")
            raise
            
    def get_model(self, model_name: str) -> Any:
        """Get a model from the registry."""
        if model_name not in self._models:
            raise KeyError(f"Model {model_name} not found in registry")
        return self._models[model_name]
        
    def get_tokenizer(self, model_name: str) -> Any:
        """Get a tokenizer from the registry."""
        if model_name not in self._tokenizers:
            raise KeyError(f"Tokenizer for model {model_name} not found in registry")
        return self._tokenizers[model_name]
        
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
        
    def clear(self) -> None:
        """Clear all models and tokenizers from the registry."""
        self._models.clear()
        self._tokenizers.clear()
        torch.cuda.empty_cache()
        
    @property
    def models(self) -> List[str]:
        """Get list of registered model names."""
        return list(self._models.keys())