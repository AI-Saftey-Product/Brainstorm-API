"""Singleton registry for managing ML models."""

import logging
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton registry for managing ML models.
    Ensures models are loaded only once and reused across the application.
    """
    
    _instance = None
    _lock = Lock()
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ModelRegistry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a model instance, initializing it if necessary.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The requested model instance
        """
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    self._models[model_name] = self._initialize_model(model_name)
        return self._models[model_name]
    
    def _initialize_model(self, model_name: str) -> Any:
        """
        Initialize a new model instance.
        
        Args:
            model_name: Name of the model to initialize
            
        Returns:
            Initialized model instance
        """
        try:
            logger.info(f"Initializing model: {model_name}")
            
            if model_name == "use_encoder":
                from app.tests.nlp.adversarial.utils import get_use_model
                return get_use_model()
            elif model_name == "bert_score":
                from app.tests.nlp.adversarial.utils import get_bert_score
                return get_bert_score()
            elif model_name == "toxicity":
                from app.tests.nlp.adversarial.utils import get_detoxify
                return get_detoxify()
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            raise
    
    def clear(self):
        """Clear all loaded models from memory."""
        with self._lock:
            self._models.clear()
            
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names."""
        return list(self._models.keys()) 