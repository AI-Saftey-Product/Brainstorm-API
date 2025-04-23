"""Base adapter for model interactions."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    """Base class for all model adapters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base adapter with configuration."""
        self.config = config or {}
        self.model = None
        self.model_config = None
        self.model_id = None
        self.api_key = None
        
    def set_model(self, model: Any) -> None:
        """Set the model to be adapted."""
        self.model = model
    
    def get_model(self) -> Any:
        """Get the current model."""
        return self.model
    
    def adapt_input(self, input_data: Any) -> Any:
        """Adapt input data for the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement adapt_input")
    
    def adapt_output(self, output_data: Any) -> Any:
        """Adapt output data from the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement adapt_output")
    
    def validate_config(self) -> bool:
        """Validate adapter configuration. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement validate_config")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
        
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response."""
        pass
        
    @abstractmethod
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
        
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the model API."""
        pass

    @abstractmethod
    async def invoke(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Invoke the model with the given input and parameters."""
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return sanitized parameters for the model."""
        pass
    
    @abstractmethod
    async def get_supported_tests(self) -> List[str]:
        """Get a list of test IDs that are supported by this model type."""
        pass 

# Create an alias for backward compatibility
ModelAdapter = BaseModelAdapter

__all__ = ['BaseModelAdapter', 'ModelAdapter']