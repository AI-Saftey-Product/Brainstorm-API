"""Base adapter for model interactions."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseModelAdapter(ABC):
    """Base class for all model adapters."""
    
    def __init__(self):
        """Initialize the base adapter."""
        self.model_config = None
        self.model_id = None
        self.api_key = None
        
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
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize the model adapter with configuration."""
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