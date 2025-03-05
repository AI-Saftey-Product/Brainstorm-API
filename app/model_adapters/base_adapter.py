from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseModelAdapter(ABC):
    """Base interface for all model adapters."""
    
    @abstractmethod
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize the model adapter with configuration."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the model."""
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