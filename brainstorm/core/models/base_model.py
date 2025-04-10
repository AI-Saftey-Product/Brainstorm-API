"""Base class for all model implementations."""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all model implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model with configuration."""
        self.config = config or {}
        self.parameters = self.config.get("parameters", {})
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate")
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update model parameters."""
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return self.parameters.copy()
    
    def validate_config(self) -> bool:
        """Validate model configuration. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement validate_config") 