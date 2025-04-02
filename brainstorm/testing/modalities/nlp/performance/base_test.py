"""Base class for NLP performance tests."""
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from brainstorm.core.models.base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseNLPPerformanceTest:
    """Base class for NLP performance tests."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test with configuration."""
        self.config = config or {}
        self.test_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.model = None
        
        # Initialize model if provided in config
        if "model" in config:
            self.model = config["model"]
        else:
            # Default to OpenAI model
            from brainstorm.core.models.openai_model import OpenAIModel
            self.model = OpenAIModel(config)
    
    async def run(self, model_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the test with the given model parameters."""
        try:
            # Update model parameters if provided
            if model_parameters:
                self.model.update_parameters(model_parameters)
            
            # Run test implementation
            result = await self._run_test_implementation(model_parameters or {})
            
            # Add metadata
            result["metadata"] = {
                "test_id": self.test_id,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "model": self.model.__class__.__name__,
                "model_parameters": self.model.get_parameters()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            raise
    
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the test implementation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_test_implementation") 