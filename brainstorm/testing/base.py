"""Base classes for testing framework."""
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseTest(ABC):
    """Abstract base class for all tests."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test with configuration."""
        self.config = config or {}
        self.test_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run the test."""
        pass
        
    async def run_test(self, model_adapter=None, model_settings=None) -> Dict[str, Any]:
        """Compatibility method that delegates to run.
        
        This exists to maintain compatibility with the test service that expects a run_test method.
        """
        if model_adapter and hasattr(self, 'model'):
            self.model = model_adapter
            
        if model_settings:
            kwargs = model_settings
        else:
            kwargs = {}
            
        return await self.run(**kwargs)

class BasePerformanceTest(BaseTest):
    """Base class for performance tests."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.model = None
        
        # Initialize model if provided in config
        if config and "model" in config:
            self.model = config["model"]
        else:
            # Configure default model - try different import paths
            try:
                # Try importing from brainstorm.core.models
                from brainstorm.core.models.openai_model import OpenAIModel
                self.model = OpenAIModel(config or {})
            except ImportError:
                try:
                    # Try importing from brainstorm.models
                    from brainstorm.models.openai_model import OpenAIModel
                    self.model = OpenAIModel(config or {})
                except ImportError:
                    logger.warning("OpenAIModel not available, no model initialized")
    
    async def run(self, model_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the test with the given model parameters."""
        try:
            # Update model parameters if provided
            if model_parameters and self.model:
                self.model.update_parameters(model_parameters)
            
            # Run test implementation
            result = await self._run_test_implementation(model_parameters or {})
            
            # Add metadata
            result["metadata"] = {
                "test_id": self.test_id,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "model": self.model.__class__.__name__ if self.model else "None",
                "model_parameters": self.model.get_parameters() if self.model else {}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            raise
    
    @abstractmethod
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the test implementation. Must be implemented by subclasses."""
        pass 