"""Base class for NLP performance tests."""
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from brainstorm.core.models.base_model import BaseModel
from brainstorm.testing.base import BaseTest

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
            # Try different import paths for OpenAI model
            try:
                # Try importing from brainstorm.core.models
                from brainstorm.core.models.openai_model import OpenAIModel
                self.model = OpenAIModel(config)
            except ImportError:
                try:
                    # Try importing from brainstorm.models
                    from brainstorm.models.openai_model import OpenAIModel
                    self.model = OpenAIModel(config)
                except ImportError:
                    logger.warning("OpenAIModel not available, no model initialized")
    
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
    
    async def run_test(self, model_adapter=None, model_settings=None) -> Dict[str, Any]:
        """Compatibility method for test service to call run.
        
        Args:
            model_adapter: Model adapter to use
            model_settings: Model settings to use
            
        Returns:
            Dict containing test results
        """
        if model_adapter:
            self.model = model_adapter
            
        return await self.run(model_settings or {})
    
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the test implementation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_test_implementation")


# Bridge class for compatibility with both import styles
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
            # Configure default model
            try:
                from brainstorm.core.models.openai_model import OpenAIModel
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
    
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the test implementation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_test_implementation")
        
    async def _generate_response(self, prompt: str, system_prompt: str = None, use_messages: bool = False):
        """Helper method to generate a response from the model."""
        if use_messages:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            response = await self.model.generate_chat_completion(messages)
            return response.choices[0].message.content
        else:
            if system_prompt:
                response = await self.model.generate(prompt=prompt, system=system_prompt)
            else:
                response = await self.model.generate(prompt=prompt)
            return response.completion
    
    def _get_model(self, model_name: str = None):
        """Helper method to get or create a model instance."""
        if model_name is None:
            return self.model
        try:
            # Try importing from brainstorm.core.models
            from brainstorm.core.models.openai_model import OpenAIModel
            return OpenAIModel({"model": model_name})
        except ImportError:
            try:
                # Try importing from brainstorm.models
                from brainstorm.models.openai_model import OpenAIModel
                return OpenAIModel({"model": model_name})
            except ImportError:
                logger.warning(f"Unable to create model {model_name}")
                return self.model 