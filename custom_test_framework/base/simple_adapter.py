"""Simple model adapter implementation for CLI use."""
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Simple base adapter class to avoid dependency on the main app's BaseModelAdapter
class BaseModelAdapter(ABC):
    """Minimal base adapter interface."""
    
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


class SimpleModelAdapter(BaseModelAdapter):
    """
    Simple model adapter for CLI use.
    
    This adapter provides a minimal implementation that can be used 
    when the main app model adapters aren't available. It attempts to
    connect to models using standard APIs.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the simple adapter.
        
        Args:
            model_config: Configuration for the model
        """
        self.model_config = model_config
        self.model_id = model_config.get("model_id", "")
        self.api_key = model_config.get("api_key")
        self.modality = model_config.get("modality", "NLP")
        self.model_type = model_config.get("sub_type", "Text Generation")
        self.endpoint_url = model_config.get("endpoint_url")
        self.client = None
        
        # Look for specific provider in config
        self.provider = model_config.get("source", "").lower()
        if not self.provider:
            # Try to infer from model_id
            if "openai" in self.model_id.lower():
                self.provider = "openai"
            elif "claude" in self.model_id.lower():
                self.provider = "anthropic"
            elif "huggingface" in self.model_id.lower():
                self.provider = "huggingface"
            else:
                self.provider = "generic"
    
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """
        Initialize the adapter with model configuration.
        
        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        self.model_id = model_config.get("model_id", self.model_id)
        self.api_key = model_config.get("api_key", self.api_key)
        
        # Try to set up appropriate client based on provider
        try:
            if self.provider == "openai":
                import openai
                self.client = openai.Client(api_key=self.api_key)
                logger.info("Initialized OpenAI client")
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Initialized Anthropic client")
            elif self.provider == "huggingface":
                try:
                    from huggingface_hub import InferenceClient
                    self.client = InferenceClient(
                        model=self.model_id,
                        token=self.api_key
                    )
                    logger.info("Initialized HuggingFace client")
                except ImportError:
                    logger.warning("huggingface_hub not installed, using HTTP requests")
                    import httpx
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    self.client = httpx.AsyncClient(headers=headers)
            else:
                # Generic HTTP client
                import httpx
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                self.client = httpx.AsyncClient(headers=headers)
                logger.info("Initialized generic HTTP client")
        except ImportError as e:
            logger.warning(f"Could not initialize client for {self.provider}: {e}")
            logger.warning("Will use mock responses for testing")
    
    async def validate_connection(self) -> bool:
        """
        Validate the connection to the model API.
        
        Returns:
            True if connection is valid, False otherwise
        """
        # For simple test adapter, we'll just return True
        # In a real implementation, this would check the API connection
        logger.info("Validating connection (simplified)")
        
        if not self.client:
            logger.warning("No client initialized, using mock mode")
            return True
            
        try:
            # Simple validation based on provider
            if self.provider == "openai":
                # List models to verify API key works
                _ = await asyncio.to_thread(lambda: self.client.models.list())
                return True
            elif self.provider == "anthropic":
                # No simple validation endpoint, just assume it works if client initialized
                return True
            elif self.provider == "huggingface":
                if hasattr(self.client, "get_model_status"):
                    _ = await asyncio.to_thread(lambda: self.client.get_model_status(self.model_id))
                    return True
                else:
                    # Using HTTP client
                    url = f"https://api-inference.huggingface.co/models/{self.model_id}"
                    response = await self.client.get(url)
                    return 200 <= response.status_code < 500
            else:
                # For generic, if we have an endpoint URL, try to access it
                if self.endpoint_url:
                    response = await self.client.get(self.endpoint_url)
                    return 200 <= response.status_code < 500
                return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    async def generate(self, prompt: str, **parameters) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt
            parameters: Additional parameters
            
        Returns:
            Generated text
        """
        logger.info(f"Generating text with prompt: {prompt[:30]}...")
        
        if not self.client:
            # Mock response for testing
            return f"This is a mock response to: {prompt[:30]}..."
        
        try:
            # Generate based on provider
            if self.provider == "openai":
                response = await asyncio.to_thread(
                    lambda: self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        **parameters
                    )
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = await asyncio.to_thread(
                    lambda: self.client.messages.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        **parameters
                    )
                )
                return response.content[0].text
                
            elif self.provider == "huggingface":
                if hasattr(self.client, "text_generation"):
                    # Using HuggingFace InferenceClient
                    response = await asyncio.to_thread(
                        lambda: self.client.text_generation(
                            prompt,
                            **parameters
                        )
                    )
                    return response
                else:
                    # Using HTTP client
                    url = f"https://api-inference.huggingface.co/models/{self.model_id}"
                    payload = {"inputs": prompt, "parameters": parameters}
                    response = await self.client.post(url, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "")
                        return str(result)
                    else:
                        logger.error(f"Error from HuggingFace API: {response.text}")
                        return f"Error: {response.status_code}"
            else:
                # Generic API
                if self.endpoint_url:
                    payload = {"prompt": prompt}
                    payload.update(parameters)
                    
                    response = await self.client.post(self.endpoint_url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Try to extract text from common response formats
                        if isinstance(result, str):
                            return result
                        elif isinstance(result, dict):
                            for key in ["text", "generated_text", "content", "output", "result"]:
                                if key in result:
                                    return result[key]
                            # Just convert the whole response to string
                            return str(result)
                        else:
                            return str(result)
                    else:
                        logger.error(f"Error from API: {response.text}")
                        return f"Error: {response.status_code}"
                else:
                    # No endpoint, return mock
                    return f"Mock response (no endpoint): {prompt[:30]}..."
                    
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response.
        
        Args:
            messages: List of role/content message pairs
            kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Convert chat format to single prompt for simplicity
        prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
        return await self.generate(prompt, **kwargs)
    
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        # Simple adapter doesn't implement embeddings properly
        # Return mock embeddings for testing
        logger.warning("Using mock embeddings - simple adapter doesn't implement real embeddings")
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    async def invoke(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """
        Invoke the model with input data.
        
        Args:
            input_data: Input data (prompt or structured input)
            parameters: Model parameters
            
        Returns:
            Model output
        """
        # Simplify to handle text generation for basic cases
        if isinstance(input_data, str):
            result = await self.generate(input_data, **parameters)
            return {"text": result}
        elif isinstance(input_data, dict) and "prompt" in input_data:
            result = await self.generate(input_data["prompt"], **parameters)
            return {"text": result}
        elif isinstance(input_data, dict) and "messages" in input_data:
            result = await self.chat(input_data["messages"], **parameters)
            return {"text": result}
        else:
            # Unknown format, just try to generate with string representation
            result = await self.generate(str(input_data), **parameters)
            return {"text": result}
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated parameters
        """
        # Simple implementation just returns the parameters as-is
        # In a full implementation, this would validate against a schema
        return parameters
    
    async def get_supported_tests(self) -> List[str]:
        """
        Get list of supported test IDs.
        
        Returns:
            List of test IDs
        """
        # Simple mapping based on model type
        if self.model_type in ["Text Generation", "Text2Text Generation"]:
            return ["custom_content_moderation_test", "custom_bias_test", "custom_hallucination_test"]
        elif self.model_type in ["Question Answering"]:
            return ["custom_qa_test", "custom_factual_consistency_test"]
        else:
            # Default tests for any model type
            return ["custom_content_moderation_test"] 