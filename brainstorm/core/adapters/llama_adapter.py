"""Llama adapter for making API calls to Llama models."""
import logging
import asyncio
from typing import Any, Dict, List, Optional, Iterator
import json

from brainstorm.db.models.model import ModelDefinition

try:
    from llamaapi import LlamaAPI
    LLAMA_API_AVAILABLE = True
except ImportError:
    LLAMA_API_AVAILABLE = False
    LlamaAPI = None

from brainstorm.core.adapters.base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class LlamaNLPAdapter(BaseModelAdapter):
    """Adapter for Llama NLP models."""
    
    def __init__(self, config: ModelDefinition):
        """
        Initialize the Llama adapter.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()
        if not LLAMA_API_AVAILABLE:
            raise ImportError("llamaapi package is not installed. Please install it with: pip install llamaapi")
            
        self.model_config = config
        self.model_id = config.provider_model
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("API key is required for Llama API")
            
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Llama API client."""
        try:
            self.client = LlamaAPI(self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Llama client: {str(e)}")
            self.client = None
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response."""
        if not self.client:
            raise ValueError("Llama client not initialized")
            
        try:
            # Extract parameters
            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 1.0)
            
            # Build API request
            api_request = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            
            # Make API call
            response = self.client.run(api_request)
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            return f"Error: {str(e)}"

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.client:
            raise ValueError("Llama client not initialized")
            
        try:
            # Build API request
            api_request = {
                "model": self.model_id,
                "input": texts[0] if len(texts) == 1 else texts,
                "task": "embedding"
            }
            
            # Make API call
            response = self.client.run(api_request)
            result = response.json()
            
            # Return the embeddings
            if len(texts) == 1:
                return [result["data"][0]["embedding"]]
            return [item["embedding"] for item in result["data"]]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [[0.0, 0.0, 0.0] for _ in texts]

    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return sanitized parameters."""
        # Basic parameter validation
        validated = {}
        if "max_tokens" in parameters:
            validated["max_tokens"] = min(max(parameters["max_tokens"], 1), 4096)
        if "temperature" in parameters:
            validated["temperature"] = max(min(parameters["temperature"], 2.0), 0.0)
        if "top_p" in parameters:
            validated["top_p"] = max(min(parameters["top_p"], 1.0), 0.0)
        return validated

    async def get_supported_tests(self) -> List[str]:
        """Get list of supported test IDs."""
        return [
            "nlp_bias_test",
            "nlp_toxicity_test",
            "nlp_hallucination_test",
            "nlp_security_test",
            "nlp_adversarial_robustness_test"
        ]

    async def invoke(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Invoke the model with input data and parameters."""
        try:
            if isinstance(input_data, list) and all(isinstance(msg, dict) for msg in input_data):
                return await self.chat(input_data, **parameters)
            elif isinstance(input_data, str):
                return await self.generate(input_data, **parameters)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            return f"Error: {str(e)}"
    
    async def validate_connection(self) -> bool:
        """Validate the connection to the Llama API."""
        if not self.client:
            return False
            
        try:
            # Make a simple API call to test connection
            api_request = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }
            response = self.client.run(api_request)
            response.json()  # This will raise an error if the response is invalid
            logger.info("Llama connection validated")
            return True
                
        except Exception as e:
            logger.error(f"Error validating Llama connection: {str(e)}")
            return False
    
    async def generate(self, prompt: str, **parameters) -> str:
        """Generate text using the Llama API."""
        if not self.client:
            raise ValueError("Llama client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            
            # Build API request
            api_request = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            
            # Make API call
            response = self.client.run(api_request)
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return f"Error: {str(e)}"
    
    async def stream_generate(self, prompt: str, **parameters) -> Iterator[str]:
        """Stream text generation using the Llama API."""
        if not self.client:
            raise ValueError("Llama client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            
            # Build API request
            api_request = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True
            }
            
            # Make API call
            response = self.client.run(api_request)
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        data = json.loads(chunk)
                        if "choices" in data and data["choices"]:
                            yield data["choices"][0]["delta"].get("content", "")
                    except json.JSONDecodeError:
                        continue
                            
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield f"Error: {str(e)}" 