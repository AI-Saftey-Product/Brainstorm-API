from __future__ import annotations

import abc
import json
import logging
import uuid
import asyncio  # Added for sleep functionality
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator

from brainstorm.core.adapters.gcp_maas_adapter import GCP_MAAS_NLPAdapter
from brainstorm.core.adapters.openai_adapter import OpenAINLPAdapter
from brainstorm.core.adapters.llama_adapter import LlamaNLPAdapter
import httpx
from pydantic import ValidationError

# Import from huggingface_hub with proper error handling
from huggingface_hub import InferenceClient

from brainstorm.db.models.model import ModelDefinition, ModelProvider

# Try to import specific types, but handle gracefully if missing
try:
    from huggingface_hub.inference._text_generation import TextGenerationResponse
except ImportError:
    # Define a placeholder type for backward compatibility
    class TextGenerationResponse:
        def __init__(self, generated_text=""):
            self.generated_text = generated_text

from brainstorm.api.v1.schemas.parameters.nlp_parameters import get_nlp_parameter_schema
from brainstorm.core.adapters.base_adapter import BaseModelAdapter
from brainstorm.core.config import settings


logger = logging.getLogger(__name__)


class NLPModelAdapter(BaseModelAdapter):
    """Base adapter for NLP models regardless of provider."""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.model_config = None
        self.model_type = None
        self.api_key = None
        self.model_id = None
        self.websocket_manager = None
        self.test_run_id = None
    
    @abstractmethod
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize the adapter with model configuration."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the model API."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **parameters) -> str:
        """Generate text from the model given a prompt."""
        pass
    
    async def invoke(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Invoke the NLP model based on the model type."""
        if not self.client:
            raise ValueError("Adapter not initialized. Call initialize() first.")
        
        # Validate parameters
        validated_params = await self.validate_parameters(parameters)
        
        try:
            # Handle different NLP model types
            if self.model_type in ["Text Generation", "Text2Text Generation"]:
                return await self._invoke_text_generation(input_data, validated_params)
            elif self.model_type in ["Question Answering", "Table Question Answering"]:
                return await self._invoke_question_answering(input_data, validated_params)
            elif self.model_type in ["Text Classification", "Zero-Shot Classification"]:
                return await self._invoke_classification(input_data, validated_params)
            else:
                # Default to text completion
                return await self._invoke_text_generation(input_data, validated_params)
        except Exception as e:
            logger.error(f"Error invoking NLP model: {str(e)}")
            raise
    
    async def _invoke_text_generation(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke text generation model."""
        # Send model input notification via WebSocket
        try:
            if hasattr(self, 'websocket_manager') and self.websocket_manager:
                from datetime import datetime
                
                # Use test_run_id preferentially over test_id for the WebSocket channel
                channel_id = None
                if hasattr(self, 'test_run_id') and self.test_run_id:
                    channel_id = self.test_run_id
                elif hasattr(self, 'test_id') and self.test_id:
                    channel_id = self.test_id
                    
                if channel_id:
                    logger.info(f"Sending model input notification via adapter to channel: {channel_id}")
                    await self.websocket_manager.send_notification(channel_id, {
                        "type": "model_input",
                        "model_id": self.model_id,
                        "test_id": "text_generation",
                        "prompt": prompt,
                        "prompt_type": "default",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.info(f"Model input notification sent via adapter")
        except Exception as e:
            logger.error(f"Error sending model input notification from adapter: {str(e)}")
        
        # Generate response
        response = await self.generate(prompt, **parameters)
        
        # Send model output notification via WebSocket
        try:
            if hasattr(self, 'websocket_manager') and self.websocket_manager:
                from datetime import datetime
                
                # Use test_run_id preferentially over test_id for the WebSocket channel
                channel_id = None
                if hasattr(self, 'test_run_id') and self.test_run_id:
                    channel_id = self.test_run_id
                elif hasattr(self, 'test_id') and self.test_id:
                    channel_id = self.test_id
                    
                if channel_id:
                    logger.info(f"Sending model output notification via adapter to channel: {channel_id}")
                    await self.websocket_manager.send_notification(channel_id, {
                        "type": "model_output",
                        "model_id": self.model_id,
                        "test_id": "text_generation",
                        "output": response,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.info(f"Model output notification sent via adapter")
        except Exception as e:
            logger.error(f"Error sending model output notification from adapter: {str(e)}")
        
        return {
            "text": response,
            "raw_response": {"generated_text": response}
        }
    
    async def _invoke_question_answering(self, input_data: Dict[str, str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke question answering model."""
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        return await self._invoke_text_generation(prompt, parameters)
    
    async def _invoke_classification(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke classification model."""
        text = input_data.get("text", "")
        labels = input_data.get("labels", [])
        
        if not labels:
            raise ValueError("Labels must be provided for classification")
        
        prompt = f"Classify the following text into one of these categories: {', '.join(labels)}.\n\nText: {text}\n\nCategory:"
        
        result = await self._invoke_text_generation(prompt, parameters)
        
        return {
            "label": result["text"].strip(),
            "raw_response": result["raw_response"]
        }
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for the model type."""
        # Get parameter schema for this model type
        schema_class = get_nlp_parameter_schema(self.model_type)
        
        # Validate parameters against schema
        try:
            # Add model_id if it's not there
            if "model_id" not in parameters and self.model_config:
                parameters["model_id"] = self.model_config.get("id", "unknown")
                
            validated = schema_class(**parameters)
            return validated.dict(exclude={"model_id"})
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            raise ValueError(f"Invalid parameters for {self.model_type}: {str(e)}")
    
    async def get_supported_tests(self) -> List[str]:
        """Get list of tests supported by this model type."""
        # Map NLP model types to supported test IDs
        test_map = {
            "Text Generation": [
                "nlp_bias_test", "nlp_toxicity_test", 
                "nlp_hallucination_test", "nlp_security_test",
                "nlp_adversarial_robustness_test"
            ],
            "Text2Text Generation": [
                "nlp_bias_test", "nlp_toxicity_test", 
                "nlp_hallucination_test", "nlp_security_test",
                "nlp_adversarial_robustness_test"
            ],
            "Question Answering": [
                "nlp_bias_test", "nlp_factual_accuracy_test", 
                "nlp_hallucination_test", "nlp_adversarial_robustness_test"
            ],
            "Text Classification": [
                "nlp_bias_test", "nlp_classification_fairness_test",
                "nlp_adversarial_robustness_test"
            ],
            "Zero-Shot Classification": [
                "nlp_bias_test", "nlp_classification_fairness_test",
                "nlp_zero_shot_robustness_test", "nlp_adversarial_robustness_test"
            ],
            "Summarization": [
                "nlp_bias_test", "nlp_factual_accuracy_test", 
                "nlp_hallucination_test", "nlp_adversarial_robustness_test"
            ]
        }
        
        return test_map.get(self.model_type, ["nlp_basic_test"])


class GenericNLPAdapter(NLPModelAdapter):
    """Generic adapter for any NLP model API."""
    
    def __init__(self):
        super().__init__()
    
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize the adapter with model configuration."""
        # Early check - refuse to initialize if this is explicitly an OpenAI model
        source = model_config.get("source", "").lower()
        model_id = model_config.get("model_id", "")
        
        if source == "openai" or source == "azure_openai":
            logger.error(f"Attempted to initialize Generic adapter for an OpenAI source model: {model_id}")
            raise ValueError(f"OpenAI models must use the OpenAI adapter, not Generic adapter. Model: {model_id}")
        
        # Continue with normal initialization
        self.model_config = model_config
        self.model_type = model_config.get("sub_type")
        self.api_key = model_config.get("api_key")
        self.model_id = model_config.get("model_id", "default-model")
        self.endpoint_url = model_config.get("endpoint_url")
        
        if not self.endpoint_url:
            raise ValueError("Endpoint URL not provided for generic model")
        
        # Initialize HTTP client with appropriate headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.client = httpx.AsyncClient(headers=headers)
    
    async def validate_connection(self) -> bool:
        """Validate the connection to the API."""
        try:
            # Simple GET request to check if the endpoint is accessible
            response = await self.client.get(self.endpoint_url)
            return 200 <= response.status_code < 500  # Any non-server error code is considered valid
        except Exception as e:
            logger.error(f"Error validating connection to {self.endpoint_url}: {str(e)}")
            return False
            
    async def generate(self, prompt: str, **parameters) -> str:
        """Generate text from the model given a prompt."""
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        # Allow customization of request format
        if "request_format" in self.model_config:
            payload = self._format_request(prompt, parameters)
        
        response = await self.client.post(self.endpoint_url, json=payload)
        response.raise_for_status()
        
        try:
            result = response.json()
            return self._extract_generated_text(result)
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            # Fallback to raw response text
            return response.text
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response."""
        try:
            # Format messages for chat
            formatted_messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
            prompt = "\n".join(formatted_messages)
            return await self.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            return f"Error: {str(e)}"

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            # For now, return dummy embeddings
            return [[0.1, 0.2, 0.3] for _ in texts]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [[0.0, 0.0, 0.0] for _ in texts]
    
    def _format_request(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Format request based on model_config request_format."""
        request_format = self.model_config.get("request_format", {})
        payload = {}
        
        # Map model parameters to API-specific format
        for api_key, config_key in request_format.items():
            if config_key == "prompt":
                payload[api_key] = prompt
            elif config_key in parameters:
                payload[api_key] = parameters[config_key]
        
        return payload
    
    def _extract_generated_text(self, response: Any) -> str:
        """Extract generated text from API response."""
        # Default extraction paths for common APIs
        extraction_path = self.model_config.get("response_path", None)
        
        if not extraction_path:
            # Try common response formats
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                # Common patterns
                for key in ["generated_text", "text", "content", "output", "result"]:
                    if key in response:
                        return response[key]
                # Nested patterns
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "text" in choice:
                            return choice["text"]
                        elif "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"]
            elif isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], str):
                    return response[0]
                elif isinstance(response[0], dict):
                    for key in ["generated_text", "text", "content"]:
                        if key in response[0]:
                            return response[0][key]
            
            # If we can't extract, return a string representation
            return str(response)
        else:
            # Use the provided extraction path
            result = response
            for key in extraction_path.split('.'):
                if isinstance(result, dict) and key in result:
                    result = result[key]
                elif isinstance(result, list) and key.isdigit():
                    result = result[int(key)]
                else:
                    return str(response)  # Fallback if path doesn't exist
            return str(result)


class HuggingFaceNLPAdapter(BaseModelAdapter):
    """Adapter for HuggingFace NLP models."""
    
    def __init__(self, config: ModelDefinition):
        """
        Initialize the HuggingFace adapter.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()
        
        # todo: rename
        model_id = config.provider_model
        
        # Proceed with normal initialization
        self.model_config = config
        self.model_id = model_id
        self.api_key = config.api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.client = None
        self.supports_chat = "chat" in self.model_id.lower()
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the HuggingFace client."""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                model=self.model_id,
                token=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {str(e)}")
            self.client = None

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response."""
        try:
            # Format messages for chat
            formatted_messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
            prompt = "\n".join(formatted_messages)
            return await self.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            return f"Error: {str(e)}"

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            # For now, return dummy embeddings
            return [[0.1, 0.2, 0.3] for _ in texts]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [[0.0, 0.0, 0.0] for _ in texts]

    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return sanitized parameters."""
        # Basic parameter validation
        validated = {}
        if "max_tokens" in parameters:
            validated["max_tokens"] = min(max(parameters["max_tokens"], 1), 1000)
        if "temperature" in parameters:
            validated["temperature"] = max(min(parameters["temperature"], 1.0), 0.0)
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
            if isinstance(input_data, str):
                return await self.generate(input_data, **parameters)
            elif isinstance(input_data, list) and all(isinstance(msg, dict) for msg in input_data):
                return await self.chat(input_data, **parameters)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            return f"Error: {str(e)}"
    
    async def validate_connection(self) -> bool:
        """
        Validate the connection to the HuggingFace API.
        """
        try:
            # Use the InferenceClient to make a simple test call
            test_text = "Hello"
            response = self.client.text_generation(test_text, max_new_tokens=1)
            
            # If we got a response, consider it valid
            logger.info(f"HuggingFace model {self.model_id} connection validated")
            return True
                
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            return False
    
    async def generate(self, prompt: str, **parameters) -> str:
        """
        Generate text using direct API calls with minimal error handling.
        """
        try:
            return await self._generate_text(prompt, parameters)
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _generate_text(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate text using the HuggingFace model."""
        try:
            # Ensure prompt is a string
            if isinstance(prompt, dict):
                prompt = str(prompt)
            elif not isinstance(prompt, str):
                prompt = str(prompt)
                logger.warning(f"Converting prompt from {type(prompt)} to string")

            # Make the API call - text_generation is synchronous
            response = self.client.text_generation(
                prompt,
                max_new_tokens=parameters.get("max_tokens", 100),
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.95),
                do_sample=parameters.get("do_sample", True)
            )
            
            # Handle different response types
            if isinstance(response, TextGenerationResponse):
                return response.generated_text
            elif isinstance(response, str):
                return response
            elif isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"]
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return str(response)
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise
    
    async def _generate_chat(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate text using chat API with minimal error handling."""
        try:
            # Format messages
            messages = [{"role": "user", "content": prompt}]
            
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 0.9)
            
            # Use direct API calls
            model_url = f"{self.base_url}/{self.model_id}"
            
            # Create payload
            payload = {
                "inputs": {
                    "messages": messages
                },
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }
            
            # Make the request - don't check status
            response = await self.client.post(model_url, json=payload, timeout=30.0)
            
            # Parse the response
            result = response.json()
            
            # Extract generated text from result
            if isinstance(result, dict):
                if "generated_text" in result:
                    return result["generated_text"]
                elif "content" in result:
                    return result["content"]
                else:
                    return str(result)
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    # Try common fields
                    for field in ["generated_text", "content", "text", "message"]:
                        if field in result[0]:
                            return result[0][field]
                return str(result[0])
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            
            # Try simplified fallback
            try:
                # Try with simplified payload
                payload = {"inputs": prompt}
                response = await self.client.post(model_url, json=payload, timeout=30.0)
                result = response.json()
                return str(result)
            except Exception:
                pass
                
            return f"Error: {str(e)}"
    
    async def stream_generate(self, prompt: str, **parameters) -> Iterator[str]:
        """
        Stream text generation results with minimal error handling.
        """
        try:
            # Direct streaming using httpx client
            model_url = f"{self.base_url}/{self.model_id}"
            
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 100)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 0.9)
            
            # Create payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": True,
                    "stream": True
                }
            }
            
            # Use httpx to stream the response
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0
            ) as client:
                async with client.stream("POST", model_url, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield line.strip()
        
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield f"Error: {str(e)}"


# Factory function to get the right adapter based on model configuration
def get_nlp_adapter(model_config: ModelDefinition) -> GCP_MAAS_NLPAdapter | OpenAINLPAdapter | HuggingFaceNLPAdapter | LlamaNLPAdapter | GenericNLPAdapter:
    """Get the appropriate adapter for an NLP model."""
    # First check direct source property
    provider = model_config.provider
    model_id = model_config.model_id
    
    if provider == ModelProvider.OPENAI.value or provider == "azure_openai":
        logger.info(f"Using OpenAI adapter for source: {provider}, model: {model_id}")
        return OpenAINLPAdapter(model_config)
    elif provider == ModelProvider.HUGGINGFACE.value:
        logger.info(f"Using HuggingFace adapter for source: {provider}")
        return HuggingFaceNLPAdapter(model_config)
    elif provider == "llama":
        logger.info(f"Using Llama adapter for source: {provider}, model: {model_id}")
        return LlamaNLPAdapter(model_config)
    elif provider == ModelProvider.GCP_MAAS:
        logger.info(f"Using GCP_MAAS adapter for source: {provider}, model: {model_id}")
        return GCP_MAAS_NLPAdapter(model_config)
    else:
        logger.info(f"Using Generic adapter for source: {provider}, model: {model_id}")
        return GenericNLPAdapter()
