"""OpenAI adapter for making API calls to OpenAI models."""
import logging
from typing import Any, Dict, List, Optional, Iterator

from openai import AsyncOpenAI

from brainstorm.core.adapters.base_adapter import BaseModelAdapter
from brainstorm.db.models.model import ModelDefinition

logger = logging.getLogger(__name__)


class OpenAINLPAdapter(BaseModelAdapter):
    """Adapter for OpenAI NLP models."""
    
    def __init__(self, model_config: ModelDefinition):
        """
        Initialize the OpenAI adapter.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()
        self.model_config = model_config
        # todo: fix inconsistent naming
        self.model_id = model_config.provider_model.lower()
        self.api_key = model_config.api_key
        # todo: will be done via relation to Users
        self.organization_id = "TBD"
        self.client = None
        self.is_chat_model = self._is_chat_model(self.model_id)
        self._initialize_client()

    def _is_chat_model(self, model_id: str) -> bool:
        """Determine if the model is a chat model based on its ID."""
        chat_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", 
            "gpt-35-turbo", "gpt4", "gpt-4o", "claude", 'gpt-4o-mini'
        ]
        return any(model in model_id.lower() for model in chat_models)
        
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                # organization=self.organization_id
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Format messages for OpenAI chat API
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                # Map 'assistant' to correct OpenAI role
                if role == "assistant":
                    role = "assistant"
                elif role == "system":
                    role = "system"
                else:
                    role = "user"
                    
                formatted_messages.append({
                    "role": role,
                    "content": msg["content"]
                })
            
            # Extract parameters
            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 1.0)
            frequency_penalty = kwargs.get("frequency_penalty", 0.0)
            presence_penalty = kwargs.get("presence_penalty", 0.0)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            return f"Error: {str(e)}"

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Use the proper embedding model
            embedding_model = kwargs.get("embedding_model", "text-embedding-ada-002")
            
            # Make API call
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=texts
            )
            
            # Return the embeddings
            return [data.embedding for data in response.data]
            
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
        if "frequency_penalty" in parameters:
            validated["frequency_penalty"] = max(min(parameters["frequency_penalty"], 2.0), -2.0)
        if "presence_penalty" in parameters:
            validated["presence_penalty"] = max(min(parameters["presence_penalty"], 2.0), -2.0)
        return validated

    async def get_supported_tests(self) -> List[str]:
        """Get list of supported test IDs."""
        return [
            "nlp_bias_test",
            "nlp_toxicity_test",
            "nlp_hallucination_test",
            "nlp_security_test",
            "nlp_adversarial_robustness_test",
            "bigcodebench_test",
            "humaneval_test",
            "mbpp_test",
            "apps_test",
            "usaco_test"
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
        """
        Validate the connection to the OpenAI API.
        """
        if not self.client:
            return False
            
        try:
            # Make a simple models.list API call to test connection
            models = await self.client.models.list()
            logger.info(f"OpenAI connection validated, found {len(models.data)} models")
            return True
                
        except Exception as e:
            logger.error(f"Error validating OpenAI connection: {str(e)}")
            return False
    
    async def generate(self, prompt: str, **parameters) -> str:
        """
        Generate text using either chat completion or text completion API.
        """
        try:
            if self.is_chat_model:
                return await self._generate_chat(prompt, parameters)
            else:
                return await self._generate_completion(prompt, parameters)
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _generate_chat(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate text using the OpenAI chat completion API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            frequency_penalty = parameters.get("frequency_penalty", 0.0)
            presence_penalty = parameters.get("presence_penalty", 0.0)
            
            # Create system message if provided
            messages = []
            if "system" in parameters:
                messages.append({"role": "system", "content": parameters["system"]})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise
    
    async def _generate_completion(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate text using the OpenAI completion API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            frequency_penalty = parameters.get("frequency_penalty", 0.0)
            presence_penalty = parameters.get("presence_penalty", 0.0)
            
            # Make API call
            response = await self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

            return response.choices[0].text
            
        except Exception as e:
            logger.error(f"Error in completion generation: {str(e)}")
            raise
            
    async def stream_generate(self, prompt: str, **parameters) -> Iterator[str]:
        """Generate text with streaming output."""
        try:
            if self.is_chat_model:
                async for chunk in self._stream_chat(prompt, parameters):
                    yield chunk
            else:
                async for chunk in self._stream_completion(prompt, parameters):
                    yield chunk
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            yield f"Error: {str(e)}"
            
    async def _stream_chat(self, prompt: str, parameters: Dict[str, Any]) -> Iterator[str]:
        """Stream text using the OpenAI chat completion API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            frequency_penalty = parameters.get("frequency_penalty", 0.0)
            presence_penalty = parameters.get("presence_penalty", 0.0)
            
            # Create system message if provided
            messages = []
            if "system" in parameters:
                messages.append({"role": "system", "content": parameters["system"]})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            
            # Process the streaming response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming chat generation: {str(e)}")
            yield f"Error: {str(e)}"
            
    async def _stream_completion(self, prompt: str, parameters: Dict[str, Any]) -> Iterator[str]:
        """Stream text using the OpenAI completion API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Extract parameters
            max_tokens = parameters.get("max_tokens", 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 1.0)
            frequency_penalty = parameters.get("frequency_penalty", 0.0)
            presence_penalty = parameters.get("presence_penalty", 0.0)
            
            # Make streaming API call
            stream = await self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            
            # Process the streaming response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    yield chunk.choices[0].text
                    
        except Exception as e:
            logger.error(f"Error in streaming completion generation: {str(e)}")
            yield f"Error: {str(e)}" 