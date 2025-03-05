import asyncio
import json
from typing import Dict, Any, Optional, List, Union
import logging
import httpx

from app.model_adapters.base_adapter import BaseModelAdapter
from app.schemas.parameters.nlp_parameters import get_nlp_parameter_schema
from app.core.config import settings


logger = logging.getLogger(__name__)


class OpenAINLPAdapter(BaseModelAdapter):
    """Adapter for OpenAI NLP models."""
    
    def __init__(self):
        self.client = None
        self.model_config = None
        self.model_type = None
        self.api_key = None
        self.base_url = "https://api.openai.com/v1"
    
    async def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize the adapter with model configuration."""
        self.model_config = model_config
        self.model_type = model_config.get("sub_type")
        self.api_key = model_config.get("api_key") or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("API key not provided for OpenAI model")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def validate_connection(self) -> bool:
        """Validate the connection to the OpenAI API."""
        try:
            # Simple models list call to validate API key
            response = await self.client.get("/models")
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Failed to validate OpenAI connection: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error validating OpenAI connection: {str(e)}")
            return False
    
    async def invoke(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Invoke the OpenAI model based on the model type."""
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
            logger.error(f"Error invoking OpenAI model: {str(e)}")
            raise
    
    async def _invoke_text_generation(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke text generation model."""
        payload = {
            "model": parameters.get("model", "gpt-3.5-turbo"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": parameters.get("temperature", 1.0),
            "max_tokens": parameters.get("max_tokens", 1024),
            "top_p": parameters.get("top_p", 1.0),
            "frequency_penalty": parameters.get("frequency_penalty", 0.0),
            "presence_penalty": parameters.get("presence_penalty", 0.0),
        }
        
        stop_sequences = parameters.get("stop_sequences")
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        
        return {
            "text": result["choices"][0]["message"]["content"],
            "raw_response": result
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
                "nlp_hallucination_test", "nlp_security_test"
            ],
            "Text2Text Generation": [
                "nlp_bias_test", "nlp_toxicity_test", 
                "nlp_hallucination_test", "nlp_security_test"
            ],
            "Question Answering": [
                "nlp_bias_test", "nlp_factual_accuracy_test", 
                "nlp_hallucination_test"
            ],
            "Text Classification": [
                "nlp_bias_test", "nlp_classification_fairness_test"
            ],
            "Zero-Shot Classification": [
                "nlp_bias_test", "nlp_classification_fairness_test",
                "nlp_zero_shot_robustness_test"
            ],
            "Summarization": [
                "nlp_bias_test", "nlp_factual_accuracy_test", 
                "nlp_hallucination_test"
            ]
        }
        
        return test_map.get(self.model_type, ["nlp_basic_test"])


# Factory function to get the right adapter based on model configuration
def get_nlp_adapter(model_config: Dict[str, Any]) -> BaseModelAdapter:
    """Get the appropriate adapter for an NLP model."""
    # For now, we only support OpenAI, but this could be expanded
    return OpenAINLPAdapter() 