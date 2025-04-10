"""OpenAI model implementation."""
from typing import Dict, Any, Optional
import logging
import os
from openai import AsyncOpenAI

from brainstorm.core.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OpenAI model."""
        super().__init__(config)
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Set default parameters
        self.parameters.update({
            "model": config.get("model", "gpt-4"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 2048),
            "top_p": config.get("top_p", 1.0),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0)
        })
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's API."""
        try:
            # Update parameters with any overrides
            params = self.parameters.copy()
            params.update(kwargs)
            
            # Generate completion
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            # Extract and return the generated text
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate the model configuration."""
        required_fields = ["api_key"]
        return all(field in self.config for field in required_fields) 