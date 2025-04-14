"""
Configuration for the StrongReject test module.
"""
import os
from typing import Dict, Any, Optional

from brainstorm.config.api_keys import get_api_key

# Default judge model parameters
DEFAULT_JUDGE_MODEL = {
    "id": "gpt-4o",
    "provider": "openai",
    "temperature": 0.0,
    "max_tokens": 1024
}

# Jailbreak method options
JAILBREAK_METHODS = [
    "None",   # No jailbreak (direct prompting)
    "AIM",    # The AIM jailbreak prompt
]

class StrongRejectConfig:
    """Configuration for the StrongReject test."""
    
    def __init__(
        self,
        judge_model_id: str = DEFAULT_JUDGE_MODEL["id"],
        judge_model_provider: str = DEFAULT_JUDGE_MODEL["provider"],
        judge_temperature: float = DEFAULT_JUDGE_MODEL["temperature"],
        judge_max_tokens: int = DEFAULT_JUDGE_MODEL["max_tokens"],
        jailbreak_method: str = "AIM",
        max_examples: int = 100,
        api_key: Optional[str] = None
    ):
        """Initialize the StrongReject test configuration.
        
        Args:
            judge_model_id: The ID of the model to use as a judge
            judge_model_provider: The provider of the judge model (e.g., 'openai', 'anthropic')
            judge_temperature: The temperature setting for the judge model
            judge_max_tokens: The maximum number of tokens for the judge model
            jailbreak_method: The jailbreak method to use (one of JAILBREAK_METHODS)
            max_examples: The maximum number of examples to test
            api_key: The API key to use (overrides environment variables)
        """
        self.judge_model_id = judge_model_id
        self.judge_model_provider = judge_model_provider
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        
        # Validate jailbreak method
        if jailbreak_method not in JAILBREAK_METHODS:
            raise ValueError(f"Invalid jailbreak method: {jailbreak_method}. "
                           f"Must be one of: {', '.join(JAILBREAK_METHODS)}")
        self.jailbreak_method = jailbreak_method
        
        self.max_examples = max_examples
        
        # Get API key, prioritizing the provided key over environment variables
        self.api_key = api_key or get_api_key(judge_model_provider)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary for use with the test."""
        return {
            "judge_model_id": self.judge_model_id,
            "judge_model_provider": self.judge_model_provider,
            "jailbreak_method": self.jailbreak_method,
            "max_examples": self.max_examples,
            "judge_config": {
                "model_id": self.judge_model_id,
                "api_key": self.api_key,
                "max_tokens": self.judge_max_tokens,
                "temperature": self.judge_temperature
            }
        }
    
    @classmethod
    def from_env(cls) -> 'StrongRejectConfig':
        """Create a configuration from environment variables."""
        return cls(
            judge_model_id=os.getenv("STRONG_REJECT_JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL["id"]),
            judge_model_provider=os.getenv("STRONG_REJECT_JUDGE_PROVIDER", DEFAULT_JUDGE_MODEL["provider"]),
            judge_temperature=float(os.getenv("STRONG_REJECT_JUDGE_TEMPERATURE", DEFAULT_JUDGE_MODEL["temperature"])),
            judge_max_tokens=int(os.getenv("STRONG_REJECT_JUDGE_MAX_TOKENS", DEFAULT_JUDGE_MODEL["max_tokens"])),
            jailbreak_method=os.getenv("STRONG_REJECT_JAILBREAK_METHOD", "AIM"),
            max_examples=int(os.getenv("STRONG_REJECT_MAX_EXAMPLES", 100)),
            api_key=os.getenv("STRONG_REJECT_API_KEY") or get_api_key(os.getenv("STRONG_REJECT_JUDGE_PROVIDER", "openai"))
        ) 