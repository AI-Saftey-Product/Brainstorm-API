"""Utility functions for handling API keys for different model providers."""
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Dictionary to store API keys
_api_keys = {}

def register_api_key(provider: str, api_key: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Register an API key for a model provider.
    
    Args:
        provider: Name of the model provider (e.g., 'openai', 'huggingface')
        api_key: The API key string
        additional_info: Optional additional information, like organization ID
    """
    _api_keys[provider.lower()] = {
        "api_key": api_key,
        "additional_info": additional_info or {}
    }
    logger.info(f"Registered API key for {provider}")

def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a model provider.
    
    Args:
        provider: Name of the model provider (e.g., 'openai', 'huggingface')
        
    Returns:
        The API key string if available, None otherwise
    """
    provider = provider.lower()
    
    # First check if the key is already registered
    if provider in _api_keys:
        return _api_keys[provider]["api_key"]
    
    # If not, check environment variables
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    # Check provider-specific environment variables
    if not api_key:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "huggingface":
            api_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "azure_openai":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    
    # If found, register it for future use
    if api_key:
        register_api_key(provider, api_key)
        return api_key
    
    return None

def get_additional_info(provider: str, key: str) -> Optional[Any]:
    """
    Get additional information for a provider.
    
    Args:
        provider: Name of the model provider
        key: The key of the additional info
        
    Returns:
        The value if available, None otherwise
    """
    provider = provider.lower()
    
    if provider in _api_keys and "additional_info" in _api_keys[provider]:
        return _api_keys[provider]["additional_info"].get(key)
    
    # Check environment variables for specific additional info
    if provider == "openai" and key == "organization_id":
        return os.environ.get("OPENAI_ORG_ID")
    elif provider == "azure_openai" and key == "deployment_name":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    
    return None

def load_api_keys_from_env() -> None:
    """Load API keys from environment variables."""
    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        register_api_key("openai", openai_key, {
            "organization_id": os.environ.get("OPENAI_ORG_ID")
        })
    
    # HuggingFace
    hf_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_API_KEY")
    if hf_key:
        register_api_key("huggingface", hf_key)
    
    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        register_api_key("anthropic", anthropic_key)
    
    # Azure OpenAI
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        register_api_key("azure_openai", azure_key, {
            "deployment_name": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT")
        })

# Load API keys on module import
load_api_keys_from_env() 