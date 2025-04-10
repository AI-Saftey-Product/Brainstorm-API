"""
Configuration for API keys used by various testing modules.
These keys can be loaded from environment variables or a .env file.
"""
import os
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to load from .env file if it exists
env_file = Path(".env")
if env_file.exists():
    try:
        # Try to load the .env file
        from dotenv import load_dotenv
        logger.info(f"Loading environment variables from {env_file.absolute()}")
        load_dotenv(env_file, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"Error reading .env file: encoding issue. Trying with different encodings...")
        # Try with different encodings
        for encoding in ["latin-1", "cp1252", "ascii"]:
            try:
                load_dotenv(env_file, encoding=encoding)
                logger.info(f"Successfully loaded .env file with {encoding} encoding")
                break
            except Exception as e:
                logger.warning(f"Failed to load with {encoding}: {e}")
    except Exception as e:
        logger.warning(f"Error loading .env file: {str(e)}")

# API key configuration
API_KEYS: Dict[str, Optional[str]] = {
    # OpenAI API credentials
    "openai": os.getenv("OPENAI_API_KEY"),
    
    # Anthropic API credentials
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    
    # Add additional API keys as needed
    "azure_openai": os.getenv("AZURE_OPENAI_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
}

def get_api_key(provider: str) -> Optional[str]:
    """Get the API key for a specific provider.
    
    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        
    Returns:
        The API key if available, otherwise None
    """
    # Normalize the provider name to lowercase
    provider_lower = provider.lower() if provider else ""
    
    # Try to get the key from the API_KEYS dictionary
    key = API_KEYS.get(provider_lower)
    
    # Check for empty string (treat as None)
    if key == "":
        key = None
        
    # Check if the key is a placeholder
    if key and (key == "your-api-key-here" or 
                key == "REPLACE_WITH_YOUR_ACTUAL_OPENAI_API_KEY" or
                key.startswith("REPLACE") or 
                key.startswith("YOUR_")):
        logger.warning(f"API key for {provider} appears to be a placeholder: {key}")
        key = None
    
    if not key:
        logger.warning(f"No API key found for provider: {provider}")
        
    return key 