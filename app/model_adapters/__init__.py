"""Model adapters for different AI modalities."""

from app.model_adapters.nlp_adapter import get_nlp_adapter
from app.model_adapters.base_adapter import BaseModelAdapter

# Factory function to get the appropriate adapter based on modality
def get_model_adapter(modality: str, model_config: dict) -> BaseModelAdapter:
    """
    Get the appropriate model adapter based on modality.
    
    Args:
        modality: The model modality (NLP, Vision, etc.)
        model_config: Configuration for the model
        
    Returns:
        An instance of the appropriate model adapter
    """
    if modality.upper() == "NLP":
        return get_nlp_adapter(model_config)
    else:
        raise ValueError(f"Unsupported modality: {modality}") 