"""Model adapters for different AI modalities."""

from app.model_adapters.nlp_adapter import get_nlp_adapter
from app.model_adapters.base_adapter import BaseModelAdapter

# Factory function to get the appropriate adapter based on modality
def get_model_adapter(model_config: dict) -> BaseModelAdapter:
    """
    Get the appropriate model adapter based on modality.
    
    Args:
        model_config: Configuration for the model
        
    Returns:
        An instance of the appropriate model adapter
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating model adapter with config: {model_config}")
    
    # Check for source/provider first
    source = model_config.get("source", "").lower()
    if source == "huggingface":
        from app.model_adapters.nlp_adapter import HuggingFaceNLPAdapter
        logger.info("Creating HuggingFaceNLPAdapter based on source")
        return HuggingFaceNLPAdapter(model_config)
    
    # Otherwise use modality-based routing
    modality = model_config.get("modality", "").upper()
    logger.info(f"Using modality-based routing with modality: {modality}")
    
    if modality == "NLP":
        return get_nlp_adapter(model_config)
    else:
        # Default to HuggingFace for unknown modalities
        logger.warning(f"Unknown modality: {modality}, defaulting to HuggingFaceNLPAdapter")
        from app.model_adapters.nlp_adapter import HuggingFaceNLPAdapter
        return HuggingFaceNLPAdapter(model_config) 