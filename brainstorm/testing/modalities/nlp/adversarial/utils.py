"""
Utility functions for adversarial testing with graceful fallbacks.
"""
import logging
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)

# Dependency import check registry
DEPENDENCIES = {}

def check_import(module_name: str, import_function: Callable, fallback_value: Any = None) -> Any:
    """
    Safely import a module or function with fallback.
    
    Args:
        module_name: Name of the module for logging
        import_function: Function that attempts to import the module
        fallback_value: Value to return if import fails
        
    Returns:
        The imported module/function or the fallback value
    """
    if module_name in DEPENDENCIES:
        return DEPENDENCIES[module_name]
        
    try:
        result = import_function()
        DEPENDENCIES[module_name] = result
        logger.info(f"Successfully loaded dependency: {module_name}")
        return result
    except Exception as e:
        logger.warning(f"{module_name} not available: {str(e)}")
        DEPENDENCIES[module_name] = fallback_value
        return fallback_value

# Conditional imports with fallbacks
def get_sentence_transformer():
    """Get SentenceTransformer with fallback."""
    def import_fn():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    return check_import("SentenceTransformer", import_fn)

def get_detoxify():
    """Get Detoxify with fallback."""
    def import_fn():
        from detoxify import Detoxify
        return Detoxify
    return check_import("Detoxify", import_fn)

def get_use_model():
    """Get Universal Sentence Encoder with fallback."""
    def import_fn():
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return model
    return check_import("USE", import_fn)

def get_bert_score():
    """Get BERTScore with fallback."""
    def import_fn():
        import bert_score
        return bert_score
    return check_import("BERTScore", import_fn)

def calculate_similarity(text1: str, text2: str, method: str = "use") -> float:
    """
    Calculate semantic similarity between two texts with fallbacks.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method ('use', 'sbert', or 'bert_score')
        
    Returns:
        Similarity score (0-1) or -1 if calculation fails
    """
    if method == "use":
        use_model = get_use_model()
        if use_model:
            try:
                embeddings = use_model([text1, text2])
                import tensorflow as tf
                similarity = tf.reduce_sum(
                    tf.multiply(embeddings[0], embeddings[1])
                ) / (
                    tf.norm(embeddings[0]) * tf.norm(embeddings[1])
                )
                return float(similarity.numpy())
            except Exception as e:
                logger.warning(f"Error calculating USE similarity: {e}")
                
    if method == "sbert" or (method == "use" and "USE" not in DEPENDENCIES):
        SentenceTransformer = get_sentence_transformer()
        if SentenceTransformer:
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode([text1, text2])
                import numpy as np
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception as e:
                logger.warning(f"Error calculating SBERT similarity: {e}")
                
    if method == "bert_score" or (method in ["use", "sbert"] and len(DEPENDENCIES) < 2):
        bert_score = get_bert_score()
        if bert_score:
            try:
                P, R, F1 = bert_score.score([text1], [text2], lang="en", verbose=False)
                return float(F1[0])
            except Exception as e:
                logger.warning(f"Error calculating BERTScore: {e}")
    
    # Simple fallback using string similarity
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    except Exception:
        logger.warning("All similarity methods failed, returning -1")
        return -1

def check_toxicity(text: str) -> Dict[str, float]:
    """
    Check toxicity of text with fallbacks.
    
    Args:
        text: Text to check
        
    Returns:
        Dictionary of toxicity scores or empty dict if check fails
    """
    Detoxify = get_detoxify()
    if Detoxify:
        try:
            results = Detoxify('original').predict(text)
            return {k: float(v) for k, v in results.items()}
        except Exception as e:
            logger.warning(f"Error calculating toxicity: {e}")
    
    # Simple fallback checking for obvious toxic terms
    toxic_terms = ["hate", "stupid", "idiot", "kill", "die"]
    return {
        "toxicity": sum(term in text.lower() for term in toxic_terms) / len(toxic_terms),
        "severe_toxicity": 0.0,
        "obscene": 0.0,
        "identity_attack": 0.0,
        "insult": 0.0,
        "threat": 0.0,
        "sexual_explicit": 0.0
    } 