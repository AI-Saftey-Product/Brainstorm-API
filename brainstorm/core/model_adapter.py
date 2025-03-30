"""Model adapter for interfacing with language models."""
from typing import Dict, Any, List, Optional, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logger = logging.getLogger(__name__)

class ModelAdapter:
    """Adapter class for language model interactions."""
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        """Initialize the model adapter.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device if device == "cuda" else -1
            )
            self.max_length = max_length
            self.temperature = temperature
            self.top_p = top_p
            logger.info(f"Successfully initialized model adapter with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model adapter: {e}")
            raise

    def generate(self, 
                prompt: str,
                max_length: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None) -> str:
        """Generate text response for a given prompt.
        
        Args:
            prompt: Input text to generate from
            max_length: Override default max length if provided
            temperature: Override default temperature if provided
            top_p: Override default top_p if provided
            
        Returns:
            Generated text response
        """
        try:
            response = self.generator(
                prompt,
                max_length=max_length or self.max_length,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def generate_batch(self,
                      prompts: List[str],
                      max_length: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Override default max length if provided
            temperature: Override default temperature if provided
            top_p: Override default top_p if provided
            
        Returns:
            List of generated responses
        """
        try:
            responses = self.generator(
                prompts,
                max_length=max_length or self.max_length,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                num_return_sequences=1
            )
            
            # Remove prompts from responses
            cleaned_responses = []
            for prompt, response in zip(prompts, responses):
                text = response[0]["generated_text"][len(prompt):].strip()
                cleaned_responses.append(text)
                
            return cleaned_responses
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [""] * len(prompts)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model.config._name_or_path,
            "model_type": self.model.config.model_type,
            "vocab_size": self.model.config.vocab_size,
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", None),
            "device": next(self.model.parameters()).device.type
        } 