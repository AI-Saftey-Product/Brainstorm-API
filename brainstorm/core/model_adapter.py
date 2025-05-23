"""Model adapter for interfacing with language models."""
from typing import Dict, Any, List, Optional, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
from datetime import datetime

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

    def setup_notification(self, websocket_manager, test_run_id: str) -> None:
        """Setup WebSocket notification for this adapter."""
        self.websocket_manager = websocket_manager
        self.test_run_id = test_run_id
        logger.info(f"WebSocket notification setup for test_run_id: {test_run_id}")
        
    async def send_model_input_notification(self, prompt: str, prompt_type: str = "default") -> None:
        """Send model input notification via WebSocket."""
        if not self.websocket_manager or not self.test_run_id:
            logger.debug(f"Cannot send model input notification - websocket_manager: {self.websocket_manager is not None}, test_run_id: {self.test_run_id}")
            return
            
        try:
            logger.info(f"Sending model input notification for test_run_id: {self.test_run_id}")
            await self.websocket_manager.send_notification(self.test_run_id, {
                "type": "model_input",
                "model_id": self.model_id,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"Model input notification sent successfully")
        except Exception as e:
            logger.error(f"Error sending model input notification: {e}")
            
    async def send_model_output_notification(self, output: str) -> None:
        """Send model output notification via WebSocket."""
        if not self.websocket_manager or not self.test_run_id:
            logger.debug(f"Cannot send model output notification - websocket_manager: {self.websocket_manager is not None}, test_run_id: {self.test_run_id}")
            return
            
        try:
            logger.info(f"Sending model output notification for test_run_id: {self.test_run_id}")
            await self.websocket_manager.send_notification(self.test_run_id, {
                "type": "model_output",
                "model_id": self.model_id,
                "output": output,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"Model output notification sent successfully")
        except Exception as e:
            logger.error(f"Error sending model output notification: {e}")
            
    async def send_evaluation_notification(self, evaluation: Dict[str, Any]) -> None:
        """Send evaluation notification via WebSocket."""
        if not self.websocket_manager or not self.test_run_id:
            logger.debug(f"Cannot send evaluation notification - websocket_manager: {self.websocket_manager is not None}, test_run_id: {self.test_run_id}")
            return
            
        try:
            logger.info(f"Sending evaluation notification for test_run_id: {self.test_run_id}")
            await self.websocket_manager.send_notification(self.test_run_id, {
                "type": "evaluation_result",
                "model_id": self.model_id,
                "evaluation": evaluation,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"Evaluation notification sent successfully")
        except Exception as e:
            logger.error(f"Error sending evaluation notification: {e}") 