"""Base classes for adversarial attacks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from difflib import SequenceMatcher
from app.model_adapters.base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)


class AdversarialAttack(ABC):
    """Base class for all adversarial attacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize attack with configuration.
        
        Args:
            config: Configuration parameters for the attack
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.results = []
        self.success_count = 0
        self.total_count = 0
    
    @abstractmethod
    async def perturb(self, text: str) -> str:
        """
        Apply perturbation to the input text.
        
        Args:
            text: Original input text
            
        Returns:
            Perturbed text
        """
        pass
    
    async def run_attack(self, model_adapter: BaseModelAdapter, model_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run the attack against a model.
        
        Args:
            model_adapter: Adapter for the model to test
            model_parameters: Parameters for the model
            
        Returns:
            List of attack results
        """
        logger.info(f"Running {self.name} attack")
        
        # Use sample prompts for testing
        sample_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "I think artificial intelligence is the future of technology.",
            "Please summarize the main points of the article.",
            "What are the benefits and drawbacks of remote work?",
            "Explain the concept of climate change in simple terms."
        ]
        
        # Get number of examples from config
        num_examples = self.config.get("num_examples", 5)
        sample_prompts = sample_prompts[:num_examples]
        
        results = []
        self.total_count = 0
        self.success_count = 0
        
        for prompt in sample_prompts:
            try:
                # Get original output
                original_output = await model_adapter.generate(prompt, model_parameters)
                
                # Apply perturbation
                perturbed_prompt = await self.perturb(prompt)
                
                # Get output with perturbed input
                perturbed_output = await model_adapter.generate(perturbed_prompt, model_parameters)
                
                # Determine if attack was successful
                success = self._is_attack_successful(original_output, perturbed_output)
                
                # Record result
                result = {
                    "original_prompt": prompt,
                    "perturbed_prompt": perturbed_prompt,
                    "original_output": original_output,
                    "perturbed_output": perturbed_output,
                    "success": success
                }
                
                results.append(result)
                
                # Update counters
                self.total_count += 1
                if success:
                    self.success_count += 1
                    
            except Exception as e:
                logger.error(f"Error during attack execution: {str(e)}")
                results.append({
                    "original_prompt": prompt,
                    "error": str(e),
                    "success": False
                })
                self.total_count += 1
        
        self.results = results
        return results
    
    def _is_attack_successful(self, original_output: str, perturbed_output: str) -> bool:
        """
        Determine if an attack was successful.
        
        Args:
            original_output: Model output for original input
            perturbed_output: Model output for perturbed input
            
        Returns:
            True if the attack was successful (model behavior changed significantly)
        """
        # Simple string similarity check
        similarity = SequenceMatcher(None, original_output, perturbed_output).ratio()
        
        # Consider attack successful if outputs differ significantly
        return similarity < 0.7  # Threshold can be adjusted
    
    def get_success_rate(self) -> float:
        """
        Get the success rate of the attack.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    def get_defense_recommendations(self) -> List[str]:
        """
        Get recommendations for defending against this attack.
        
        Returns:
            List of defense recommendations
        """
        return [
            f"Implement input validation to detect {self.name} attempts",
            "Use robust model fine-tuning to handle perturbed inputs",
            "Consider adversarial training with examples of this attack"
        ]
    
    def get_description(self) -> str:
        """Get a description of the attack."""
        return f"{self.name} - Base class for adversarial attacks"


class RobustnessTester:
    """Orchestrates running adversarial attacks and evaluating results."""
    
    def __init__(self):
        """Initialize the robustness tester."""
        self.attacks = []
        self.metrics = {}
    
    def add_attack(self, attack: AdversarialAttack) -> None:
        """
        Add an attack to the test suite.
        
        Args:
            attack: An adversarial attack implementation
        """
        self.attacks.append(attack)
    
    async def run_tests(self, model_adapter: BaseModelAdapter, test_inputs: List[Dict[str, Any]], target_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic tests on the model."""
        try:
            results = []
            for test_input in test_inputs:
                # Generate output using model
                output = await model_adapter.generate(
                    test_input["text"],
                    target_parameters
                )
                
                # Store result
                result = {
                    "input": test_input["text"],
                    "output": output,
                    "expected": test_input.get("expected", ""),
                    "target_type": target_parameters.get("target_type", "text_generation")
                }
                results.append(result)
            
            return {
                "status": "success",
                "results": results,
                "n_examples": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _evaluate_output(self, model_output: Any, expected: Any) -> bool:
        """
        Evaluate if model output matches expected output.
        
        Args:
            model_output: Output from the model
            expected: Expected output
            
        Returns:
            True if outputs match, False otherwise
        """
        # This is a simplified implementation - in practice, this would be more sophisticated
        # and task-specific (e.g., different for classification vs. generation)
        
        # For text generation, often we'd use similarity metrics rather than exact match
        if isinstance(model_output, dict) and "text" in model_output:
            model_text = model_output["text"].strip()
        else:
            model_text = str(model_output).strip()
            
        expected_text = str(expected).strip()
        
        # For now, just check if the expected text is in the model output
        # In practice, we'd use task-appropriate metrics (BLEU, ROUGE, etc.)
        return expected_text.lower() in model_text.lower()

    def _get_input_text(self, test_input: Dict[str, Any]) -> str:
        """
        Extract input text from test input.
        
        Args:
            test_input: Test input dictionary
            
        Returns:
            Text to use for testing
        """
        if isinstance(test_input.get("text"), str):
            return test_input["text"]
        elif isinstance(test_input.get("text"), dict):
            # Handle structured inputs like question answering
            if "question" in test_input["text"] and "context" in test_input["text"]:
                return f"Question: {test_input['text']['question']}\nContext: {test_input['text']['context']}"
            elif "premise" in test_input["text"] and "hypothesis" in test_input["text"]:
                return f"Premise: {test_input['text']['premise']}\nHypothesis: {test_input['text']['hypothesis']}"
        
        # Default fallback
        return str(test_input.get("text", "")) 