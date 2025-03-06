"""Base classes for adversarial attacks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple


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
    
    async def run_tests(self, 
                         model_adapter: Any, 
                         test_inputs: List[Dict[str, Any]], 
                         model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all registered attacks on the test inputs.
        
        Args:
            model_adapter: Model adapter to use for testing
            test_inputs: List of test inputs with expected outputs
            model_parameters: Parameters for model invocation
            
        Returns:
            Test results containing metrics and example attacks
        """
        results = {
            "clean_accuracy": 0.0,
            "adversarial_accuracy": 0.0,
            "adversarial_success_rate": 0.0,
            "accuracy_degradation": 0.0,
            "robustness_score": 0.0,
            "attack_results": [],
            "attack_examples": []
        }
        
        # First, evaluate on clean inputs to get baseline
        clean_correct = 0
        for input_item in test_inputs:
            text = input_item.get("text", "")
            expected = input_item.get("expected", "")
            
            # Run model on clean input
            model_output = await model_adapter.invoke(text, model_parameters)
            
            # Check if output matches expected (simplified)
            is_correct = self._evaluate_output(model_output, expected)
            if is_correct:
                clean_correct += 1
        
        clean_accuracy = clean_correct / len(test_inputs) if test_inputs else 0
        results["clean_accuracy"] = clean_accuracy
        
        # Track results per attack
        attack_results = []
        attack_examples = []
        
        # Then, evaluate on adversarial inputs
        total_adv_correct = 0
        total_adv_attempts = 0
        
        for attack in self.attacks:
            attack_name = attack.name
            adv_correct = 0
            adv_attempts = 0
            examples = []
            
            for input_item in test_inputs:
                text = input_item.get("text", "")
                expected = input_item.get("expected", "")
                
                # Apply the attack
                perturbed_text = await attack.perturb(text)
                
                # Only count examples that were correctly classified originally
                model_output = await model_adapter.invoke(text, model_parameters)
                original_correct = self._evaluate_output(model_output, expected)
                
                if original_correct:
                    adv_attempts += 1
                    
                    # Run model on adversarial input
                    adv_output = await model_adapter.invoke(perturbed_text, model_parameters)
                    
                    # Check if output still matches expected
                    adv_correct_result = self._evaluate_output(adv_output, expected)
                    if adv_correct_result:
                        adv_correct += 1
                    
                    # Add example
                    if len(examples) < 5 and not adv_correct_result:  # Limit to 5 examples per attack
                        examples.append({
                            "original_text": text,
                            "perturbed_text": perturbed_text,
                            "expected_output": expected,
                            "model_output": adv_output,
                            "attack_successful": not adv_correct_result
                        })
            
            # Calculate attack-specific metrics
            attack_success_rate = (adv_attempts - adv_correct) / adv_attempts if adv_attempts > 0 else 0
            attack_results.append({
                "attack_name": attack_name,
                "success_rate": attack_success_rate,
                "attempts": adv_attempts,
                "correct": adv_correct
            })
            
            attack_examples.extend(examples)
            
            # Add to totals
            total_adv_correct += adv_correct
            total_adv_attempts += adv_attempts
        
        # Calculate overall metrics
        if total_adv_attempts > 0:
            adv_accuracy = total_adv_correct / total_adv_attempts
            asr = (total_adv_attempts - total_adv_correct) / total_adv_attempts
            accuracy_degradation = clean_accuracy - adv_accuracy
            robustness_score = adv_accuracy - asr  # Simple robustness score
            
            results["adversarial_accuracy"] = adv_accuracy
            results["adversarial_success_rate"] = asr
            results["accuracy_degradation"] = accuracy_degradation
            results["robustness_score"] = robustness_score
        
        results["attack_results"] = attack_results
        results["attack_examples"] = attack_examples
        
        return results
    
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