"""Test registry for AI safety tests."""
from typing import Dict, List, Any, Optional


class TestRegistry:
    """Registry for all available safety tests."""
    
    def __init__(self):
        self._tests = {}
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize the registry with available tests."""
        # NLP Tests
        self._register_nlp_tests()
    
    def _register_nlp_tests(self) -> None:
        """Register NLP tests in the registry."""
        # Bias tests
        self._tests["nlp_bias_test"] = {
            "id": "nlp_bias_test",
            "name": "NLP Bias Detection Test",
            "description": "Tests for various forms of bias in NLP model outputs",
            "category": "bias",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Text Generation", "Text2Text Generation", "Question Answering",
                "Text Classification", "Zero-Shot Classification", "Summarization"
            ],
        }
        
        # Toxicity tests
        self._tests["nlp_toxicity_test"] = {
            "id": "nlp_toxicity_test",
            "name": "NLP Toxicity Detection Test",
            "description": "Tests for toxic content generation in NLP model outputs",
            "category": "toxicity",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Text Generation", "Text2Text Generation"
            ],
        }
        
        # Hallucination tests
        self._tests["nlp_hallucination_test"] = {
            "id": "nlp_hallucination_test",
            "name": "NLP Hallucination Detection Test",
            "description": "Tests for hallucinations and factual inconsistencies in NLP model outputs",
            "category": "hallucination",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Text Generation", "Text2Text Generation", "Question Answering",
                "Summarization"
            ],
        }
        
        # Security tests
        self._tests["nlp_security_test"] = {
            "id": "nlp_security_test",
            "name": "NLP Security Vulnerability Test",
            "description": "Tests for prompt injection and security vulnerabilities in NLP models",
            "category": "security",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Text Generation", "Text2Text Generation"
            ],
        }
        
        # Classification-specific tests
        self._tests["nlp_classification_fairness_test"] = {
            "id": "nlp_classification_fairness_test",
            "name": "NLP Classification Fairness Test",
            "description": "Tests for fairness across different demographic groups in classification outputs",
            "category": "bias",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Text Classification", "Zero-Shot Classification"
            ],
        }
        
        # Factual accuracy tests
        self._tests["nlp_factual_accuracy_test"] = {
            "id": "nlp_factual_accuracy_test",
            "name": "NLP Factual Accuracy Test",
            "description": "Tests for factual accuracy in NLP model responses",
            "category": "hallucination",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Question Answering", "Summarization"
            ],
        }
        
        # Zero-shot specific tests
        self._tests["nlp_zero_shot_robustness_test"] = {
            "id": "nlp_zero_shot_robustness_test",
            "name": "NLP Zero-Shot Robustness Test",
            "description": "Tests for robustness of zero-shot classification across different prompts",
            "category": "bias",
            "compatible_modalities": ["NLP"],
            "compatible_sub_types": [
                "Zero-Shot Classification"
            ],
        }
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tests."""
        return self._tests
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test by ID."""
        return self._tests.get(test_id)
    
    def get_tests_by_modality(self, modality: str) -> Dict[str, Dict[str, Any]]:
        """Get all tests compatible with a specific modality."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if modality in test["compatible_modalities"]
        }
    
    def get_tests_by_sub_type(self, modality: str, sub_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all tests compatible with a specific model sub-type."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if modality in test["compatible_modalities"] and sub_type in test["compatible_sub_types"]
        }
    
    def get_tests_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all tests in a specific category."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if test["category"] == category
        }


# Singleton instance
test_registry = TestRegistry() 