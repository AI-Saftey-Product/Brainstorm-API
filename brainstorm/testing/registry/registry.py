"""Test registry for managing available tests."""
from typing import Dict, Any, Optional, List

class TestRegistry:
    """Registry for managing available tests."""
    
    def __init__(self):
        """Initialize the test registry with available tests."""
        self._tests = {
            # Security Tests - Prompt Injection
            "prompt_injection_test": {
                "id": "prompt_injection_test",
                "name": "Advanced Prompt Injection Test",
                "description": "Tests model resistance to various prompt injection attacks with colored logging and multi-modal support",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.security.prompt_injection_test.PromptInjectionTest",
                "default_config": {
                    "attacks": {
                        "token_smuggling": True,
                        "chain_of_thought": True,
                        "system_prompt_leakage": True,
                        "multi_modal": True,
                        "context_overflow": True,
                        "recursive": True
                    },
                    "data_provider": {
                        "type": "external_security",
                        "source": "adversarial_nli",
                        "max_examples": 1000,
                        "severity_filter": ["high", "medium"]
                    },
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - Jailbreak
            "jailbreak_test": {
                "id": "jailbreak_test",
                "name": "Jailbreak Test",
                "description": "Tests model resistance to jailbreak attempts",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.security.jailbreak_test.JailbreakTest",
                "default_config": {
                    "attack_types": ["role_play", "system_prompt", "context_manipulation"],
                    "max_examples": 100
                }
            },
            
            # Security Tests - Data Extraction
            "data_extraction_test": {
                "id": "data_extraction_test",
                "name": "Data Extraction Test",
                "description": "Tests model resistance to data extraction attempts",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.security.data_extraction_test.DataExtractionTest",
                "default_config": {
                    "attack_types": ["training_data", "system_info", "user_data"],
                    "max_examples": 100
                }
            },
            
            # Robustness Tests - Character Level
            "nlp_character_attack_test": {
                "id": "nlp_character_attack_test",
                "name": "Character-Level Adversarial Test",
                "description": "Tests model robustness against character-level adversarial attacks",
                "category": "robustness",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.CharacterLevelAttack",
                "default_config": {
                    "attack_types": ["swap", "insert", "delete"],
                    "num_examples": 5
                }
            },
            
            # Robustness Tests - Word Level
            "nlp_word_attack_test": {
                "id": "nlp_word_attack_test",
                "name": "Word-Level Adversarial Test",
                "description": "Tests model robustness against word-level adversarial attacks",
                "category": "robustness",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.word_attacks.WordLevelAttack",
                "default_config": {
                    "attack_types": ["synonym", "antonym"],
                    "num_examples": 5
                }
            },
            
            # Robustness Tests - Sentence Level
            "nlp_sentence_attack_test": {
                "id": "nlp_sentence_attack_test",
                "name": "Sentence-Level Adversarial Test",
                "description": "Tests model robustness against sentence-level adversarial attacks",
                "category": "robustness",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.sentence_attacks.SentenceLevelAttack",
                "default_config": {
                    "attack_types": ["paraphrase", "restructure"],
                    "num_examples": 5
                }
            },
            
            # Robustness Tests - Advanced
            "nlp_advanced_attack_test": {
                "id": "nlp_advanced_attack_test",
                "name": "Advanced Adversarial Test",
                "description": "Tests model robustness against advanced adversarial attacks",
                "category": "robustness",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.TextFoolerAttack",
                "default_config": {
                    "attack_types": ["combined", "targeted"],
                    "num_examples": 5
                }
            },
            
            # Robustness Tests - Optimized
            "nlp_optimized_robustness_test": {
                "id": "nlp_optimized_robustness_test",
                "name": "Optimized Robustness Test",
                "description": "Optimized test suite for model robustness evaluation",
                "category": "robustness",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.optimized_robustness_test.OptimizedRobustnessTest",
                "default_config": {
                    "test_types": ["character", "word", "sentence"],
                    "num_examples": 10,
                    "optimization_level": "medium"
                }
            },
            
            # Bias Tests - Honest
            "nlp_honest_test": {
                "id": "nlp_honest_test",
                "name": "Honest Bias Test",
                "description": "Tests model for honest responses to bias-sensitive questions",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.honest_test.HONESTTest",
                "default_config": {
                    "categories": ["gender", "race", "age"],
                    "num_examples": 10
                }
            },
            
            # Bias Tests - CDA
            "nlp_cda_test": {
                "id": "nlp_cda_test",
                "name": "Counterfactual Data Augmentation Bias Test",
                "description": "Tests model bias using counterfactual data augmentation",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.cda_test.CDATest",
                "default_config": {
                    "categories": ["gender", "race", "age"],
                    "num_examples": 10,
                    "augmentation_types": ["swap", "neutral"]
                }
            },
            
            # Bias Tests - Intersectional
            "nlp_intersectional_test": {
                "id": "nlp_intersectional_test",
                "name": "Intersectional Bias Test",
                "description": "Tests for intersectional biases across multiple demographic dimensions",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.intersectional_test.IntersectionalBiasTest",
                "default_config": {
                    "dimensions": ["gender", "race", "age", "socioeconomic"],
                    "num_examples": 10,
                    "intersection_depth": 2
                }
            },
            
            # Bias Tests - Question Answering
            "nlp_qa_test": {
                "id": "nlp_qa_test",
                "name": "Question-Answering Bias Test",
                "description": "Tests for biases in question-answering behavior",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.qa_test.QABiasTest",
                "default_config": {
                    "question_types": ["factual", "opinion", "hypothetical"],
                    "num_examples": 10,
                    "categories": ["gender", "race", "age"]
                }
            },
            
            # Bias Tests - Occupational
            "nlp_occupational_test": {
                "id": "nlp_occupational_test",
                "name": "Occupational Bias Test",
                "description": "Tests for gender and occupation-related biases",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.occupational_test.OccupationalBiasTest",
                "default_config": {
                    "occupation_categories": ["professional", "service", "manual"],
                    "num_examples": 10,
                    "test_types": ["association", "stereotype"]
                }
            },
            
            # Bias Tests - Multilingual
            "nlp_multilingual_test": {
                "id": "nlp_multilingual_test",
                "name": "Multilingual Bias Test",
                "description": "Tests for biases across different languages",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.multilingual_test.MultilingualBiasTest",
                "default_config": {
                    "languages": ["en", "es", "fr", "de", "zh"],
                    "num_examples": 10,
                    "categories": ["gender", "race", "age"]
                }
            },
            
            # Bias Tests - CrowS-Pairs
            "nlp_crows_pairs_test": {
                "id": "nlp_crows_pairs_test",
                "name": "CrowS-Pairs Bias Test",
                "description": "Tests for stereotypical biases using CrowS-Pairs dataset",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.crows_pairs_test.CrowSPairsTest",
                "default_config": {
                    "bias_types": ["gender", "race", "religion", "age", "disability"],
                    "num_examples": 10,
                    "evaluation_metrics": ["stereotype_score", "bias_score"]
                }
            },
            
            # Bias Tests - Intersect Bench
            "nlp_intersect_bench_test": {
                "id": "nlp_intersect_bench_test",
                "name": "Intersect Bench Bias Test",
                "description": "Tests for intersectional biases using Intersect Bench dataset",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.intersect_bench_test.IntersectBenchTest",
                "default_config": {
                    "intersections": ["gender_race", "gender_age", "race_age"],
                    "num_examples": 10,
                    "evaluation_metrics": ["disparity_score", "fairness_gap"]
                }
            }
        }
        
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a test by its ID."""
        return self._tests.get(test_id)
        
    def get_tests_by_category(self, category: str) -> Dict[str, Any]:
        """Get all tests in a specific category."""
        return {
            test_id: test_info 
            for test_id, test_info in self._tests.items()
            if test_info.get("category") == category
        }
        
    def get_tests_by_modality(self, modality: str) -> Dict[str, Any]:
        """Get all tests compatible with a specific modality."""
        return {
            test_id: test_info
            for test_id, test_info in self._tests.items()
            if modality in test_info.get("compatible_modalities", [])
        }
        
    def get_tests_by_model_type(self, model_type: str) -> Dict[str, Any]:
        """Get all tests compatible with a specific model type."""
        return {
            test_id: test_info
            for test_id, test_info in self._tests.items()
            if model_type in test_info.get("compatible_sub_types", [])
        }

# Create a singleton instance
test_registry = TestRegistry()