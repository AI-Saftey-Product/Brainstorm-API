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
                "default_config": {
                    "attack_types": ["typo", "unicode"],
                    "num_examples": 5,
                    "attack_params": {
                        "typo": {
                            "max_changes": 2,
                            "preserve_phonetic": True
                        },
                        "unicode": {
                            "max_changes": 2,
                            "use_homoglyphs": True
                        }
                    }
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
            "default_config": {
                    "attack_types": ["scramble", "synonym"],
                    "num_examples": 5,
                    "attack_params": {
                        "scramble": {
                            "preserve_first_last": True,
                            "max_words": 3
                        },
                        "synonym": {
                            "use_wordnet": True,
                            "max_swaps": 2
                        }
                    }
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
                "default_config": {
                    "attack_types": ["paraphrase", "shuffle"],
                    "num_examples": 5,
                    "attack_params": {
                        "paraphrase": {
                            "preserve_meaning": True,
                            "max_rewrites": 2
                        },
                        "shuffle": {
                            "preserve_grammar": True,
                            "max_shuffles": 2
                        }
                    }
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
                "default_config": {
                    "attack_types": ["textfooler", "bert", "paraphrase", "redteam"],
                    "num_examples": 5,
                    "attack_params": {
                        "textfooler": {
                            "use_similarity_threshold": 0.8,
                            "max_candidates": 10
                        },
                        "bert": {
                            "use_mlm_scoring": True,
                            "top_k_candidates": 10
                        },
                        "paraphrase": {
                            "diversity_factor": 0.8,
                            "max_paraphrases": 3
                        },
                        "redteam": {
                            "use_templates": True,
                            "max_attempts": 5
                        }
                    }
                }
            },
            
            # Bias Tests
            "nlp_honest_test": {
                "id": "nlp_honest_test",
                "name": "HONEST Stereotype Test",
                "description": "Tests for harmful stereotypes and biases using the HONEST framework",
                "category": "bias",
            "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
            "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "nlp_cda_test": {
                "id": "nlp_cda_test",
                "name": "Counterfactual Data Augmentation Test",
                "description": "Tests for bias using counterfactual data augmentation techniques",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "nlp_intersectional_test": {
                "id": "nlp_intersectional_test",
                "name": "Intersectional Bias Test",
                "description": "Tests for intersectional biases across multiple demographic attributes",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "nlp_qa_bias_test": {
                "id": "nlp_qa_bias_test",
                "name": "Question-Answering Bias Test",
                "description": "Tests for biases in question-answering scenarios",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "nlp_occupation_test": {
                "id": "nlp_occupation_test",
                "name": "Occupational Bias Test",
                "description": "Tests for biases related to occupations and professional roles",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "nlp_multilingual_test": {
                "id": "nlp_multilingual_test",
                "name": "Multilingual Bias Test",
                "description": "Tests for biases across different languages and cultural contexts",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "default_config": {
                    "max_samples": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
        }
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific test by ID."""
        return self._tests.get(test_id)
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tests."""
        return self._tests
    
    def get_tests_by_modality(self, modality: str) -> Dict[str, Dict[str, Any]]:
        """Get tests compatible with a specific modality."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if modality in test["compatible_modalities"]
        }
    
    def get_tests_by_sub_type(self, modality: str, sub_type: str) -> Dict[str, Dict[str, Any]]:
        """Get tests compatible with a specific modality and sub-type."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if modality in test["compatible_modalities"] and sub_type in test["compatible_sub_types"]
        }
    
    def get_tests_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get tests in a specific category."""
        return {
            test_id: test for test_id, test in self._tests.items()
            if test["category"] == category
        }

# Create a singleton instance
test_registry = TestRegistry() 