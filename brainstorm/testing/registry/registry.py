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
            
            # Security Tests - Strong Reject
            "strong_reject_test": {
                "id": "strong_reject_test",
                "name": "StrongREJECT Test",
                "description": "Tests LLM's ability to refuse harmful requests based on the StrongREJECT benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.security.strong_reject_test.StrongRejectTest",
                "default_config": {
                    "judge_model_id": "gpt-4o",
                    "judge_model_provider": "openai",
                    "jailbreak_method": "AIM",
                    "max_examples": 10
                }
            },
            
            # Adversarial Tests - Character Level
            "character_attack_test": {
                "id": "character_attack_test",
                "name": "Character-Level Adversarial Test",
                "description": "Tests model robustness against character-level adversarial attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.CharacterLevelAttack",
                "default_config": {
                    "attack_types": ["swap", "insert", "delete"],
                    "num_examples": 5
                }
            },
            
            # Adversarial Tests - Word Level
            "word_attack_test": {
                "id": "word_attack_test",
                "name": "Word-Level Adversarial Test",
                "description": "Tests model robustness against word-level adversarial attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.word_attacks.WordLevelAttack",
                "default_config": {
                    "attack_types": ["synonym", "antonym"],
                    "num_examples": 5
                }
            },
            
            # Adversarial Tests - Sentence Level
            "sentence_attack_test": {
                "id": "sentence_attack_test",
                "name": "Sentence-Level Adversarial Test",
                "description": "Tests model robustness against sentence-level adversarial attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.sentence_attacks.SentenceLevelAttack",
                "default_config": {
                    "attack_types": ["paraphrase", "restructure"],
                    "num_examples": 5
                }
            },
            
            # Adversarial Tests - Advanced
            "textfooler_attack_test": {
                "id": "textfooler_attack_test",
                "name": "TextFooler Attack Test",
                "description": "Tests model robustness against TextFooler attacks using contextualized embeddings",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.TextFoolerAttack",
                "default_config": {
                    "max_perturbations": 5,
                    "similarity_threshold": 0.8,
                    "use_pos_tagging": True
                }
            },
            
            "bert_attack_test": {
                "id": "bert_attack_test",
                "name": "BERT Attack Test",
                "description": "Tests model robustness against BERT-based adversarial attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.BERTAttack",
                "default_config": {
                    "max_perturbations": 5,
                    "mask_probability": 0.15,
                    "use_context": True
                }
            },
            
            "neural_paraphrase_attack_test": {
                "id": "neural_paraphrase_attack_test",
                "name": "Neural Paraphrase Attack Test",
                "description": "Tests model robustness against neural paraphrase-based attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.NeuralParaphraseAttack",
                "default_config": {
                    "max_paraphrases": 3,
                    "similarity_threshold": 0.8,
                    "use_beam_search": True
                }
            },
            
            "jailbreak_attack_test": {
                "id": "jailbreak_attack_test",
                "name": "Jailbreak Attack Test",
                "description": "Tests model robustness against jailbreak attempts",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.JailbreakAttack",
                "default_config": {
                    "max_attempts": 5,
                    "use_authority": True,
                    "use_context": True
                }
            },
            
            "redteam_attack_test": {
                "id": "redteam_attack_test",
                "name": "Red Team Attack Test",
                "description": "Tests model robustness against red team adversarial attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.advanced_attacks.RedTeamAttack",
                "default_config": {
                    "attack_types": ["authority", "misalignment", "context", "boundary"],
                    "max_perturbations": 3,
                    "use_all_techniques": True
                }
            },
            
            # Robustness Tests - Optimized
            "optimized_robustness_test": {
                "id": "optimized_robustness_test",
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
            "honest_test": {
                "id": "honest_test",
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
            "cda_test": {
                "id": "cda_test",
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
            "intersectional_test": {
                "id": "intersectional_test",
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
            "qa_bias_test": {
                "id": "qa_bias_test",
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
            "occupational_test": {
                "id": "occupational_test",
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
            "multilingual_test": {
                "id": "multilingual_test",
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
            "crows_pairs_test": {
                "id": "crows_pairs_test",
                "name": "CrowS-Pairs Bias Test",
                "description": "Tests for stereotypical biases using CrowS-Pairs dataset",
                "category": "bias",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.bias.crows_pairs_test.CrowSPairsTest",
                "default_config": {
                    "bias_types": ["gender", "race", "religion", "age", "disability"],
                    "num_examples": 10
                }
            },
            
            # Adversarial Tests - Character Level
            "typo_attack_test": {
                "id": "typo_attack_test",
                "name": "Typo Attack Test",
                "description": "Tests model robustness against typo-based character perturbations",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.TypoAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "preserve_word_boundaries": True
                }
            },
            
            "homoglyph_attack_test": {
                "id": "homoglyph_attack_test",
                "name": "Homoglyph Attack Test",
                "description": "Tests model robustness against homoglyph-based character substitutions",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.HomoglyphAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "preserve_word_boundaries": True
                }
            },
            
            "punctuation_attack_test": {
                "id": "punctuation_attack_test",
                "name": "Punctuation Attack Test",
                "description": "Tests model robustness against punctuation-based perturbations",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.PunctuationAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "preserve_word_boundaries": True
                }
            },
            
            "unicode_attack_test": {
                "id": "unicode_attack_test",
                "name": "Unicode Attack Test",
                "description": "Tests model robustness against Unicode-based character perturbations",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.character_attacks.UnicodeAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "preserve_word_boundaries": True
                }
            },
            
            # Adversarial Tests - Word Level
            "synonym_attack_test": {
                "id": "synonym_attack_test",
                "name": "Synonym Attack Test",
                "description": "Tests model robustness against synonym-based word substitutions",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.word_attacks.SynonymAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "min_word_length": 4,
                    "preserve_pos": True
                }
            },
            
            "word_scramble_attack_test": {
                "id": "word_scramble_attack_test",
                "name": "Word Scramble Attack Test",
                "description": "Tests model robustness against scrambled word attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.word_attacks.WordScrambleAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "min_word_length": 4,
                    "preserve_pos": True
                }
            },
            
            "word_insert_delete_attack_test": {
                "id": "word_insert_delete_attack_test",
                "name": "Word Insert/Delete Attack Test",
                "description": "Tests model robustness against word insertion and deletion attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.word_attacks.WordInsertDeleteAttack",
                "default_config": {
                    "intensity": 0.1,
                    "max_perturbations": 3,
                    "min_word_length": 4,
                    "preserve_pos": True,
                    "insert_probability": 0.5
                }
            },
            
            # Adversarial Tests - Sentence Level
            "distractor_sentence_attack_test": {
                "id": "distractor_sentence_attack_test",
                "name": "Distractor Sentence Attack Test",
                "description": "Tests model robustness against distractor sentence attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.sentence_attacks.DistractorSentenceAttack",
                "default_config": {
                    "intensity": 0.5,
                    "max_perturbations": 1,
                    "preserve_meaning": True,
                    "distractor_types": ["irrelevant", "contradictory", "confusing"]
                }
            },
            
            "sentence_shuffle_attack_test": {
                "id": "sentence_shuffle_attack_test",
                "name": "Sentence Shuffle Attack Test",
                "description": "Tests model robustness against sentence reordering attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.sentence_attacks.SentenceShuffleAttack",
                "default_config": {
                    "intensity": 0.5,
                    "max_perturbations": 1,
                    "preserve_meaning": True
                }
            },
            
            "paraphrase_attack_test": {
                "id": "paraphrase_attack_test",
                "name": "Paraphrase Attack Test",
                "description": "Tests model robustness against various paraphrase-based attacks",
                "category": "adversarial",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Question Answering", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.adversarial.sentence_attacks.ParaphraseAttack",
                "default_config": {
                    "intensity": 0.5,
                    "max_perturbations": 1,
                    "preserve_meaning": True,
                    "paraphrase_types": [
                        "active_to_passive",
                        "passive_to_active",
                        "combine_sentences",
                        "split_sentence",
                        "reorder_clauses",
                        "change_connectors",
                        "simple_reorder"
                    ]
                }
            }
        }
        
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test information by ID."""
        return self._tests.get(test_id)
        
    def get_tests_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tests in a category."""
        return [test for test in self._tests.values() if test["category"] == category]

    def get_tests_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """Get all tests compatible with a modality."""
        return [test for test in self._tests.values() if modality in test["compatible_modalities"]]

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Get all registered tests."""
        return list(self._tests.values())

# Create a singleton instance
test_registry = TestRegistry()