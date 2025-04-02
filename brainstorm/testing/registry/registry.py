"""Test registry for managing available tests."""
from typing import Dict, Any, Optional, List

class TestRegistry:
    """Registry for managing available tests."""
    
    def __init__(self):
        """Initialize the test registry with available tests."""
        self._tests = {
            # Performance Tests - APPS
            "apps_test": {
                "id": "apps_test",
                "name": "APPS Test",
                "description": "Tests model performance on Python programming tasks across three difficulty levels (introductory, interview, competition)",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.apps_test.APPSTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "level": "introductory"
                }
            },
            
            # Performance Tests - BigCodeBench
            "bigcodebench_test": {
                "id": "bigcodebench_test",
                "name": "BigCodeBench Test",
                "description": "Tests model performance on code generation tasks from the BigCodeBench dataset",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Benchmark"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.bigcodebench_test.BigCodeBenchTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "version": "v1",
                    "subset": None
                }
            },
            
            # Performance Tests - CORE-Bench
            "core_bench_test": {
                "id": "core_bench_test",
                "name": "CORE-Bench Test",
                "description": "Tests model performance on code generation tasks from the CORE-Bench dataset",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.core_bench_test.COREBenchTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "difficulty": "easy",
                    "field": "all",
                    "language": "python",
                    "capsule_ids": None,
                    "exclude_capsule_ids": None,
                    "filter_out_gpu": False,
                    "vllm_model": None,
                    "max_retries": 3,
                    "backoff_factor": 2,
                    "max_messages": 10,
                    "token_limit": 4096
                }
            },
            
            # Performance Tests - ClassEval
            "class_eval_test": {
                "id": "class_eval_test",
                "name": "ClassEval Test",
                "description": "Tests model performance on class-level Python code generation tasks",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.class_eval_test.ClassEvalTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "few_shot": 1,
                    "few_shot_seed": 42
                }
            },
            
            # Performance Tests - DS-1000
            "ds_1000_test": {
                "id": "ds_1000_test",
                "name": "DS-1000 Test",
                "description": "Tests model performance on data science code generation tasks across seven Python libraries (NumPy, Pandas, TensorFlow, PyTorch, SciPy, Scikit-learn, Matplotlib)",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.ds_1000_test.DS1000Test",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": True,
                    "max_concurrent": 3,
                    "libraries": ["numpy", "pandas", "tensorflow", "pytorch", "scipy", "sklearn", "matplotlib"]
                }
            },
            
            # Performance Tests - HumanEval
            "humaneval_test": {
                "id": "humaneval_test",
                "name": "HumanEval Test",
                "description": "Tests model performance on synthesizing programs from docstrings using the official HumanEval benchmark",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.humaneval_test.HumanEvalTest",
                "default_config": {
                    "epochs": 5,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Performance Tests - MBPP
            "mbpp_test": {
                "id": "mbpp_test",
                "name": "MBPP Test",
                "description": "Tests model performance on synthesizing short Python programs from natural language descriptions using the Mostly Basic Python Programming dataset",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Benchmark"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.mbpp_test.MBPPTest",
                "default_config": {
                    "epochs": 5,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "use_sanitized": True,
                    "difficulty_level": "entry"
                }
            },
            
            # Performance Tests - SciCode
            "scicode_test": {
                "id": "scicode_test",
                "name": "SciCode Test",
                "description": "Tests model performance on generating code to solve scientific research problems from mathematics, physics, chemistry, biology, and materials science",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Benchmark"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.scicode_test.SciCodeTest",
                "default_config": {
                    "epochs": 5,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 65,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "provide_scientific_background": False,
                    "timeout": 300,
                    "include_dev_set": False
                }
            },
            
            # Performance Tests - USACO
            "usaco_test": {
                "id": "usaco_test",
                "name": "USACO Test",
                "description": "Tests model performance on solving USA Computing Olympiad programming problems across four difficulty levels (bronze, silver, gold, platinum).",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Benchmark"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.usaco_test.USACOTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 307,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "difficulty": None,
                    "timeout": 300
                }
            },
            
            # Vision Security Tests - Visual Prompt Injection
            "visual_prompt_injection_test": {
                "id": "visual_prompt_injection_test",
                "name": "CyberSecEval3",
                "description": "Tests model susceptibility to visual prompt injection attacks based on CyberSecEval3 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.vision.security.visual_prompt_injection_test.VisualPromptInjectionTest",
                "default_config": {
                    "epochs": 5,
                    "temperature": 0.6,
                    "max_tokens": 2048,
                    "judge_llm": "gpt-4o",
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - AIR Bench
            "air_bench_test": {
                "id": "air_bench_test",
                "name": "AIR Bench Test",
                "description": "Tests model safety across various risk categories derived from government regulations and company policies",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Safety Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.air_bench_test.AIRBenchTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
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
            
            # Security Tests - CyberSecEval2 Prompt Injection
            "cyberseceval2_prompt_injection_test": {
                "id": "cyberseceval2_prompt_injection_test",
                "name": "CyberSecEval2 Prompt Injection Test",
                "description": "Tests model susceptibility to prompt injection attacks based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Chat"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_prompt_injection_test.CyberSecEval2PromptInjectionTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "judge_llm": "gpt-4o",
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - CyberSecEval2 Vulnerability Exploit
            "cyberseceval2_vulnerability_exploit_test": {
                "id": "cyberseceval2_vulnerability_exploit_test",
                "name": "CyberSecEval2 Vulnerability Exploit Test",
                "description": "Tests model capability to find and exploit vulnerabilities in code based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Security Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_vulnerability_exploit_test.CyberSecEval2VulnerabilityExploitTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - CyberSecEval2 Interpreter Abuse
            "cyberseceval2_interpreter_abuse_test": {
                "id": "cyberseceval2_interpreter_abuse_test",
                "name": "CyberSecEval2 Interpreter Abuse Test",
                "description": "Tests model susceptibility to interpreter abuse attacks based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Security Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_interpreter_abuse_test.CyberSecEval2InterpreterAbuseTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - CyberSecEval2 Code Generation
            "cyberseceval2_code_generation_test": {
                "id": "cyberseceval2_code_generation_test",
                "name": "CyberSecEval2 Code Generation Test",
                "description": "Tests model capability to generate secure code based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Security Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_code_generation_test.CyberSecEval2CodeGenerationTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - CyberSecEval2 Code Understanding
            "cyberseceval2_code_understanding_test": {
                "id": "cyberseceval2_code_understanding_test",
                "name": "CyberSecEval2 Code Understanding Test",
                "description": "Tests model capability to understand and analyze code for security vulnerabilities based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Analysis", "Security Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_code_understanding_test.CyberSecEval2CodeUnderstandingTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - CyberSecEval2 Code Completion
            "cyberseceval2_code_completion_test": {
                "id": "cyberseceval2_code_completion_test",
                "name": "CyberSecEval2 Code Completion Test",
                "description": "Tests model capability to complete code securely based on CyberSecEval2 benchmark",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Code Generation", "Security Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.cyberseceval2_code_completion_test.CyberSecEval2CodeCompletionTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
                }
            },
            
            # Security Tests - XSTest
            "xs_test": {
                "id": "xs_test",
                "name": "XSTest",
                "description": "Tests for exaggerated safety behaviors in LLMs using 250 safe and 200 unsafe prompts",
                "category": "security",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Text Generation", "Safety Analysis"],
                "test_class": "brainstorm.testing.modalities.nlp.security.xs_test.XSTest",
                "default_config": {
                    "epochs": 4,
                    "temperature": 0.75,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3
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
            },
            "assistant_bench_test": {
                "id": "assistant_bench_test",
                "name": "AssistantBench Test",
                "description": "Tests model performance on real-world web-based tasks, evaluating capabilities in information retrieval, analysis, and task completion across different domains.",
                "category": "performance",
                "compatible_modalities": ["NLP"],
                "compatible_sub_types": ["Web Agent", "Benchmark"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.assistant_bench_test.AssistantBenchTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 214,
                    "challenges": None,
                    "shuffle": True,
                    "max_concurrent": 3,
                    "difficulty": None,
                    "token_limit": 500000,
                    "web_search_model": None
                }
            },
            "usaco": {
                "id": "usaco",
                "name": "USACO Test",
                "description": "Evaluates model performance on solving USA Computing Olympiad programming problems across four difficulty levels (bronze, silver, gold, platinum).",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["code"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.usaco_test.USACOTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 307,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 1,
                    "difficulty": None,
                    "timeout": 180,
                    "include_dev_set": False
                }
            },
            "gaia": {
                "id": "gaia",
                "name": "GAIA Test",
                "description": "Evaluates web agent capabilities on real-world tasks, consisting of 450 questions testing tool use on realistic assistant tasks, primarily web browsing.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["agent", "web_agent"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.gaia_test.GAIATest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 450,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 3,
                    "subset": "2023_all",
                    "split": "validation",
                    "instance_ids": None,
                    "max_attempts": 1,
                    "max_messages": 100,
                    "code_timeout": 180
                }
            },
            "mind2web": {
                "id": "mind2web",
                "name": "Mind2Web Test",
                "description": "Evaluates web agent capabilities on real-world web tasks across diverse websites and domains, testing generalization and task completion abilities.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["web_agent"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.mind2web_test.Mind2WebTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 1,
                    "split": "test_domain",
                    "domain": None,
                    "subdomain": None,
                    "max_attempts": 1,
                    "max_messages": 100,
                    "code_timeout": 180
                }
            },
            "sycophancy": {
                "id": "sycophancy",
                "name": "Sycophancy Test",
                "description": "Evaluates model tendency to agree with incorrect assertions when challenged, based on Anthropic's sycophancy benchmark.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["text_generation"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.sycophancy_test.SycophancyTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "num_test_cases": 100,
                    "shuffle": False,
                    "max_concurrent": 1,
                    "dataset_names": ["are_you_sure"],
                    "scorer_model": "openai/gpt-4",
                    "max_attempts": 1,
                    "max_messages": 10,
                    "code_timeout": 120
                }
            },
            "agentharm": {
                "id": "agentharm",
                "name": "AgentHarm Test",
                "description": "Evaluates harmful behavior in LLM agents across varied tasks and scenarios, measuring refusal behavior and response appropriateness.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["agent"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.agentharm_test.AgentHarmTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "max_concurrent": 1,
                    "split": "test_public",
                    "behavior_ids": [],
                    "detailed_behaviors": None,
                    "hint_included": None,
                    "n_irrelevant_tools": 0,
                    "refusal_judge": "openai/gpt-4",
                    "semantic_judge": "openai/gpt-4",
                    "max_attempts": 1,
                    "max_messages": 100,
                    "code_timeout": 180
                }
            },
            "lab_bench": {
                "id": "lab_bench",
                "name": "LAB-Bench Test",
                "description": "Evaluates LLM capabilities for scientific research workflows across domains like chemistry, biology, materials science, and general science tasks.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["text_generation", "multimodal"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.lab_bench_test.LABBenchTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "num_test_cases": 100,
                    "challenges": None,
                    "shuffle": False,
                    "shuffle_choices": False,
                    "max_concurrent": 3,
                    "subsets": ["LitQA2"],
                    "metrics": ["precision", "coverage"]
                }
            },
            "wmdp": {
                "id": "wmdp",
                "name": "WMDP Test",
                "description": "Evaluates hazardous knowledge in LLMs across biosecurity, cybersecurity, and chemical security domains using the Weapons of Mass Destruction Proxy dataset.",
                "category": "performance",
                "compatible_modalities": ["nlp"],
                "compatible_sub_types": ["text_generation"],
                "test_class": "brainstorm.testing.modalities.nlp.performance.wmdp_test.WMDPTest",
                "default_config": {
                    "epochs": 1,
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "num_test_cases": 500,
                    "domains": ["bio", "chem", "cyber"],
                    "shuffle": False,
                    "max_concurrent": 5
                }
            }
        }
        
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a test by its ID."""
        return self._tests.get(test_id)
        
    def get_tests_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tests in a category."""
        return [test for test in self._tests.values() if test["category"] == category]

    def get_tests_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """Get all tests for a modality."""
        return [test for test in self._tests.values() if modality in test["compatible_modalities"]]

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Get all available tests."""
        return list(self._tests.values())

# Create a singleton instance
test_registry = TestRegistry()