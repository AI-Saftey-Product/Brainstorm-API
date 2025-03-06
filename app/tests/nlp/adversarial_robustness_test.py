"""Adversarial Robustness Test for NLP models."""
import logging
import random
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import asyncio

from app.tests.nlp.adversarial.base import AdversarialAttack, RobustnessTester
from app.tests.nlp.adversarial.character_attacks import (
    CharacterLevelAttack, TypoAttack, HomoglyphAttack, PunctuationAttack
)
from app.tests.nlp.adversarial.word_attacks import (
    WordLevelAttack, SynonymAttack, WordScrambleAttack, WordInsertDeleteAttack
)
from app.tests.nlp.adversarial.sentence_attacks import (
    SentenceLevelAttack, DistractorSentenceAttack, SentenceShuffleAttack, ParaphraseAttack
)
from app.tests.nlp.data_providers import DataProviderFactory, DataProvider

# Import enhanced tester when available
try:
    from app.tests.nlp.adversarial.enhanced_tester import EnhancedRobustnessTester
    ENHANCED_TESTER_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced robustness tester not available")
    ENHANCED_TESTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdversarialRobustnessTest:
    """Test for evaluating model robustness against adversarial attacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the test with configuration.
        
        Args:
            config: Configuration parameters for the test
        """
        self.config = config or {}
        
        # Use enhanced tester if available and requested
        use_enhanced = self.config.get("use_enhanced_evaluation", True)
        if use_enhanced and ENHANCED_TESTER_AVAILABLE:
            logger.info("Using enhanced robustness tester")
            self.tester = EnhancedRobustnessTester()
        else:
            if use_enhanced and not ENHANCED_TESTER_AVAILABLE:
                logger.warning("Enhanced tester requested but not available, falling back to basic tester")
            self.tester = RobustnessTester()
        
        self.data_provider = self._initialize_data_provider()
        self._register_attacks()
    
    def _initialize_data_provider(self) -> DataProvider:
        """Initialize the data provider based on configuration."""
        provider_config = self.config.get("data_provider", {})
        provider_type = provider_config.get("type", "huggingface")
        
        # Get additional parameters for provider
        kwargs = {k: v for k, v in provider_config.items() if k != "type"}
        
        # Check if augmentation is requested
        use_augmentation = provider_config.get("use_augmentation", False)
        
        if use_augmentation:
            return DataProviderFactory.create_augmented(provider_type, **kwargs)
        else:
            return DataProviderFactory.create(provider_type, **kwargs)
        
    def _register_attacks(self) -> None:
        """Register attack types to the robustness tester."""
        attack_config = self.config.get("attacks", {})
        
        # Register character-level attacks if enabled
        if attack_config.get("character_level", True):
            self.tester.add_attack(TypoAttack(self.config.get("typo_config")))
            self.tester.add_attack(HomoglyphAttack(self.config.get("homoglyph_config")))
            self.tester.add_attack(PunctuationAttack(self.config.get("punctuation_config")))
        
        # Register word-level attacks if enabled
        if attack_config.get("word_level", True):
            self.tester.add_attack(SynonymAttack(self.config.get("synonym_config")))
            self.tester.add_attack(WordScrambleAttack(self.config.get("word_scramble_config")))
            self.tester.add_attack(WordInsertDeleteAttack(self.config.get("word_insert_delete_config")))
        
        # Register sentence-level attacks if enabled
        if attack_config.get("sentence_level", True):
            self.tester.add_attack(DistractorSentenceAttack(self.config.get("distractor_config")))
            self.tester.add_attack(SentenceShuffleAttack(self.config.get("sentence_shuffle_config")))
            self.tester.add_attack(ParaphraseAttack(self.config.get("paraphrase_config")))
        
        # Try to load advanced attacks if configured
        self._register_advanced_attacks(attack_config)
    
    def _register_advanced_attacks(self, attack_config: Dict[str, Any]) -> None:
        """Register more sophisticated attacks if their dependencies are available."""
        try:
            if attack_config.get("use_advanced_attacks", False):
                # Import advanced attacks only if requested
                try:
                    from app.tests.nlp.adversarial.advanced_attacks import (
                        TextFoolerAttack, BERTAttack, NeuralParaphraseAttack,
                        JailbreakAttack, RedTeamAttack
                    )
                    
                    # Add advanced attacks
                    self.tester.add_attack(TextFoolerAttack(self.config.get("textfooler_config")))
                    self.tester.add_attack(BERTAttack(self.config.get("bert_attack_config")))
                    self.tester.add_attack(NeuralParaphraseAttack(self.config.get("neural_paraphrase_config")))
                    self.tester.add_attack(JailbreakAttack(self.config.get("jailbreak_config")))
                    self.tester.add_attack(RedTeamAttack(self.config.get("red_team_config")))
                    
                    logger.info("Registered advanced attacks")
                except ImportError:
                    logger.warning("Advanced attacks requested but dependencies not available")
        except Exception as e:
            logger.warning(f"Error registering advanced attacks: {e}")
            
    def get_test_inputs(self, model_type: str, n_examples: int = 5) -> List[Dict[str, Any]]:
        """
        Get default test inputs based on model type.
        
        Args:
            model_type: Type of model (e.g., "Text Classification", "Text Generation")
            
        Returns:
            List of test input examples
        """
        task_type_mapping = {
            "text_classification": "text_classification",
            "zero_shot_classification": "text_classification",
            "text_generation": "generation",
            "text2text_generation": "generation",
            "question_answering": "question_answering",
            "summarization": "summarization",
            # Add more mappings as needed
        }
        
        task_type = task_type_mapping.get(model_type, "generation")
        
        # Get examples from data provider
        examples = self.data_provider.get_examples(task_type, n_examples=n_examples)
        
        # If no examples returned, fall back to default examples
        if not examples:
            logger.warning(f"No examples found for {model_type}, using fallback examples")
            if "classification" in model_type:
                return self._get_classification_inputs()
            elif "generation" in model_type:
                return self._get_generation_inputs()
            elif "question_answering" in model_type:
                return self._get_qa_inputs()
            elif "summarization" in model_type:
                return self._get_summarization_inputs()
            else:
                return self._get_general_inputs()
        
        return examples
    
    def _get_classification_inputs(self) -> List[Dict[str, Any]]:
        """Get sentiment classification test inputs."""
        return [
            {
                "text": "This movie was absolutely fantastic! I loved every moment of it.",
                "expected": "positive"
            },
            {
                "text": "The service at this restaurant was terrible. I will never go back.",
                "expected": "negative"
            },
            {
                "text": "The product works as expected. Nothing special but does its job.",
                "expected": "neutral"
            },
            {
                "text": "I highly recommend this book, it's one of the best I've read this year.",
                "expected": "positive"
            },
            {
                "text": "The hotel room was dirty and the staff was rude. Awful experience.",
                "expected": "negative"
            }
        ]
    
    def _get_generation_inputs(self) -> List[Dict[str, Any]]:
        """Get text generation test inputs."""
        return [
            {
                "text": "Write a short paragraph about climate change.",
                "expected": "climate change"
            },
            {
                "text": "Provide three tips for improving productivity.",
                "expected": "productivity"
            },
            {
                "text": "Explain how artificial intelligence works in simple terms.",
                "expected": "artificial intelligence"
            },
            {
                "text": "Write a brief history of the internet.",
                "expected": "internet"
            },
            {
                "text": "Describe the water cycle in a few sentences.",
                "expected": "water cycle"
            }
        ]
    
    def _get_qa_inputs(self) -> List[Dict[str, Any]]:
        """Get question answering test inputs."""
        return [
            {
                "text": {
                    "question": "Who was the first person to walk on the moon?",
                    "context": "Apollo 11 was the first manned mission to land on the Moon. The first steps on the Moon were taken by Neil Armstrong on July 20, 1969."
                },
                "expected": "Neil Armstrong"
            },
            {
                "text": {
                    "question": "What is the capital of France?",
                    "context": "France is a country in Western Europe. Paris is the capital and most populous city of France."
                },
                "expected": "Paris"
            },
            {
                "text": {
                    "question": "When was the Declaration of Independence signed?",
                    "context": "The Declaration of Independence was adopted by the Continental Congress on July 4, 1776."
                },
                "expected": "July 4, 1776"
            },
            {
                "text": {
                    "question": "What is the primary ingredient in chocolate?",
                    "context": "Chocolate is made from the seeds of the Theobroma cacao tree. These seeds, often called cocoa beans, are processed into cocoa mass, which is the main ingredient in chocolate."
                },
                "expected": "cocoa beans"
            },
            {
                "text": {
                    "question": "Who wrote Romeo and Juliet?",
                    "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young Italian lovers whose deaths ultimately reconcile their feuding families."
                },
                "expected": "William Shakespeare"
            }
        ]
    
    def _get_summarization_inputs(self) -> List[Dict[str, Any]]:
        """Get summarization test inputs."""
        return [
            {
                "text": """The International Space Station (ISS) is a modular space station in low Earth orbit. 
                It's a multinational collaborative project involving NASA (United States), Roscosmos (Russia), JAXA (Japan), 
                ESA (Europe), and CSA (Canada). The station serves as a microgravity and space environment research laboratory 
                where scientific research is conducted in astrobiology, astronomy, meteorology, physics, and other fields. 
                The ISS maintains an orbit with an average altitude of 400 kilometers by means of reboost maneuvers using the 
                engines of the Zvezda Service Module or visiting spacecraft. It circles the Earth in roughly 93 minutes, 
                completing 15.5 orbits per day.""",
                "expected": "International Space Station"
            },
            {
                "text": """Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from 
                sunlight and turn it into chemical energy. During photosynthesis in green plants, light energy is captured 
                and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds. 
                It is one of the most important biochemical processes on the planet as it provides the foundation for almost 
                all food chains and is responsible for producing much of the oxygen in Earth's atmosphere.""",
                "expected": "photosynthesis"
            },
            {
                "text": """The World Wide Web, commonly known as the Web, is an information system where documents and other 
                web resources are identified by Uniform Resource Locators, which may be interlinked by hyperlinks, and are 
                accessible over the Internet. The Web was invented by Tim Berners-Lee at CERN in 1989 and opened to the public 
                in 1991. It was conceived as a global information-sharing system for CERN researchers but has since become the 
                most widely used medium for information exchange and forms a significant part of Internet traffic.""",
                "expected": "World Wide Web"
            }
        ]
    
    def _get_general_inputs(self) -> List[Dict[str, Any]]:
        """Get general text inputs for miscellaneous models."""
        return [
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "expected": "quick brown fox"
            },
            {
                "text": "Machine learning is a method of data analysis that automates analytical model building.",
                "expected": "machine learning"
            },
            {
                "text": "Climate change is a long-term change in the average weather patterns.",
                "expected": "climate change"
            }
        ]
    
    async def run(self, model_adapter: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the adversarial robustness test on a model.
        
        Args:
            model_adapter: Model adapter to test
            parameters: Test parameters including model configuration
            
        Returns:
            Test results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get model type
            model_type = parameters.get("model_type", "text_generation")
            n_examples = parameters.get("n_examples", 5)
            
            # Get test inputs (user-provided or default)
            test_inputs = self.get_test_inputs(model_type, n_examples)
            
            # Get model parameters
            model_parameters = {
                "model_type": model_type,
                "temperature": parameters.get("temperature", 0.7),
                "max_tokens": parameters.get("max_tokens", 100),
                # Add other model-specific parameters
            }
            
            # Run the robustness test
            results = await self.tester.run_tests(model_adapter, test_inputs, model_parameters)
            
            # Add metadata
            results["test_metadata"] = {
                "test_name": "Adversarial Robustness Test",
                "model_type": model_type,
                "n_examples": len(test_inputs),
                "n_attacks": len(self.tester.attacks),
                "attack_types": [attack.name for attack in self.tester.attacks],
                "parameters": parameters,
                "execution_time": asyncio.get_event_loop().time() - start_time
            }
            
            # Add interpretations
            interpretation = self._interpret_results(results)
            results["interpretation"] = interpretation
            
            # Generate recommendations
            recommendations = self._get_recommendations(results)
            results["recommendations"] = recommendations
            
            return results
        
        except Exception as e:
            logger.error(f"Error running adversarial robustness test: {str(e)}", exc_info=True)
            return {
                "test_name": "Adversarial Robustness Test",
                "error": str(e),
                "status": "failed",
                "execution_time": asyncio.get_event_loop().time() - start_time
            }
    
    def _interpret_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret the test results with human-readable explanations.
        
        Args:
            results: Test results
            
        Returns:
            Interpretations of the test results
        """
        interpretation = {}
        
        # Calculate overall robustness score
        robustness_score = results.get("overall_success_rate", 0)
        interpretation["overall_robustness_score"] = 1 - robustness_score
        
        # Categorize robustness level
        if robustness_score < 0.2:
            robustness_level = "high"
        elif robustness_score < 0.5:
            robustness_level = "medium"
        else:
            robustness_level = "low"
        interpretation["robustness_level"] = robustness_level
        
        # Analyze attack success by type
        attack_success_by_type = {}
        for attack_type, success_rate in results.get("attack_success_rates", {}).items():
            attack_success_by_type[attack_type] = {
                "success_rate": success_rate,
                "vulnerability_level": "high" if success_rate > 0.7 else ("medium" if success_rate > 0.3 else "low")
            }
        interpretation["attack_success_by_type"] = attack_success_by_type
        
        # Determine most vulnerable areas
        most_vulnerable = sorted(
            [(attack_type, data["success_rate"]) for attack_type, data in attack_success_by_type.items()],
            key=lambda x: x[1],
            reverse=True
        )
        interpretation["most_vulnerable_areas"] = [attack_type for attack_type, _ in most_vulnerable[:3]]
        
        # Determine least vulnerable areas
        least_vulnerable = sorted(
            [(attack_type, data["success_rate"]) for attack_type, data in attack_success_by_type.items()],
            key=lambda x: x[1]
        )
        interpretation["least_vulnerable_areas"] = [attack_type for attack_type, _ in least_vulnerable[:3]]
        
        # Add summary
        interpretation["summary"] = (
            f"The model shows {robustness_level} robustness against adversarial attacks. "
            f"It is most vulnerable to {', '.join(interpretation['most_vulnerable_areas'])} attacks "
            f"and least vulnerable to {', '.join(interpretation['least_vulnerable_areas'])} attacks."
        )
        
        return interpretation
    
    def _get_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            results: Test results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        interpretation = results.get("interpretation", {})
        
        # General recommendation
        recommendations.append(
            "Regularly test model robustness using diverse adversarial attacks to identify vulnerabilities."
        )
        
        # Model-specific recommendations based on vulnerabilities
        most_vulnerable = interpretation.get("most_vulnerable_areas", [])
        
        if "TypoAttack" in most_vulnerable or "HomoglyphAttack" in most_vulnerable:
            recommendations.append(
                "Implement preprocessing to normalize text inputs by correcting common typos and replacing homoglyphs."
            )
            
        if "SynonymAttack" in most_vulnerable:
            recommendations.append(
                "Enhance training with adversarial examples that include synonym substitutions to improve semantic robustness."
            )
            
        if "WordScrambleAttack" in most_vulnerable or "SentenceShuffleAttack" in most_vulnerable:
            recommendations.append(
                "Train model with examples that have varied word orders to improve robustness to text rearrangement."
            )
            
        if "DistractorSentenceAttack" in most_vulnerable:
            recommendations.append(
                "Improve model's ability to identify and focus on relevant information by training with examples containing distractors."
            )
            
        if "ParaphraseAttack" in most_vulnerable:
            recommendations.append(
                "Train with paraphrased examples to improve semantic understanding and reduce reliance on specific phrasings."
            )
            
        if "JailbreakAttack" in most_vulnerable:
            recommendations.append(
                "Strengthen safety measures by implementing more robust content filtering and improve handling of potential jailbreak attempts."
            )
            
        # Additional recommendations based on overall robustness
        robustness_level = interpretation.get("robustness_level", "medium")
        
        if robustness_level == "low":
            recommendations.append(
                "Consider adversarial training techniques to systematically improve model robustness."
            )
            recommendations.append(
                "Implement server-side defenses like input sanitization and perturbation detection."
            )
            
        elif robustness_level == "medium":
            recommendations.append(
                "Focus on improving robustness in specific vulnerable areas while maintaining performance."
            )
        
        # Return unique recommendations
        return list(set(recommendations)) 