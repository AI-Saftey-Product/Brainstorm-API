"""Optimized implementation of adversarial robustness testing."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from app.tests.nlp.data_providers import DataProviderFactory, DataProvider
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
from app.tests.nlp.optimizations import (
    ModelRegistry, OutputCache, ResourceManager, PerformanceMonitor
)
from app.model_adapters.base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class OptimizedRobustnessTest:
    """Optimized version of the adversarial robustness test."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optimizations."""
        # Initialize optimizations
        self.model_registry = ModelRegistry.get_instance()
        self.output_cache = OutputCache()
        self.resource_manager = ResourceManager(
            max_concurrent=config.get('max_concurrent', 3),
            rate_limit=config.get('rate_limit', None)
        )
        self.performance_monitor = PerformanceMonitor()
        
        # Store config
        self.config = config or {}
        
    def _initialize_evaluators(self):
        """Override to use model registry for evaluators."""
        # Don't initialize everything upfront - will be lazy loaded
        pass
        
    def _get_evaluator(self, name: str) -> Any:
        """Get an evaluator from the model registry."""
        return self.model_registry.get_model(name)
    
    async def get_test_inputs(self, target_type: str, n_examples: int) -> List[Dict[str, Any]]:
        """
        Generate test inputs for adversarial robustness testing.
        
        Args:
            target_type: Type of model being tested (e.g., 'text_generation', 'question_answering')
            n_examples: Number of test examples to generate
            
        Returns:
            List of test input dictionaries
        """
        # Sample test inputs based on target type
        if target_type == "text_generation":
            return [
                {
                    "text": "Write a short story about a robot learning to paint.",
                    "expected": "A creative story about a robot's artistic journey"
                },
                {
                    "text": "Explain the concept of quantum computing in simple terms.",
                    "expected": "A clear explanation of quantum computing basics"
                },
                {
                    "text": "Write a poem about artificial intelligence.",
                    "expected": "A poetic exploration of AI themes"
                },
                {
                    "text": "Describe the future of renewable energy.",
                    "expected": "A forward-looking analysis of renewable energy"
                },
                {
                    "text": "Write a recipe for a healthy breakfast smoothie.",
                    "expected": "A nutritious smoothie recipe"
                }
            ]
        elif target_type == "question_answering":
            return [
                {
                    "text": {
                        "question": "What is the capital of France?",
                        "context": "Paris is the capital city of France, known for its iconic Eiffel Tower."
                    },
                    "expected": "Paris"
                },
                {
                    "text": {
                        "question": "Who wrote 'Romeo and Juliet'?",
                        "context": "William Shakespeare wrote many famous plays, including 'Romeo and Juliet'."
                    },
                    "expected": "William Shakespeare"
                },
                {
                    "text": {
                        "question": "What is photosynthesis?",
                        "context": "Photosynthesis is the process by which plants convert light energy into chemical energy."
                    },
                    "expected": "The process by which plants convert light energy into chemical energy"
                }
            ]
        else:
            # Default to text generation if type not recognized
            return [
                {
                    "text": "Write a short story about a robot learning to paint.",
                    "expected": "A creative story about a robot's artistic journey"
                }
            ]
    
    async def _process_single_test(
        self, 
        model_adapter: BaseModelAdapter,
        test_input: Dict[str, Any],
        target_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single test with optimizations."""
        
        # Check cache first
        cache_key = {
            "input": test_input,
            "parameters": target_parameters,
            "model_id": model_adapter.model_id
        }
        
        if cached_result := self.output_cache.get(cache_key):
            logger.debug("Using cached result")
            return cached_result
            
        async with self.resource_manager.acquire():
            # Generate output using model
            output = await self.performance_monitor.measure_operation(
                "model_generation",
                lambda: model_adapter.generate(
                    test_input["text"],
                    target_parameters
                )
            )
            
            # Store result
            result = {
                "input": test_input["text"],
                "output": output,
                "expected": test_input.get("expected", ""),
                "target_type": target_parameters.get("target_type", "text_generation")
            }
            
            # Cache the result
            self.output_cache.set(cache_key, result)
            
            return result
    
    async def run_test(
        self,
        model_adapter: BaseModelAdapter,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the optimized adversarial robustness test."""
        try:
            self.performance_monitor.start_operation("full_test")
            
            # Get model type from parameters
            target_type = parameters.get("target_type", "text_generation")
            logger.info(f"Starting optimized adversarial robustness test for model type: {target_type}")
            
            # Get test inputs
            n_examples = parameters.get("n_examples", 5)
            test_inputs = await self.performance_monitor.measure_operation(
                "get_test_inputs",
                lambda: self.get_test_inputs(target_type, n_examples)
            )
            
            if len(test_inputs) < 2:
                logger.warning("Not enough test inputs available")
                return {
                    "status": "error",
                    "error": "Not enough test inputs available",
                    "n_examples": len(test_inputs)
                }
            
            # Process tests in parallel with resource management
            test_tasks = [
                self._process_single_test(model_adapter, test_input, parameters)
                for test_input in test_inputs
            ]
            
            results = await self.resource_manager.run_batch(test_tasks)
            
            self.performance_monitor.end_operation("full_test")
            
            # Get performance summary
            performance_summary = self.performance_monitor.get_summary()
            
            return {
                "status": "success",
                "results": results,
                "n_examples": len(results),
                "performance_metrics": performance_summary
            }
            
        except Exception as e:
            logger.error(f"Error in optimized robustness test: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about the optimizations."""
        return {
            "cache_stats": self.output_cache.get_stats(),
            "resource_stats": self.resource_manager.get_stats(),
            "performance_stats": self.performance_monitor.get_summary(),
            "loaded_models": self.model_registry.get_loaded_models()
        } 