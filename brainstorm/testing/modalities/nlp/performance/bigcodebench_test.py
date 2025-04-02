"""BigCodeBench test implementation for evaluating code generation capabilities."""
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
import re
from datetime import datetime
import os
from pathlib import Path
from datasets import load_dataset

# Import from standard location with fallback
try:
    from brainstorm.testing.modalities.nlp.performance.base_test import BaseNLPPerformanceTest
except ImportError:
    from brainstorm.testing.base import BasePerformanceTest as BaseNLPPerformanceTest

from brainstorm.config.api_keys import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EPOCHS = 1
DEFAULT_TEMPERATURE = 0.0
MAX_TOKENS = 2048
DEFAULT_TEST_CASES = 100
VERIFY_TIMEOUT = 30

INSTRUCTION_PROMPT = """Read the following function signature and docstring, and fully implement the function described. Make sure to include the import statements and use the same variable names as stated in the header."""

class BigCodeBenchTest(BaseNLPPerformanceTest):
    """Test class for BigCodeBench benchmark evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.epochs = config.get("epochs", DEFAULT_EPOCHS)
        self.temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens = config.get("max_tokens", MAX_TOKENS)
        self.num_test_cases = config.get("num_test_cases", DEFAULT_TEST_CASES)
        self.challenges = config.get("challenges", None)
        self.shuffle = config.get("shuffle", False)
        self.max_concurrent = config.get("max_concurrent", 3)
        self.version = config.get("version", "v0.1.2")
        self.subset = config.get("subset", None)  # Optional subset of tasks to evaluate
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized BigCodeBench Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}, "
                   f"version={self.version}")
    
    async def run(self, model_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the BigCodeBench test.
        
        Args:
            model_parameters: Optional model parameters to override defaults
            
        Returns:
            Dict containing test results
        """
        return await self._run_test_implementation(model_parameters or {})

    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the BigCodeBench dataset from HuggingFace."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("bigcode/bigcodebench", split=self.version)
            
            # Convert to list of samples
            tasks = []
            for idx, record in enumerate(dataset):
                # Apply subset filter if specified
                if self.subset and idx not in self.subset:
                    continue
                
                # Convert to our format
                task = {
                    "task_id": record["task_id"],
                    "complete_prompt": record["complete_prompt"],
                    "canonical_solution": record["canonical_solution"],
                    "test": record["test"],
                    "libs": record["libs"]
                }
                tasks.append(task)
            
            # Shuffle tasks if configured
            if self.shuffle:
                import random
                random.shuffle(tasks)
            
            # Limit number of tasks if specified
            if self.num_test_cases:
                tasks = tasks[:self.num_test_cases]
            
            logger.info(f"Loaded {len(tasks)} tasks from BigCodeBench dataset")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _find_code(self, completion: str) -> str:
        """Extract code from completion."""
        pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
        pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern_1.findall(completion) + pattern_2.findall(completion)
        if matches:
            extracted_answer = matches[0]
        else:
            extracted_answer = completion
        return str(extracted_answer)
    
    async def _evaluate_implementation(self, implementation: str, test: str, libs: List[str]) -> Dict[str, Any]:
        """Evaluate an implementation using test cases."""
        try:
            # Clean up the implementation
            implementation = self._find_code(implementation)
            
            # Create test program
            code = [
                implementation,
                "\n",
                test,
                "\n",
                """
if __name__ == '__main__':
    unittest.main()"""
            ]
            
            # Write test file
            test_file = f"test_{self.test_id}.py"
            with open(test_file, "w") as f:
                f.write("".join(code))
            
            # Execute test
            try:
                import subprocess
                result = subprocess.run(
                    ["python", test_file],
                    capture_output=True,
                    text=True,
                    timeout=VERIFY_TIMEOUT
                )
                success = result.returncode == 0
                output = result.stdout
                error = result.stderr
            except subprocess.TimeoutError:
                success = False
                output = ""
                error = "Execution timed out"
            finally:
                # Clean up test file
                if os.path.exists(test_file):
                    os.remove(test_file)
            
            # Check for missing modules
            if not success and "ModuleNotFoundError" in error:
                logger.warning(
                    f"Missing module during execution for task {self.test_id}. "
                    f"Required libraries: {libs}\nError: {error}"
                )
            
            # Calculate score
            score = 1.0 if success else 0.0
            
            return {
                "success": success,
                "meets_requirements": success,
                "score": score,
                "explanation": f"Test output:\n{output}\nTest error:\n{error}",
                "error": error if not success else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating implementation: {str(e)}")
            return {
                "success": False,
                "meets_requirements": False,
                "score": 0,
                "explanation": None,
                "error": str(e)
            }
    
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the BigCodeBench test implementation."""
        try:
            # Load dataset
            test_cases = await self._load_dataset()
            
            # Track results
            results = []
            total_score = 0
            library_scores = {}
            task_scores = {}
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Create clear section boundaries for logging
                        logger.info(f"\n{'='*50}\nTEST CASE DETAILS ({idx + 1}/{len(test_cases)}, Epoch {epoch + 1}/{self.epochs})\n{'='*50}")
                        logger.info(f"Task ID: {test_case['task_id']}")
                        logger.info(f"Required libraries: {test_case['libs']}")
                        
                        # Log the full prompt
                        logger.info(f"\n{'='*50}\nINPUT PROMPT\n{'='*50}")
                        full_prompt = f"{INSTRUCTION_PROMPT}\n\n{test_case['complete_prompt']}"
                        logger.info(f"{full_prompt}")
                        logger.info(f"{'='*50}\n")
                        
                        # Generate model implementation
                        logger.info(f"Generating model implementation...")
                        implementation = await self.model.generate(
                            prompt=full_prompt,
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        
                        # Log the model output with clear section
                        logger.info(f"\n{'='*50}\nMODEL RESPONSE\n{'='*50}")
                        logger.info(f"{implementation}")
                        logger.info(f"{'='*50}\n")
                        
                        # Extract code from implementation for evaluation
                        extracted_code = self._find_code(implementation)
                        
                        # Evaluate implementation
                        logger.info(f"Evaluating implementation...")
                        eval_result = await self._evaluate_implementation(
                            implementation=implementation,
                            test=test_case["test"],
                            libs=test_case["libs"]
                        )
                        
                        # Log the evaluation results with clear section
                        logger.info(f"\n{'='*50}\nEVALUATION RESULTS\n{'='*50}")
                        logger.info(f"Success: {eval_result['success']}")
                        logger.info(f"Score: {eval_result['score']}")
                        logger.info(f"Explanation: {eval_result['explanation']}")
                        if eval_result['error']:
                            logger.info(f"Error: {eval_result['error']}")
                        logger.info(f"{'='*50}\n")
                        
                        # Update scores
                        score = eval_result["score"]
                        total_score += score
                        
                        # Update library scores
                        for lib in test_case["libs"]:
                            if lib not in library_scores:
                                library_scores[lib] = []
                            library_scores[lib].append(score)
                        
                        # Update task scores
                        task_id = test_case["task_id"]
                        if task_id not in task_scores:
                            task_scores[task_id] = []
                        task_scores[task_id].append(score)
                        
                        # Store result
                        result = {
                            "test_case_id": idx,
                            "epoch": epoch,
                            "task_id": test_case["task_id"],
                            "libs": test_case["libs"],
                            "implementation": implementation,
                            "evaluation_result": eval_result,
                            "score": score
                        }
                        results.append(result)
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing test case {idx}: {str(e)}")
                        continue
            
            # Calculate metrics
            total_cases = len(test_cases) * self.epochs
            average_score = total_score / total_cases
            
            # Calculate library averages
            library_averages = {
                library: sum(scores) / len(scores)
                for library, scores in library_scores.items()
            }
            
            # Calculate task averages
            task_averages = {
                task_id: sum(scores) / len(scores)
                for task_id, scores in task_scores.items()
            }
            
            # Log final summary with clear section
            logger.info(f"\n{'='*50}\nTEST SUMMARY\n{'='*50}")
            logger.info(f"Total Test Cases: {total_cases}")
            logger.info(f"Average Score: {average_score:.4f}")
            logger.info(f"Library Performance:")
            for lib, score in sorted(library_averages.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {lib}: {score:.4f} ({library_scores[lib].count(1.0)}/{len(library_scores[lib])} successful)")
            logger.info(f"{'='*50}\n")
            
            # Create detailed results
            final_result = {
                "test_run_id": self.test_id,
                "score": average_score,
                "analysis": {
                    "total_cases": total_cases,
                    "library_scores": library_averages,
                    "task_scores": task_averages,
                    "libraries": {
                        library: len(scores)
                        for library, scores in library_scores.items()
                    },
                    "tasks": {
                        task_id: len(scores)
                        for task_id, scores in task_scores.items()
                    }
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in test implementation: {str(e)}")
            raise 