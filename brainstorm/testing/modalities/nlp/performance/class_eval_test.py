"""ClassEval test implementation for evaluating class-level Python code generation capabilities."""
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
import re
from datetime import datetime
import os
from pathlib import Path
from datasets import load_dataset

from brainstorm.testing.modalities.nlp.performance.base_test import BaseNLPPerformanceTest
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

INSTRUCTION = """
You are an expert Python programmer. You will be given a task, and the tests that your code must pass.
"""

class ClassEvalTest(BaseNLPPerformanceTest):
    """Test class for ClassEval benchmark evaluation."""
    
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
        self.few_shot = config.get("few_shot", 1)
        self.few_shot_seed = config.get("few_shot_seed", 42)
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized ClassEval Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}, "
                   f"few_shot={self.few_shot}, few_shot_seed={self.few_shot_seed}")
    
    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the ClassEval dataset from HuggingFace."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("FudanSELab/ClassEval", split="test")
            
            # Convert to list of samples
            tasks = []
            for record in dataset:
                # Convert to our format
                task = {
                    "task_id": record["task_id"],
                    "skeleton": record["skeleton"],
                    "test": record["test"],
                    "solution_code": record["solution_code"],
                    "import_statement": record["import_statement"],
                    "class_description": record["class_description"],
                    "methods_info": record["methods_info"],
                    "class_name": record["class_name"],
                    "test_classes": record["test_classes"],
                    "class_constructor": record["class_constructor"],
                    "fields": record["fields"]
                }
                tasks.append(task)
            
            # Shuffle tasks if configured
            if self.shuffle:
                import random
                random.seed(self.few_shot_seed)
                random.shuffle(tasks)
            
            # Limit number of tasks if specified
            if self.num_test_cases:
                tasks = tasks[:self.num_test_cases]
            
            logger.info(f"Loaded {len(tasks)} tasks from ClassEval dataset")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _find_code(self, completion: str) -> str:
        """Extract code from completion."""
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[0] if len(matches) >= 1 else completion
        return str(extracted_answer)
    
    def _construct_prompt(self, record: Dict[str, Any]) -> str:
        """Construct the prompt for a task."""
        prompt = f"""Task ID: {record['task_id']}

Class Description:
{record['class_description']}

Class Name: {record['class_name']}

Fields:
{record['fields']}

Constructor:
{record['class_constructor']}

Methods Information:
{record['methods_info']}

Skeleton Code:
{record['skeleton']}

Test Classes:
{record['test_classes']}

Please implement the class according to the description and test cases."""
        return prompt
    
    async def _evaluate_implementation(self, implementation: str, test: str) -> Dict[str, Any]:
        """Evaluate an implementation using test cases."""
        try:
            # Clean up the implementation
            implementation = self._find_code(implementation)
            
            # Create test program
            code = implementation + "\n" + test
            
            # Write test file
            test_file = f"test_{self.test_id}.py"
            with open(test_file, "w") as f:
                f.write(code)
            
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
        """Run the ClassEval test implementation."""
        try:
            # Load dataset
            test_cases = await self._load_dataset()
            
            # Track results
            results = []
            total_score = 0
            task_scores = {}
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Log test case details
                        logger.info(f"\nProcessing test case {idx + 1}/{len(test_cases)} (Epoch {epoch + 1}/{self.epochs})")
                        logger.info(f"Task ID: {test_case['task_id']}")
                        logger.info(f"Class Name: {test_case['class_name']}")
                        
                        # Generate model implementation
                        logger.info("Generating model implementation...")
                        implementation = await self.model.generate(
                            prompt=f"{INSTRUCTION}\n\n{self._construct_prompt(test_case)}",
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        logger.info(f"Model implementation: {implementation}")
                        
                        # Evaluate implementation
                        logger.info("Evaluating implementation...")
                        eval_result = await self._evaluate_implementation(
                            implementation=implementation,
                            test=test_case["test"]
                        )
                        
                        # Update scores
                        score = eval_result["score"]
                        total_score += score
                        
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
                            "class_name": test_case["class_name"],
                            "implementation": implementation,
                            "evaluation_result": eval_result,
                            "score": score
                        }
                        results.append(result)
                        
                        # Log result
                        logger.info(f"Test case result: Score = {score}")
                        if eval_result["error"]:
                            logger.error(f"Error: {eval_result['error']}")
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing test case {idx}: {str(e)}")
                        continue
            
            # Calculate metrics
            total_cases = len(test_cases) * self.epochs
            average_score = total_score / total_cases
            
            # Calculate task averages
            task_averages = {
                task_id: sum(scores) / len(scores)
                for task_id, scores in task_scores.items()
            }
            
            # Create detailed results
            final_result = {
                "test_run_id": self.test_id,
                "score": average_score,
                "analysis": {
                    "total_cases": total_cases,
                    "task_scores": task_averages,
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