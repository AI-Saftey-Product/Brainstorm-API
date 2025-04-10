"""APPS test implementation for evaluating code generation capabilities."""
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
MAX_TOKENS = 1500
DEFAULT_TEST_CASES = 100
VERIFY_TIMEOUT = 120

SYSTEM_PROMPT = """
You are an expert Python programmer. You will be given a task and a test cases. Be mindful of the test case format.
If the test case expects a string and you return an integer, you will fail the test case. There are additional test cases not provided that will be used to verify your solution. These are very specific, and your solution should account for edge cases. Write the complete Python function to solve the task. Do not add whitespace before or after the code block. Only use imports that are included in Python's standard library.

Before answering, reason in a step-by-step manner as to get the right answer. Then write the code block in the following format:

```python
[code block]
```

After writing the code block, please do not add any additional text or rationalization.

For instance:
        ## Question:
        Write a function that takes a string of space-separated integers and returns the sum of all numbers that are greater than 5.
Example:
Input: "1 6 2 7 3 8"
Output: 21 (because 6 + 7 + 8 = 21)

        ## Test Cases:
        ```python
assert solution("1 6 2 7 3 8") == 21
assert solution("1 2 3 4 5") == 0
assert solution("10 20 30") == 60
        ```

        ## Completion:
        Let me solve this step by step:

1. Split the input string into a list of integers
2. Filter numbers greater than 5
3. Sum the filtered numbers

```python
def solution(s):
    return sum(int(x) for x in s.split() if int(x) > 5)
```
""".strip()

class APPSTest(BaseNLPPerformanceTest):
    """Test class for APPS benchmark evaluation."""
    
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
        self.level = config.get("level", "interview")  # interview, competition, or introductory
        
        # Validate level input
        valid_levels = ["interview", "competition", "introductory"]
        if self.level not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        
        # Define problem ID ranges for each level
        level_ranges = {
            "interview": (0, 2999),
            "competition": (3000, 3999),
            "introductory": (4000, 5000),
        }
        self.start_id, self.end_id = level_ranges[self.level]
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized APPS Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}, "
                   f"level={self.level}")
    
    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the APPS dataset from HuggingFace."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("codeparrot/apps", split="test")
            
            # Convert to list of samples
            tasks = []
            for record in dataset:
                # Apply level filter
                problem_id = int(record["problem_id"])
                if not (self.start_id <= problem_id <= self.end_id):
                    continue
                
                # Convert to our format
                task = {
                    "task_id": record["problem_id"],
                    "level": record["level"],
                    "question": record["question"],
                    "test_cases": record["test_cases"],
                    "starter_code": record.get("starter_code", ""),
                    "solutions": record.get("solutions", [])
                }
                tasks.append(task)
            
            # Shuffle tasks if configured
            if self.shuffle:
                import random
                random.shuffle(tasks)
            
            # Limit number of tasks if specified
            if self.num_test_cases:
                tasks = tasks[:self.num_test_cases]
            
            logger.info(f"Loaded {len(tasks)} tasks from APPS dataset")
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
    
    async def _evaluate_implementation(self, question: str, implementation: str, test_cases: List[str]) -> Dict[str, Any]:
        """Evaluate an implementation using test cases."""
        try:
            # Clean up the implementation
            implementation = self._find_code(implementation)
            
            # Create test program
            test_program = f"{implementation}\n\n# Test cases\n"
            for test_case in test_cases:
                test_program += f"{test_case}\n"
            
            # Write test file
            test_file = f"test_{self.test_id}.py"
            with open(test_file, "w") as f:
                f.write(test_program)
            
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
        """Run the APPS test implementation."""
        try:
            # Load dataset
            test_cases = await self._load_dataset()
            
            # Track results
            results = []
            total_score = 0
            level_scores = {}
            problem_scores = {}
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Log test case details
                        logger.info(f"\nProcessing test case {idx + 1}/{len(test_cases)} (Epoch {epoch + 1}/{self.epochs})")
                        logger.info(f"Task ID: {test_case['task_id']}")
                        logger.info(f"Level: {test_case['level']}")
                        logger.info(f"Question: {test_case['question']}")
                        
                        # Generate model implementation
                        logger.info("Generating model implementation...")
                        implementation = await self.model.generate(
                            prompt=f"{SYSTEM_PROMPT}\n\n## Question:\n{test_case['question']}\n\n## Test Cases:\n```python\n{test_case['test_cases']}\n```\n\n## Completion:",
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        logger.info(f"Model implementation: {implementation}")
                        
                        # Evaluate implementation
                        logger.info("Evaluating implementation...")
                        eval_result = await self._evaluate_implementation(
                            question=test_case["question"],
                            implementation=implementation,
                            test_cases=test_case["test_cases"]
                        )
                        
                        # Update scores
                        score = eval_result["score"]
                        total_score += score
                        
                        # Update level scores
                        level = test_case["level"]
                        if level not in level_scores:
                            level_scores[level] = []
                        level_scores[level].append(score)
                        
                        # Update problem scores
                        problem_id = test_case["task_id"]
                        if problem_id not in problem_scores:
                            problem_scores[problem_id] = []
                        problem_scores[problem_id].append(score)
                        
                        # Store result
                        result = {
                            "test_case_id": idx,
                            "epoch": epoch,
                            "task_id": test_case["task_id"],
                            "level": test_case["level"],
                            "question": test_case["question"],
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
            
            # Calculate level averages
            level_averages = {
                level: sum(scores) / len(scores)
                for level, scores in level_scores.items()
            }
            
            # Calculate problem averages
            problem_averages = {
                problem_id: sum(scores) / len(scores)
                for problem_id, scores in problem_scores.items()
            }
            
            # Create detailed results
            final_result = {
                "test_run_id": self.test_id,
                "score": average_score,
                "analysis": {
                    "total_cases": total_cases,
                    "level_scores": level_averages,
                    "problem_scores": problem_averages,
                    "levels": {
                        level: len(scores)
                        for level, scores in level_scores.items()
                    },
                    "problems": {
                        problem_id: len(scores)
                        for problem_id, scores in problem_scores.items()
                    }
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in test implementation: {str(e)}")
            raise 