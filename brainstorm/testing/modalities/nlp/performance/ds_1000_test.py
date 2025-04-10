"""DS-1000 test implementation for evaluating data science code generation capabilities."""
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
DEFAULT_EPOCHS = 4
DEFAULT_TEMPERATURE = 0.75
MAX_TOKENS = 2048
DEFAULT_TEST_CASES = 100
DATASET_HF_REPO_ID = "xlangai/DS-1000"

DEFAULT_SYSTEM_MESSAGE = (
    "Write a short code following the given format and indentation. "
    "Place the executable code between <code> and </code> tags, without any other non-executable things"
)

MODEL_SPECIFIC_SYSTEM_MESSAGES = {
    "openai/gpt-4.*": "Only provide the code completion needed. Don't repeat the context code."
}

class DS1000Test(BaseNLPPerformanceTest):
    """Test class for DS-1000 benchmark evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.epochs = config.get("epochs", DEFAULT_EPOCHS)
        self.temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens = config.get("max_tokens", MAX_TOKENS)
        self.num_test_cases = config.get("num_test_cases", DEFAULT_TEST_CASES)
        self.challenges = config.get("challenges", None)
        self.shuffle = config.get("shuffle", True)  # Default to True to match Inspect AI
        self.max_concurrent = config.get("max_concurrent", 3)
        self.library = config.get("library", None)  # Optional library filter
        self.perturbation = config.get("perturbation", None)  # Optional perturbation filter
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized DS-1000 Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}")
    
    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the DS-1000 dataset from HuggingFace."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(DATASET_HF_REPO_ID, split="test")
            
            # Convert to list of samples
            tasks = []
            for record in dataset:
                # Apply filters if specified
                if self.library and record["metadata"]["library"] != self.library:
                    continue
                if self.perturbation and record["metadata"]["perturbation_type"] != self.perturbation:
                    continue
                
                # Convert to our format
                task = {
                    "task_id": record["metadata"]["problem_id"],
                    "library": record["metadata"]["library"],
                    "category": record["metadata"]["perturbation_type"],
                    "task": record["prompt"],
                    "description": record["code_context"],
                    "test_cases": [{
                        "input": record["code_context"],
                        "output": record["reference_code"]
                    }]
                }
                tasks.append(task)
            
            # Shuffle tasks if configured
            if self.shuffle:
                import random
                random.shuffle(tasks)
            
            # Limit number of tasks if specified
            if self.num_test_cases:
                tasks = tasks[:self.num_test_cases]
            
            logger.info(f"Loaded {len(tasks)} tasks from DS-1000 dataset")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _get_system_message(self) -> str:
        """Get the appropriate system message based on the model."""
        model_name = str(self.model)
        sys_message = DEFAULT_SYSTEM_MESSAGE
        for pattern, message in MODEL_SPECIFIC_SYSTEM_MESSAGES.items():
            if re.match(pattern, model_name):
                sys_message = message
        return sys_message
    
    def _postprocess_code(self, code: str) -> str:
        """Clean up generated code following the paper's repository."""
        code = self._extract_from_tags(code, "```python\n", "\n```")
        code = self._extract_from_tags(code, "```\n", "\n```")
        code = self._extract_from_tags(code, "<code>", "</code>")
        code = self._extract_from_tags(code, "", "</code>")
        code = self._extract_from_tags(code, "", "\nEND SOLUTION")
        code = self._extract_from_tags(code, "", "\n### END SOLUTION")
        return code
    
    def _extract_from_tags(self, text: str, start_tag: str, end_tag: str) -> str:
        """Extract code from between first occurrence of optional tags."""
        start_index = len(start_tag) if text.startswith(start_tag) else 0
        end_index = text.find(end_tag, len(start_tag))
        end_index = end_index if end_index != -1 else len(text)
        return text[start_index:end_index]
    
    async def _evaluate_implementation(self, library: str, category: str, task: str, description: str, implementation: str, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate an implementation using the judge model."""
        try:
            # Clean up the implementation
            implementation = self._postprocess_code(implementation)
            
            # Create test program
            test_program = (
                f"{description}\n"
                f"solution={repr(implementation)}\n"
                "test_execution(solution)\n"
                + ("test_string(solution)\n" if "test_string(" in description else "")
            )
            
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
                    timeout=180
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
        """Run the DS-1000 test implementation."""
        try:
            # Load dataset
            test_cases = await self._load_dataset()
            
            # Track results
            results = []
            total_score = 0
            library_scores = {}
            category_scores = {}
            task_scores = {}
            
            # Get system message
            system_message = self._get_system_message()
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Log test case details
                        logger.info(f"\nProcessing test case {idx + 1}/{len(test_cases)} (Epoch {epoch + 1}/{self.epochs})")
                        logger.info(f"Task ID: {test_case['task_id']}")
                        logger.info(f"Library: {test_case['library']}")
                        logger.info(f"Category: {test_case['category']}")
                        logger.info(f"Task: {test_case['task']}")
                        
                        # Generate model implementation
                        logger.info("Generating model implementation...")
                        implementation = await self.model.generate(
                            prompt=f"{system_message}\n\n{test_case['task']}",
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        logger.info(f"Model implementation: {implementation}")
                        
                        # Evaluate implementation
                        logger.info("Evaluating implementation...")
                        eval_result = await self._evaluate_implementation(
                            library=test_case["library"],
                            category=test_case["category"],
                            task=test_case["task"],
                            description=test_case["description"],
                            implementation=implementation,
                            test_cases=test_case["test_cases"]
                        )
                        
                        # Update scores
                        score = eval_result["score"]
                        total_score += score
                        
                        # Update library scores
                        library = test_case["library"]
                        if library not in library_scores:
                            library_scores[library] = []
                        library_scores[library].append(score)
                        
                        # Update category scores
                        category = test_case["category"]
                        if category not in category_scores:
                            category_scores[category] = []
                        category_scores[category].append(score)
                        
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
                            "library": test_case["library"],
                            "category": test_case["category"],
                            "task": test_case["task"],
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
            
            # Calculate library averages
            library_averages = {
                library: sum(scores) / len(scores)
                for library, scores in library_scores.items()
            }
            
            # Calculate category averages
            category_averages = {
                category: sum(scores) / len(scores)
                for category, scores in category_scores.items()
            }
            
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
                    "library_scores": library_averages,
                    "category_scores": category_averages,
                    "task_scores": task_averages,
                    "libraries": {
                        library: len(scores)
                        for library, scores in library_scores.items()
                    },
                    "categories": {
                        category: len(scores)
                        for category, scores in category_scores.items()
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