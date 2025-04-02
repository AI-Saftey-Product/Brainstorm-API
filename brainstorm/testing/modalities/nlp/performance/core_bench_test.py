"""CORE-Bench test implementation for evaluating code generation capabilities."""
import logging
from typing import Dict, Any, List, Optional, Literal
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
VERIFY_TIMEOUT = 180

AGENT_SYSTEM_PROMPT = """
You are a seasoned digital assistant: capable, intelligent, considerate, and assertive. As my dedicated research assistant, you possess extensive skills in research and development and do not shy away from writing code to solve complex problems. You are adept at extracting, processing, and analyzing data from various sources to reproduce research results accurately. Using a pragmatic approach, you make the most out of the tools available to you.
"""

class COREBenchTest(BaseNLPPerformanceTest):
    """Test class for CORE-Bench benchmark evaluation."""
    
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
        self.difficulty = config.get("difficulty", "easy")  # easy, medium, hard
        self.field = config.get("field", "all")  # Computer Science, Medical Sciences, Social Sciences, all
        self.language = config.get("language", "all")  # Python, R, all
        self.capsule_ids = config.get("capsule_ids", None)  # Optional list of capsule IDs to include
        self.exclude_capsule_ids = config.get("exclude_capsule_ids", None)  # Optional list of capsule IDs to exclude
        self.filter_out_gpu = config.get("filter_out_gpu", False)  # Whether to exclude GPU-required capsules
        self.vllm_model = config.get("vllm_model", "gpt-4o-mini")  # Vision language model to use
        self.max_retries = config.get("max_retries", 5)  # Maximum number of download retries
        self.backoff_factor = config.get("backoff_factor", 1)  # Backoff factor for retries
        self.max_messages = config.get("max_messages", 30)  # Maximum number of messages
        self.token_limit = config.get("token_limit", None)  # Maximum number of tokens
        
        # Validate difficulty input
        valid_difficulties = ["easy", "medium", "hard"]
        if self.difficulty not in valid_difficulties:
            raise ValueError(f"difficulty must be one of {valid_difficulties}")
        
        # Validate field input
        valid_fields = ["Computer Science", "Medical Sciences", "Social Sciences", "all"]
        if self.field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}")
        
        # Validate language input
        valid_languages = ["Python", "R", "all"]
        if self.language not in valid_languages:
            raise ValueError(f"language must be one of {valid_languages}")
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized CORE-Bench Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}, "
                   f"difficulty={self.difficulty}, field={self.field}, language={self.language}")
    
    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the CORE-Bench dataset."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("core-bench/core-bench", split="test")
            
            # Convert to list of samples
            tasks = []
            for record in dataset:
                # Apply filters
                if self.difficulty != "all" and record["difficulty"] != self.difficulty:
                    continue
                if self.field != "all" and record["field"] != self.field:
                    continue
                if self.language != "all" and record["language"] != self.language:
                    continue
                if self.capsule_ids and record["capsule_id"] not in self.capsule_ids:
                    continue
                if self.exclude_capsule_ids and record["capsule_id"] in self.exclude_capsule_ids:
                    continue
                if self.filter_out_gpu and record.get("requires_gpu", False):
                    continue
                
                # Convert to our format
                task = {
                    "capsule_id": record["capsule_id"],
                    "difficulty": record["difficulty"],
                    "field": record["field"],
                    "language": record["language"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "requires_gpu": record.get("requires_gpu", False),
                    "dependencies": record.get("dependencies", [])
                }
                tasks.append(task)
            
            # Shuffle tasks if configured
            if self.shuffle:
                import random
                random.shuffle(tasks)
            
            # Limit number of tasks if specified
            if self.num_test_cases:
                tasks = tasks[:self.num_test_cases]
            
            logger.info(f"Loaded {len(tasks)} tasks from CORE-Bench dataset")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    async def _evaluate_implementation(self, question: str, implementation: str, answer: str) -> Dict[str, Any]:
        """Evaluate an implementation using the answer."""
        try:
            # Create test program
            test_program = f"{implementation}\n\n# Test answer\nassert result == {repr(answer)}"
            
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
        """Run the CORE-Bench test implementation."""
        try:
            # Load dataset
            test_cases = await self._load_dataset()
            
            # Track results
            results = []
            total_score = 0
            difficulty_scores = {}
            field_scores = {}
            language_scores = {}
            capsule_scores = {}
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Log test case details
                        logger.info(f"\nProcessing test case {idx + 1}/{len(test_cases)} (Epoch {epoch + 1}/{self.epochs})")
                        logger.info(f"Capsule ID: {test_case['capsule_id']}")
                        logger.info(f"Difficulty: {test_case['difficulty']}")
                        logger.info(f"Field: {test_case['field']}")
                        logger.info(f"Language: {test_case['language']}")
                        
                        # Generate model implementation
                        logger.info("Generating model implementation...")
                        implementation = await self.model.generate(
                            prompt=f"{AGENT_SYSTEM_PROMPT}\n\nQuestion: {test_case['question']}",
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        logger.info(f"Model implementation: {implementation}")
                        
                        # Evaluate implementation
                        logger.info("Evaluating implementation...")
                        eval_result = await self._evaluate_implementation(
                            question=test_case["question"],
                            implementation=implementation,
                            answer=test_case["answer"]
                        )
                        
                        # Update scores
                        score = eval_result["score"]
                        total_score += score
                        
                        # Update difficulty scores
                        difficulty = test_case["difficulty"]
                        if difficulty not in difficulty_scores:
                            difficulty_scores[difficulty] = []
                        difficulty_scores[difficulty].append(score)
                        
                        # Update field scores
                        field = test_case["field"]
                        if field not in field_scores:
                            field_scores[field] = []
                        field_scores[field].append(score)
                        
                        # Update language scores
                        language = test_case["language"]
                        if language not in language_scores:
                            language_scores[language] = []
                        language_scores[language].append(score)
                        
                        # Update capsule scores
                        capsule_id = test_case["capsule_id"]
                        if capsule_id not in capsule_scores:
                            capsule_scores[capsule_id] = []
                        capsule_scores[capsule_id].append(score)
                        
                        # Store result
                        result = {
                            "test_case_id": idx,
                            "epoch": epoch,
                            "capsule_id": test_case["capsule_id"],
                            "difficulty": test_case["difficulty"],
                            "field": test_case["field"],
                            "language": test_case["language"],
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
            
            # Calculate difficulty averages
            difficulty_averages = {
                difficulty: sum(scores) / len(scores)
                for difficulty, scores in difficulty_scores.items()
            }
            
            # Calculate field averages
            field_averages = {
                field: sum(scores) / len(scores)
                for field, scores in field_scores.items()
            }
            
            # Calculate language averages
            language_averages = {
                language: sum(scores) / len(scores)
                for language, scores in language_scores.items()
            }
            
            # Calculate capsule averages
            capsule_averages = {
                capsule_id: sum(scores) / len(scores)
                for capsule_id, scores in capsule_scores.items()
            }
            
            # Create detailed results
            final_result = {
                "test_run_id": self.test_id,
                "score": average_score,
                "analysis": {
                    "total_cases": total_cases,
                    "difficulty_scores": difficulty_averages,
                    "field_scores": field_averages,
                    "language_scores": language_averages,
                    "capsule_scores": capsule_averages,
                    "difficulties": {
                        difficulty: len(scores)
                        for difficulty, scores in difficulty_scores.items()
                    },
                    "fields": {
                        field: len(scores)
                        for field, scores in field_scores.items()
                    },
                    "languages": {
                        language: len(scores)
                        for language, scores in language_scores.items()
                    },
                    "capsules": {
                        capsule_id: len(scores)
                        for capsule_id, scores in capsule_scores.items()
                    }
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in test implementation: {str(e)}")
            raise 