"""BigCodeBench test implementation for evaluating Python library usage."""
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
import re
from datetime import datetime

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

class BigCodeBenchTest(BaseNLPPerformanceTest):
    """Test class for BigCodeBench evaluation."""
    
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
        
        # Get API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.api_key = get_api_key("openai")
            if not self.api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide it in the config.")
        
        logger.info(f"Initialized BigCodeBench Test with epochs={self.epochs}, "
                   f"temperature={self.temperature}, num_test_cases={self.num_test_cases}")
    
    async def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the BigCodeBench dataset with Python library usage tasks."""
        try:
            # In a real implementation, we would load from HuggingFace
            # For now, we'll use a sample dataset
            return [
                # NumPy tasks
                {
                    "prompt": "Write a function to create a 3x3 identity matrix using NumPy.",
                    "library": "numpy",
                    "category": "matrix_operations",
                    "test_cases": [
                        {"input": "", "output": "array([[1., 0., 0.],\n       [0., 1., 0.],\n       [0., 0., 1.]])"}
                    ]
                },
                {
                    "prompt": "Create a function that calculates the mean and standard deviation of a 1D array using NumPy.",
                    "library": "numpy",
                    "category": "statistics",
                    "test_cases": [
                        {"input": "array([1, 2, 3, 4, 5])", "output": "(3.0, 1.4142135623730951)"}
                    ]
                },
                # Pandas tasks
                {
                    "prompt": "Write a function to read a CSV file and return the first 5 rows using Pandas.",
                    "library": "pandas",
                    "category": "data_io",
                    "test_cases": [
                        {"input": "'sample.csv'", "output": "DataFrame with 5 rows"}
                    ]
                },
                {
                    "prompt": "Create a function to group data by a column and calculate the mean of numeric columns using Pandas.",
                    "library": "pandas",
                    "category": "data_analysis",
                    "test_cases": [
                        {"input": "df.groupby('category')", "output": "Grouped DataFrame with means"}
                    ]
                },
                # Matplotlib tasks
                {
                    "prompt": "Write a function to create a line plot of x and y data using Matplotlib.",
                    "library": "matplotlib",
                    "category": "visualization",
                    "test_cases": [
                        {"input": "x=[1,2,3], y=[4,5,6]", "output": "Line plot figure"}
                    ]
                },
                {
                    "prompt": "Create a function to generate a scatter plot with different colors for each category using Matplotlib.",
                    "library": "matplotlib",
                    "category": "visualization",
                    "test_cases": [
                        {"input": "x=[1,2,3], y=[4,5,6], c=['r','g','b']", "output": "Scatter plot figure"}
                    ]
                }
            ]
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _get_judge_prompt_template(self) -> str:
        """Get the judge prompt template for evaluating code solutions."""
        return """You are a Python library expert. Please evaluate the following code solution.

Problem: {{PROBLEM}}
Library: {{LIBRARY}}
Solution: {{SOLUTION}}
Test Cases: {{TEST_CASES}}

Please provide:
1. Whether the solution correctly uses the specified library
2. A score from 0-1 where:
   - 0: Solution doesn't use the library or has major issues
   - 0.5: Solution partially uses the library or has minor issues
   - 1: Solution correctly uses the library and follows best practices
3. A brief explanation of your evaluation

Format your response as:
## uses_library: [yes/no]
## the_score: [score]
## explanation: [your explanation]"""
    
    async def _evaluate_solution(self, problem: str, library: str, solution: str, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate a code solution using the judge model."""
        try:
            # Get judge prompt template
            judge_prompt_template = self._get_judge_prompt_template()
            judge_prompt = judge_prompt_template.replace("{{PROBLEM}}", problem).replace("{{LIBRARY}}", library).replace("{{SOLUTION}}", solution).replace("{{TEST_CASES}}", json.dumps(test_cases, indent=2))
            
            # Generate evaluation from judge model
            eval_response = await self.model.generate(
                prompt=judge_prompt,
                temperature=0.1,  # Lower temperature for more consistent evaluation
                max_tokens=min(512, self.max_tokens)
            )
            
            # Parse evaluation response
            pattern = re.compile(r"##\s*uses_library\s*:(.*)##\s*the_score\s*:(.*)##\s*explanation\s*:(.*)", re.DOTALL)
            eval_parts = pattern.search(eval_response)
            
            if not eval_parts:
                raise ValueError(f"Malformed evaluation response: {eval_response}")
            
            uses_library = eval_parts[1].strip().lower() == "yes"
            score = float(eval_parts[2].strip())
            explanation = eval_parts[3].strip()
            
            return {
                "success": True,
                "uses_library": uses_library,
                "score": score,
                "explanation": explanation,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            return {
                "success": False,
                "uses_library": False,
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
            category_scores = {}
            
            # Process each test case
            for epoch in range(self.epochs):
                for idx, test_case in enumerate(test_cases):
                    try:
                        # Log test case details
                        logger.info(f"\nProcessing test case {idx + 1}/{len(test_cases)} (Epoch {epoch + 1}/{self.epochs})")
                        logger.info(f"Library: {test_case['library']}")
                        logger.info(f"Category: {test_case['category']}")
                        logger.info(f"Problem: {test_case['prompt']}")
                        
                        # Generate model solution
                        logger.info("Generating model solution...")
                        solution = await self.model.generate(
                            prompt=test_case["prompt"],
                            temperature=self.temperature,
                            max_tokens=min(2048, self.max_tokens)
                        )
                        logger.info(f"Model solution: {solution}")
                        
                        # Evaluate solution
                        logger.info("Evaluating solution...")
                        eval_result = await self._evaluate_solution(
                            problem=test_case["prompt"],
                            library=test_case["library"],
                            solution=solution,
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
                        
                        # Store result
                        result = {
                            "test_case_id": idx,
                            "epoch": epoch,
                            "library": test_case["library"],
                            "category": test_case["category"],
                            "problem": test_case["prompt"],
                            "model_solution": solution,
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
            
            # Create detailed results
            final_result = {
                "test_run_id": self.test_id,
                "score": average_score,
                "analysis": {
                    "total_cases": total_cases,
                    "library_scores": library_averages,
                    "category_scores": category_averages,
                    "libraries": {
                        library: len(scores)
                        for library, scores in library_scores.items()
                    },
                    "categories": {
                        category: len(scores)
                        for category, scores in category_scores.items()
                    }
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in test implementation: {str(e)}")
            raise 