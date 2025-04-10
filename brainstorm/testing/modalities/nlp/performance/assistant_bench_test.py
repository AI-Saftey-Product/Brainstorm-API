"""AssistantBench test implementation for evaluating web agent capabilities."""
import json
import os
import re
import ast
import math
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import requests
from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest

@dataclass
class AssistantBenchProblem:
    """Represents a single AssistantBench problem."""
    problem_id: str
    task: str
    answer: Union[str, int, float, List[Any], Dict[str, Any]]
    explanation: str
    difficulty: str

class AssistantBenchTest(BasePerformanceTest):
    """Test for evaluating web agent capabilities on real-world tasks."""
    
    INITIAL_PROMPT = dedent("""
You are a helpful AI assistant that can perform real-world tasks on the web.
Your goal is to help users with their tasks by searching for information, analyzing data, and providing accurate answers.

For each task:
1. Understand the task requirements
2. Search for relevant information if needed
3. Analyze the information to find the answer
4. Provide a clear and concise answer

Your answer should be:
- Accurate and well-supported
- In the correct format (number, string, list, or JSON)
- Based on reliable sources
- Accompanied by a brief explanation of your reasoning

[BEGIN TASK]
{task}
[END TASK]
""").strip()
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AssistantBench test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - difficulty: Optional difficulty level to filter problems
                - token_limit: Maximum tokens allowed for the entire task
                - web_search_model: Optional model to use for web search
        """
        super().__init__(config)
        self.difficulty = config.get("difficulty")
        self.token_limit = config.get("token_limit", 500000)
        self.web_search_model = config.get("web_search_model")
        self.problems: List[AssistantBenchProblem] = []
        
    def load_dataset(self) -> None:
        """Load the AssistantBench dataset."""
        # Get the data directory path
        data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "assistant_bench"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download problems if not exists
        problems_path = data_dir / "problems.json"
        if not problems_path.exists():
            self._download_problems(data_dir)
            
        # Load and parse problems
        self.problems = []
        with open(problems_path, "r") as f:
            records = json.load(f)
            for record in records:
                # Skip if difficulty doesn't match
                if self.difficulty and record["difficulty"] != self.difficulty:
                    continue
                    
                problem = AssistantBenchProblem(
                    problem_id=record["id"],
                    task=record["task"],
                    answer=record["answer"],
                    explanation=record["explanation"],
                    difficulty=record["difficulty"]
                )
                self.problems.append(problem)
                
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.problems)
            
    def _download_problems(self, data_dir: Path) -> None:
        """Download the AssistantBench problems.
        
        Args:
            data_dir: Directory to save the problems
        """
        # Download from HuggingFace dataset
        dataset_url = "https://huggingface.co/datasets/AssistantBench/AssistantBench/raw/main/validation.json"
        problems_path = data_dir / "problems.json"
        
        print(f"Downloading {dataset_url}")
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with open(problems_path, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download dataset from {dataset_url} - {response.status_code}")
            
    def _extract_number(self, s: Union[str, int, float]) -> float:
        """Extract a number from a string, handling various formats.
        
        Args:
            s: Input string, integer, or float
            
        Returns:
            Extracted number as a float
        """
        if isinstance(s, (int, float)):
            return float(s)
            
        # Remove thousand separators and convert decimal separators
        s = s.replace(",", "").replace(" ", "")
        
        # Try to find a number pattern
        match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
        
        if match:
            return float(match.group())
            
        raise ValueError(f"Could not extract number from '{s}'")
        
    def _parse_numeric_values(self, data: Any) -> Any:
        """Recursively parse numeric values in dictionaries and lists.
        
        Args:
            data: Input data to parse
            
        Returns:
            Parsed data with numeric values
        """
        if isinstance(data, dict):
            return {k: self._parse_numeric_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._parse_numeric_values(item) for item in data]
        elif isinstance(data, str):
            try:
                return self._extract_number(data)
            except ValueError:
                return data
        return data
        
    def _parse_and_classify(self, input_data: str) -> Any:
        """Parse the input string and classify its type.
        
        Args:
            input_data: Input string to parse
            
        Returns:
            Parsed and classified data
        """
        # Strip whitespace
        input_data = input_data.strip()
        
        # Try to parse as JSON first
        try:
            parsed_data = json.loads(input_data)
            return self._parse_numeric_values(parsed_data)
        except json.JSONDecodeError:
            pass
            
        # Try to evaluate as a Python literal
        try:
            parsed_data = ast.literal_eval(input_data)
            return self._parse_numeric_values(parsed_data)
        except (ValueError, SyntaxError):
            pass
            
        # If it's a multiline string, try to parse each line as JSON
        if "\n" in input_data:
            try:
                parsed_data = [json.loads(line.strip()) for line in input_data.splitlines()]
                return self._parse_numeric_values(parsed_data)
            except json.JSONDecodeError:
                return [line.strip() for line in input_data.splitlines()]
                
        # Check if it's a number
        try:
            return self._extract_number(input_data)
        except ValueError:
            pass
            
        # If all else fails, return as a string
        return input_data
        
    def _calculate_number_score(self, pred: float, gold: float) -> float:
        """Calculate score for numeric predictions.
        
        Args:
            pred: Predicted number
            gold: Gold standard number
            
        Returns:
            Score between 0 and 1
        """
        if pred == gold:
            return 1.0
            
        # If both are negative, just make them positive
        if pred < 0 and gold < 0:
            pred, gold = abs(pred), abs(gold)
        # If one is negative, add the absolute difference to both
        elif pred < 0 or gold < 0:
            diff = abs(abs(pred) - abs(gold))
            pred += diff
            gold += diff
            
        # Calculate the ratio
        ratio = max(pred, gold) / min(pred, gold) if min(pred, gold) > 0 else float("inf")
        
        # Apply the order magnitude metric
        score = max(0, 1 - math.log10(ratio))
        
        return score
        
    def _calculate_json_score(self, pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
        """Calculate score for JSON (dictionary) predictions.
        
        Args:
            pred: Predicted dictionary
            gold: Gold standard dictionary
            
        Returns:
            Score between 0 and 1
        """
        if len(pred) == 0 and len(gold) == 0:
            return 1
            
        all_keys = set(pred.keys()) | set(gold.keys())
        scores = []
        
        for key in all_keys:
            if key not in pred or key not in gold:
                scores.append(0.0)
            elif isinstance(gold[key], (int, float)) and isinstance(pred[key], (int, float)):
                scores.append(self._calculate_number_score(float(pred[key]), float(gold[key])))
            elif isinstance(gold[key], str) and isinstance(pred[key], str):
                scores.append(self._compute_f1(pred[key], gold[key]))
            elif isinstance(gold[key], dict) and isinstance(pred[key], dict):
                scores.append(self._calculate_json_score(pred[key], gold[key]))
            elif isinstance(gold[key], list) and isinstance(pred[key], list):
                scores.append(self._score_answer(pred[key], gold[key]))
            else:
                scores.append(0.0)
                
        return sum(scores) / len(scores) if scores else 0.0
        
    def _compute_f1(self, pred: str, gold: str) -> float:
        """Compute F1 score between predicted and gold strings.
        
        Args:
            pred: Predicted string
            gold: Gold standard string
            
        Returns:
            F1 score between 0 and 1
        """
        pred_words = set(pred.lower().split())
        gold_words = set(gold.lower().split())
        
        if not pred_words or not gold_words:
            return 0.0
            
        intersection = len(pred_words & gold_words)
        precision = intersection / len(pred_words)
        recall = intersection / len(gold_words)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
        
    def _score_answer(
        self,
        pred: Union[str, int, float, List[Any], Dict[str, Any]],
        gold: Union[str, int, float, List[Any], Dict[str, Any]]
    ) -> float:
        """Score the answer provided by the model against the gold standard.
        
        Args:
            pred: Predicted answer
            gold: Gold standard answer
            
        Returns:
            Score between 0 and 1
        """
        pred = self._parse_and_classify(pred) if isinstance(pred, str) else pred
        gold = self._parse_and_classify(gold) if isinstance(gold, str) else gold
        
        if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
            return self._calculate_number_score(pred, gold)
        elif isinstance(gold, str) and isinstance(pred, str):
            return self._compute_f1(pred, gold)
        elif isinstance(gold, list) and isinstance(pred, list):
            if all(isinstance(item, str) for item in gold + pred):
                return self._compute_f1(" ".join(pred), " ".join(gold))
            elif all(isinstance(item, (int, float)) for item in gold + pred):
                return self._compute_f1(
                    " ".join([str(x) for x in pred]),
                    " ".join([str(x) for x in gold])
                )
            elif all(isinstance(item, dict) for item in gold + pred):
                return sum(
                    self._calculate_json_score(p, g)
                    for p, g in zip(pred, gold)
                ) / max(len(pred), len(gold))
        elif isinstance(gold, dict) and isinstance(pred, dict):
            return self._calculate_json_score(pred, gold)
            
        return 0.0  # Mismatched types
        
    def evaluate_problem(self, problem: AssistantBenchProblem) -> Dict[str, Any]:
        """Evaluate a single AssistantBench problem.
        
        Args:
            problem: The AssistantBench problem to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Build prompt
        prompt = self.INITIAL_PROMPT.format(
            task=problem.task
        )
        
        # Generate response
        response = self.model.generate(
            prompt=prompt,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        # Extract answer from response
        answer = response.text.strip()
        
        # Score the answer
        score = self._score_answer(answer, problem.answer)
        
        return {
            "problem_id": problem.problem_id,
            "task": problem.task,
            "generated_answer": answer,
            "gold_answer": problem.answer,
            "score": score,
            "explanation": problem.explanation,
            "difficulty": problem.difficulty
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the AssistantBench test.
        
        Returns:
            Dict containing test results
        """
        # Load dataset
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config["num_test_cases"], len(self.problems))
        test_cases = self.problems[:num_cases]
        
        # Evaluate problems
        results = []
        for problem in test_cases:
            result = self.evaluate_problem(problem)
            results.append(result)
            
        # Calculate overall metrics
        total_problems = len(results)
        total_score = sum(r["score"] for r in results)
        average_score = total_score / total_problems if total_problems > 0 else 0
        
        # Calculate metrics by difficulty
        difficulty_metrics = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r["difficulty"] == difficulty]
            if diff_results:
                diff_score = sum(r["score"] for r in diff_results)
                difficulty_metrics[difficulty] = {
                    "total_problems": len(diff_results),
                    "average_score": diff_score / len(diff_results)
                }
        
        return {
            "total_problems": total_problems,
            "average_score": average_score,
            "difficulty_metrics": difficulty_metrics,
            "results": results
        } 