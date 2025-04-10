"""MBPP test implementation for evaluating code generation capabilities."""
import json
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset  # type: ignore
from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest
from brainstorm.testing.modalities.nlp.performance.humaneval_test import HumanEvalTest

@dataclass
class MBPPExample:
    """Represents a single MBPP example."""
    task_id: int
    text: str
    code: str
    test_list: List[str]
    difficulty: str
    category: str
    source_file: str
    test_imports: List[str]

class MBPPTest(BasePerformanceTest):
    """Test for evaluating code generation on the MBPP dataset."""
    
    PROMPT_TEMPLATE = """
You are an expert Python programmer. You will be given a task, and the tests that your code must pass.
Write the Python function to solve the task. Do not give additional explanations, just output the
Python function. Only use imports that are included in Python's standard library.
""".strip()
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MBPP test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - use_sanitized: Whether to use sanitized test split
                - difficulty_level: Target difficulty level (entry/intermediate/advanced)
        """
        super().__init__(config)
        self.use_sanitized = config.get("use_sanitized", True)
        self.difficulty_level = config.get("difficulty_level", "entry")
        self.examples: List[MBPPExample] = []
        self.few_shot_examples: List[MBPPExample] = []
        
    def load_dataset(self) -> None:
        """Load the MBPP dataset."""
        # Load dataset from HuggingFace
        dataset = load_dataset(
            "google-research-datasets/mbpp",
            "sanitized" if self.use_sanitized else "full",
            split="test"
        )
        
        # Convert to MBPPExample objects
        self.examples = []
        for item in dataset:
            if item["difficulty"] == self.difficulty_level:
                example = MBPPExample(
                    task_id=item["task_id"],
                    text=item["text"],
                    code=item["code"],
                    test_list=item["test_list"],
                    difficulty=item["difficulty"],
                    category=item["category"],
                    source_file=item.get("source_file", ""),
                    test_imports=item.get("test_imports", [])
                )
                self.examples.append(example)
                
        # Load few-shot examples (task IDs 2, 3, 4)
        few_shot_ids = [2, 3, 4]
        self.few_shot_examples = [
            ex for ex in self.examples if ex.task_id in few_shot_ids
        ]
                
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.examples)
            
    def _build_prompt(self, example: MBPPExample) -> str:
        """Build the prompt for code generation.
        
        Args:
            example: The MBPP example to generate code for
            
        Returns:
            The complete prompt including few-shot examples
        """
        prompt = self.PROMPT_TEMPLATE + "\n\nFor example:\n\n"
        
        # Add few-shot examples
        for i, sample in enumerate(self.few_shot_examples):
            test_cases = "\n".join(sample.test_list)
            prompt += "".join([
                f"## Prompt {i + 1}\n",
                "```python\n",
                f"{sample.text}\n",
                "```\n\n",
                f"## Test Case {i + 1}\n",
                "```python\n",
                f"{test_cases}\n```\n\n",
                f"## Completion {i + 1}\n",
                "```python\n",
                f"{sample.code}\n```\n\n",
            ])
            
        # Add current task
        prompt += f"""
# Now, do it for the following task.

## Prompt:
```python
{example.text}
```

## Test Case:
```python
{chr(10).join(example.test_list)}
```

## Completion:
"""
        return prompt
            
    def _extract_code(self, completion: str) -> str:
        """Extract code from completion, handling markdown formatting.
        
        Args:
            completion: The raw completion text
            
        Returns:
            The extracted code
        """
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        if matches:
            return matches[0].strip()
        return completion.strip()
            
    def evaluate_example(self, example: MBPPExample) -> Dict[str, Any]:
        """Evaluate a single MBPP example.
        
        Args:
            example: The MBPP example to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Build prompt with few-shot examples
        prompt = self._build_prompt(example)

        # Generate code
        response = self.model.generate(
            prompt=prompt,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        # Extract generated code
        generated_code = self._extract_code(response.text)
        
        # Prepare test environment
        test_code = f"""
{generated_code}

# Test cases
{chr(10).join(example.test_list)}
"""
        
        # Run tests in sandboxed environment
        try:
            # TODO: Implement sandboxed execution
            # For now, using exec() as fallback
            exec(test_code, {}, {})
            passed = True
            error = None
        except Exception as e:
            passed = False
            error = str(e)
            
        return {
            "task_id": example.task_id,
            "difficulty": example.difficulty,
            "category": example.category,
            "passed": passed,
            "error": error,
            "generated_code": generated_code,
            "reference_code": example.code,
            "source_file": example.source_file,
            "test_imports": example.test_imports
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the MBPP test.
        
        Returns:
            Dict containing test results
        """
        # Load dataset
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config["num_test_cases"], len(self.examples))
        test_cases = self.examples[:num_cases]
        
        # Evaluate examples
        results = []
        for example in test_cases:
            result = self.evaluate_example(example)
            results.append(result)
            
        # Calculate metrics
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = passed / total if total > 0 else 0
        
        # Calculate pass@k metrics
        pass_at_1 = pass_rate
        pass_at_2 = sum(1 for r in results[:2] if r["passed"]) / min(2, total)
        pass_at_5 = sum(1 for r in results[:5] if r["passed"]) / min(5, total)
        
        return {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "pass_at_1": pass_at_1,
            "pass_at_2": pass_at_2,
            "pass_at_5": pass_at_5,
            "results": results
        } 