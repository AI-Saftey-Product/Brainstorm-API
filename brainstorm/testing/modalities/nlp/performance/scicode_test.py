"""SciCode test implementation for evaluating scientific code generation capabilities."""
import json
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

import requests
from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest

@dataclass
class SciCodeSubProblem:
    """Represents a single subproblem in a SciCode problem."""
    step_description: str
    function_header: str
    return_line: str
    test_list: List[str]
    reference_code: str
    step_background: Optional[str] = None

@dataclass
class SciCodeProblem:
    """Represents a single SciCode problem."""
    problem_id: str
    problem_description: str
    required_dependencies: List[str]
    subproblems: List[SciCodeSubProblem]
    scientific_background: Optional[str] = None

class SciCodeTest(BasePerformanceTest):
    """Test for evaluating code generation on scientific problems."""
    
    INITIAL_PROMPT = """
PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

For each subproblem provided you must solve it as follows:
 1. Generate scientific background required for the next step, in a comment
 2. Implement a function to solve the problem provided, using the provided header and return line

The response must be formatted as ```python```
""".strip()
    
    INITIAL_PROMPT_WITH_BACKGROUND = """
PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation
 4. scientific background information that may be used to inform your response

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

You must solve each subproblem provided by implementing a function to solve the subproblem provided, using the provided header and return line. Remember that the functions you have defined to solve previous subproblems can be used in your solution.

The response must be formatted as ```python```
""".strip()
    
    SUBPROBLEM_PROMPT = """
Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python``` and includes necessary background as a comment at the top.

SUBPROBLEM DESCRIPTION:
{step_description}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
""".strip()
    
    SUBPROBLEM_PROMPT_WITH_BACKGROUND = """
Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python```.

SUBPROBLEM DESCRIPTION:
{step_description}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

SCIENTIFIC BACKGROUND:
{step_background}

Example:
```python

[Insert the Python code here based on the provided function header and dependencies.]
```
""".strip()
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SciCode test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - provide_scientific_background: Whether to include scientific background
                - timeout: Maximum time allowed for code execution
                - include_dev_set: Whether to include development set
        """
        super().__init__(config)
        self.provide_scientific_background = config.get("provide_scientific_background", False)
        self.timeout = config.get("timeout", 300)
        self.include_dev_set = config.get("include_dev_set", False)
        self.problems: List[SciCodeProblem] = []
        
    def load_dataset(self) -> None:
        """Load the SciCode dataset."""
        # Get the data directory path
        data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "scicode"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download problems if not exists
        problems_path = data_dir / ("problems_incl_dev.jsonl" if self.include_dev_set else "problems_excl_dev.jsonl")
        if not problems_path.exists():
            self._download_problems(problems_path)
            
        # Load and parse problems
        self.problems = []
        with open(problems_path, "r") as f:
            for line in f:
                problem_data = json.loads(line)
                subproblems = []
                for step in problem_data["steps"]:
                    subproblem = SciCodeSubProblem(
                        step_description=step["description"],
                        function_header=step["function_header"],
                        return_line=step["return_line"],
                        test_list=step.get("test_list", []),
                        reference_code=step.get("reference_code", ""),
                        step_background=step.get("background")
                    )
                    subproblems.append(subproblem)
                    
                problem = SciCodeProblem(
                    problem_id=problem_data["problem_id"],
                    problem_description=problem_data["description"],
                    required_dependencies=problem_data["required_dependencies"],
                    subproblems=subproblems,
                    scientific_background=problem_data.get("scientific_background")
                )
                self.problems.append(problem)
                
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.problems)
            
    def _download_problems(self, target_path: Path) -> None:
        """Download the SciCode problems.
        
        Args:
            target_path: Path to save the problems
        """
        base_url = "https://raw.githubusercontent.com/scicode-bench/SciCode/69a8cfc/eval/data"
        
        if self.include_dev_set:
            urls = [
                f"{base_url}/problems_all.jsonl",
                f"{base_url}/problems_dev.jsonl"
            ]
            with open(target_path, "wb") as f:
                for url in urls:
                    response = requests.get(url)
                    response.raise_for_status()
                    f.write(response.content)
        else:
            url = f"{base_url}/problems_all.jsonl"
            response = requests.get(url)
            response.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(response.content)
                
    def _build_initial_prompt(self, problem: SciCodeProblem) -> str:
        """Build the initial prompt for a problem.
        
        Args:
            problem: The SciCode problem
            
        Returns:
            The complete initial prompt
        """
        template = self.INITIAL_PROMPT_WITH_BACKGROUND if self.provide_scientific_background else self.INITIAL_PROMPT
        return template.format(
            required_dependencies="\n".join(problem.required_dependencies)
        )
            
    def _build_subproblem_prompt(self, subproblem: SciCodeSubProblem) -> str:
        """Build the prompt for a subproblem.
        
        Args:
            subproblem: The SciCode subproblem
            
        Returns:
            The complete subproblem prompt
        """
        template = self.SUBPROBLEM_PROMPT_WITH_BACKGROUND if self.provide_scientific_background else self.SUBPROBLEM_PROMPT
        return template.format(
            step_description=subproblem.step_description,
            function_header=subproblem.function_header,
            return_line=subproblem.return_line,
            step_background=subproblem.step_background or ""
        )
            
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
            
    def evaluate_problem(self, problem: SciCodeProblem) -> Dict[str, Any]:
        """Evaluate a single SciCode problem.
        
        Args:
            problem: The SciCode problem to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Build initial prompt
        initial_prompt = self._build_initial_prompt(problem)
        
        # Generate code for each subproblem
        subproblem_results = []
        generated_functions = {}
        
        for i, subproblem in enumerate(problem.subproblems):
            # Build subproblem prompt
            subproblem_prompt = self._build_subproblem_prompt(subproblem)
            
            # Generate code
            response = self.model.generate(
                prompt=subproblem_prompt,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            
            # Extract generated code
            generated_code = self._extract_code(response.text)
            
            # Prepare test environment
            test_code = f"""
{generated_code}

# Test cases
{chr(10).join(subproblem.test_list)}
"""
            
            # Run tests
            try:
                exec(test_code, {}, {})
                passed = True
                error = None
            except Exception as e:
                passed = False
                error = str(e)
                
            subproblem_result = {
                "step_number": i + 1,
                "passed": passed,
                "error": error,
                "generated_code": generated_code,
                "reference_code": subproblem.reference_code
            }
            subproblem_results.append(subproblem_result)
            
            if passed:
                # Add function to namespace for use in later subproblems
                exec(generated_code, generated_functions)
            
        # Calculate overall problem metrics
        total_steps = len(subproblem_results)
        passed_steps = sum(1 for r in subproblem_results if r["passed"])
        pass_rate = passed_steps / total_steps if total_steps > 0 else 0
        
        return {
            "problem_id": problem.problem_id,
            "total_steps": total_steps,
            "passed_steps": passed_steps,
            "pass_rate": pass_rate,
            "subproblem_results": subproblem_results
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the SciCode test.
        
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
        total_steps = sum(r["total_steps"] for r in results)
        total_passed_steps = sum(r["passed_steps"] for r in results)
        overall_pass_rate = total_passed_steps / total_steps if total_steps > 0 else 0
        
        return {
            "total_problems": total_problems,
            "total_steps": total_steps,
            "total_passed_steps": total_passed_steps,
            "overall_pass_rate": overall_pass_rate,
            "results": results
        } 