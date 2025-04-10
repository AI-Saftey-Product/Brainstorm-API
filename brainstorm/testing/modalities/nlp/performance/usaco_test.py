"""USACO test implementation for evaluating competitive programming capabilities."""
import json
import os
import re
import signal
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

# Platform-specific imports
try:
    import resource
except ImportError:
    # For Windows compatibility
    resource = None

import requests
# Try to import from the standard location, but fall back to the new location if needed
try:
    from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest
except ImportError:
    from brainstorm.testing.base import BasePerformanceTest

@dataclass
class USACOProblem:
    """Represents a single USACO problem."""
    problem_id: str
    title: str
    difficulty: str  # bronze, silver, gold, platinum
    description: str
    input_format: str
    output_format: str
    sample_input: List[str]
    sample_output: List[str]
    time_limit: int  # in milliseconds
    memory_limit: int  # in megabytes
    test_cases: List[Dict[str, str]]  # List of {"input": str, "output": str}
    reference_solution: str
    test_assets_path: Path

class USACOTest(BasePerformanceTest):
    """Test for evaluating code generation on USACO problems."""
    
    INITIAL_PROMPT = dedent("""
Please reply with a Python 3 solution to the below problem. Make sure to wrap your code in '```python' and '```' Markdown
delimiters, and include exactly one block of code with the entire solution
(in the final code step).
Reason through the problem and:
1. Restate the problem in plain English
2. Conceptualize a solution first in plain English
3. Write a pseudocode solution
4. Output the final Python solution with your solution steps in comments.
No outside libraries are allowed.

[BEGIN PROBLEM]
{description}
[END PROBLEM]
""").strip()
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the USACO test.
        
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
                - timeout: Maximum time allowed for code execution
                - json_basename: Dataset to use ("usaco_subset307" or "usaco_v2")
        """
        super().__init__(config)
        self.difficulty = config.get("difficulty")
        self.timeout = config.get("timeout", 300)
        self.json_basename = config.get("json_basename", "usaco_subset307")
        self.problems: List[USACOProblem] = []
        
    def load_dataset(self) -> None:
        """Load the USACO dataset."""
        # Get the data directory path
        data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "usaco"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download problems if not exists
        problems_path = data_dir / f"{self.json_basename}_dict.json"
        if not problems_path.exists():
            self._download_problems(data_dir)
            
        # Load and parse problems
        self.problems = []
        with open(problems_path, "r") as f:
            records_dict = json.load(f)
            for record in records_dict.values():
                # Skip if difficulty doesn't match
                if self.difficulty and record["problem_level"] != self.difficulty:
                    continue
                    
                # Generate test assets path
                problem_id = record["problem_id"]
                if os.name == "nt":  # Windows
                    problem_id = problem_id.replace("?", "_").replace(":", "_")
                test_assets_path = data_dir / "usaco_v3" / "tests" / problem_id
                
                problem = USACOProblem(
                    problem_id=problem_id,
                    title=record["name"],
                    difficulty=record["problem_level"],
                    description=record["description"],
                    input_format=record.get("input_format", ""),
                    output_format=record.get("output_format", ""),
                    sample_input=record.get("sample_input", []),
                    sample_output=record.get("sample_output", []),
                    time_limit=record["runtime_limit"],
                    memory_limit=record["memory_limit"],
                    test_cases=[],  # Will be loaded on demand
                    reference_solution=record.get("reference_solution", ""),
                    test_assets_path=test_assets_path
                )
                self.problems.append(problem)
                
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.problems)
            
    def _download_problems(self, data_dir: Path) -> None:
        """Download the USACO problems.
        
        Args:
            data_dir: Directory to save the problems
        """
        # Download from Google Drive
        zip_url = "https://drive.usercontent.google.com/download?id=1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi&export=download&authuser=0&confirm=t&uuid=11ea8398-6e91-4a2b-92af-7f6aaf3c2f05&at=AEz70l4HhAdBdGm-QXYS7Y_mknoJ:1741251361013"
        zip_path = data_dir / "usaco_main.zip"
        
        # Download zip file
        print(f"Downloading {zip_url}")
        response = requests.get(zip_url)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download source from {zip_url} - {response.status_code}")
            
        # Extract zip file
        print(f"Unzipping {zip_path}")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir.parent)
            
        # Clean up
        if zip_path.exists():
            zip_path.unlink()
            
    def _load_test_cases(self, problem: USACOProblem) -> Tuple[List[bytes], List[str]]:
        """Load test cases for a problem.
        
        Args:
            problem: The USACO problem
            
        Returns:
            Tuple of (test_inputs, test_outputs)
        """
        test_inputs = []
        test_outputs = []
        num_tests = len(problem.test_cases)
        
        for test_num in range(1, num_tests + 1):
            test_names = [
                [f"{test_num}.in", f"{test_num}.out"],  # 1.in / 1.out
                [f"I.{test_num}", f"O.{test_num}"],  # I.1 / O.1
            ]
            
            for in_filename, out_filename in test_names:
                try:
                    with open(problem.test_assets_path / in_filename, "rb") as f:
                        test_inputs.append(f.read())
                    with open(problem.test_assets_path / out_filename, "r") as f:
                        test_outputs.append(f.read())
                    break
                except FileNotFoundError:
                    continue
                    
        assert len(test_inputs) == num_tests
        assert len(test_outputs) == num_tests
        
        return test_inputs, test_outputs
            
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
            
    def _generate_code(self, code: str, max_secs: int, max_bytes: int) -> str:
        """Generate code with resource limits.
        
        Args:
            code: The code to run
            max_secs: Maximum execution time in seconds
            max_bytes: Maximum memory in bytes
            
        Returns:
            The code with resource limits
        """
        preamble = dedent(f"""\
            import signal
            import resource
            import sys

            soft_secs, hard_secs = resource.getrlimit(resource.RLIMIT_CPU)
            soft_bytes, hard_bytes = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_CPU, ({max_secs}, hard_secs))
            resource.setrlimit(resource.RLIMIT_AS, ({max_bytes}, hard_bytes))
            signal.signal(signal.SIGXCPU, lambda signo,frame: sys.exit(1))

        """)
        
        return preamble + code
            
    def _execute_code(self, code: str, test_input: bytes, timeout: int) -> Tuple[str, str, bool]:
        """Execute code with timeout.
        
        Args:
            code: Python code to execute
            test_input: Input data
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, timed_out)
        """
        import threading
        import subprocess
        import tempfile
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(code.encode())
            f.flush()
            temp_name = f.name
            
        # Set resource limits if available (Unix only)
        def set_memory_limit(preexec_fn, limit_mb):
            if resource is not None:
                # Convert MB to bytes
                limit_bytes = limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        
        # Memory limit in MB
        memory_limit = 512
        
        # Result container
        result = {"stdout": "", "stderr": "", "timed_out": False}
        
        # Thread to execute code
        def execute():
            try:
                # Only set preexec_fn on Unix systems
                preexec_fn = None
                if resource is not None:
                    preexec_fn = lambda: set_memory_limit(None, memory_limit)
                
                # Run the code
                process = subprocess.Popen(
                    [sys.executable, temp_name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=preexec_fn
                )
                
                stdout, stderr = process.communicate(input=test_input, timeout=timeout)
                result["stdout"] = stdout.decode(errors="replace")
                result["stderr"] = stderr.decode(errors="replace")
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                process.kill()
            except Exception as ex:
                result["stderr"] = str(ex)
        
        # Start thread
        thread = threading.Thread(target=execute)
        thread.start()
        thread.join(timeout + 1)  # Give a little extra time for cleanup
        
        # Cleanup
        try:
            os.unlink(temp_name)
        except:
            pass
            
        return result["stdout"], result["stderr"], result["timed_out"]
            
    def _clean_output(self, output: str) -> str:
        """Clean output for comparison.
        
        Args:
            output: Raw output
            
        Returns:
            Cleaned output
        """
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
            
        # Remove whitespace
        output = output.strip()
        
        # Remove trailing newlines
        output = output.rstrip("\n")
        
        # Normalize line endings
        output = output.replace("\r\n", "\n")
        
        return output
            
    def evaluate_problem(self, problem: USACOProblem) -> Dict[str, Any]:
        """Evaluate a single USACO problem.
        
        Args:
            problem: USACO problem to evaluate
            
        Returns:
            Evaluation results
        """
        # Generate prompt
        system_prompt = "You are a competitive programming assistant for USACO (USA Computing Olympiad) problems."
        prompt = self.INITIAL_PROMPT.format(description=problem.description)
        
        # Generate code
        try:
            # Generate code using model
            output = self.model.generate(
                system=system_prompt,
                user=prompt,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 4096)
            )
            
            # Extract code
            completion = output.completion
            code = self._extract_code(completion)
            
            # Prepare code for execution
            input_dtype = None
            if problem.input_format:
                if "integer" in problem.input_format.lower():
                    input_dtype = "int"
                elif "float" in problem.input_format.lower() or "real" in problem.input_format.lower():
                    input_dtype = "float"
            
            code = self._generate_code(code, problem.time_limit, problem.memory_limit)
            
            # Load test cases
            cases = problem.test_cases
            
            # If test cases not loaded, load them
            if not cases:
                test_assets_path = problem.test_assets_path
                if test_assets_path.exists() and test_assets_path.is_dir():
                    files = list(test_assets_path.iterdir())
                    
                    # Count test cases
                    num_tests = 0
                    for f in files:
                        if f.name.endswith(".in") or f.name.startswith("I."):
                            num_tests += 1
                    
                    # Load test cases
                    test_inputs, test_outputs = self._load_test_cases(problem)
                    
                    # Store test cases
                    cases = []
                    for i in range(len(test_inputs)):
                        cases.append({"input": test_inputs[i], "output": test_outputs[i]})
                    
                    # Update problem
                    problem.test_cases = cases
            
            # Execute code on each test case
            test_results = []
            for i, case in enumerate(cases):
                result = {}
                
                # Run code
                try:
                    # Execute code with timeout
                    stdout, stderr, timed_out = self._execute_code(
                        code, 
                        case["input"] if isinstance(case["input"], bytes) else case["input"].encode(),
                        problem.time_limit / 1000
                    )
                    
                    # Process result
                    result = {
                        "test_case": i,
                        "success": False,
                        "stdout": stdout,
                        "stderr": stderr,
                        "timed_out": timed_out
                    }
                    
                    # Compare output with expected
                    expected = case["output"]
                    if not timed_out and not stderr:
                        # Clean output and expected
                        clean_stdout = self._clean_output(stdout)
                        clean_expected = self._clean_output(expected)
                        
                        # Check if output matches expected
                        result["success"] = clean_stdout == clean_expected
                        result["output_match"] = result["success"]
                    
                except Exception as e:
                    result = {
                        "test_case": i,
                        "success": False,
                        "error": str(e),
                        "timed_out": False
                    }
                
                test_results.append(result)
            
            # Aggregate results
            success_count = sum(1 for r in test_results if r.get("success", False))
            
            # Prepare result
            result = {
                "problem_id": problem.problem_id,
                "title": problem.title,
                "difficulty": problem.difficulty,
                "success": success_count == len(test_results) and len(test_results) > 0,
                "success_count": success_count,
                "total_count": len(test_results),
                "success_rate": success_count / len(test_results) if test_results else 0,
                "code": code,
                "test_results": test_results,
                "completion": completion
            }
            
            return result
            
        except Exception as e:
            return {
                "problem_id": problem.problem_id,
                "title": problem.title,
                "difficulty": problem.difficulty,
                "success": False,
                "error": str(e)
            }
        
    def run(self) -> Dict[str, Any]:
        """Run the USACO test.
        
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
        total_tests = sum(r["total_count"] for r in results)
        total_passed_tests = sum(r["success_count"] for r in results)
        overall_pass_rate = total_passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate metrics by difficulty
        difficulty_metrics = {}
        for difficulty in ["bronze", "silver", "gold", "platinum"]:
            diff_results = [r for r in results if r["difficulty"] == difficulty]
            if diff_results:
                diff_tests = sum(r["total_count"] for r in diff_results)
                diff_passed = sum(r["success_count"] for r in diff_results)
                difficulty_metrics[difficulty] = {
                    "total_problems": len(diff_results),
                    "total_tests": diff_tests,
                    "passed_tests": diff_passed,
                    "pass_rate": diff_passed / diff_tests if diff_tests > 0 else 0
                }
        
        return {
            "total_problems": total_problems,
            "total_tests": total_tests,
            "total_passed_tests": total_passed_tests,
            "overall_pass_rate": overall_pass_rate,
            "difficulty_metrics": difficulty_metrics,
            "results": results
        } 