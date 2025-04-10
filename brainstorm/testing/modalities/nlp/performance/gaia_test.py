"""GAIA test implementation for evaluating web agent capabilities."""
import json
import os
import re
import string
import tempfile
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
import shutil
from huggingface_hub import snapshot_download
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio

from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest

@dataclass
class GAIATask:
    """Represents a single GAIA task."""
    task_id: str
    question: str
    answer: str
    level: str
    annotator_metadata: Dict[str, Any]
    files: List[str]

class GAIATest(BasePerformanceTest):
    """Test for evaluating web agent capabilities on real-world tasks."""
    
    TASK_PROMPT = dedent("""
You are an autonomous AI agent, consisting of a large language model and a scaffolding program that runs on a linux VM.

You have full control over your VM through the scaffolding program's provided functions, including: writing and editing files, running bash commands, installing packages etc.. From your VM, you can access the internet.

You are state-of-the-art, and excellent at analysis, planning and programming. You possess encyclopedic knowledge of the world. You write in a very direct and concise style.

You are resourceful and adaptive, and you never give up. Before deciding something can't be done, you try it out. You consider multiple options and choose the best one. If your current approach doesn't work, you formulate a new plan. You are given a task you need to solve completely on your own.

Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.

{file}

Here is the question:

{question}
""").strip()
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the GAIA test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - subset: Which GAIA subset to evaluate (2023_all, 2023_level1, 2023_level2, 2023_level3)
                - split: Which split to evaluate (test or validation)
                - instance_ids: Optional list of specific instance IDs to evaluate
                - max_attempts: Maximum number of submission attempts
                - max_messages: Maximum number of messages before giving up
                - code_timeout: Timeout for code execution in seconds
        """
        super().__init__(config)
        self.subset = config.get("subset", "2023_all")
        self.split = config.get("split", "validation")
        self.instance_ids = config.get("instance_ids")
        self.max_attempts = config.get("max_attempts", 1)
        self.max_messages = config.get("max_messages", 100)
        self.code_timeout = config.get("code_timeout", 180)
        self.tasks: List[GAIATask] = []
        
    def load_dataset(self) -> None:
        """Load the GAIA dataset."""
        # Get the data directory path
        data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "gaia"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset if required
        if not os.path.exists(data_dir / "GAIA"):
            try:
                snapshot_download(
                    repo_id="gaia-benchmark/GAIA",
                    repo_type="dataset",
                    local_dir=data_dir,
                )
            except Exception as ex:
                shutil.rmtree(data_dir / "GAIA", True)
                raise ex
                
        # Load and parse tasks
        self.tasks = []
        dataset_path = data_dir / "GAIA" / "2023" / self.split
        
        # Read the dataset file
        with open(dataset_path / f"{self.subset}.json", "r") as f:
            records = json.load(f)
            
        for record in records:
            # Skip if instance_id doesn't match
            if self.instance_ids and record["task_id"] not in self.instance_ids:
                continue
                
            # Discover files
            files_location = dataset_path
            files = [file for file in os.listdir(files_location) if str(record["task_id"]) in file]
            
            task = GAIATask(
                task_id=record["task_id"],
                question=record["Question"],
                answer=record["Final answer"],
                level=record["Level"],
                annotator_metadata=record["Annotator Metadata"],
                files=files
            )
            self.tasks.append(task)
            
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.tasks)
            
    def _normalize_number_str(self, number_str: str) -> float:
        """Normalize a number string by removing common units and commas.
        
        Args:
            number_str: Input number string
            
        Returns:
            Normalized number as float
        """
        # Remove common units and commas
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
            
        try:
            return float(number_str)
        except ValueError:
            return float("inf")
            
    def _split_string(self, s: str, char_list: List[str] = [",", ";"]) -> List[str]:
        """Split a string by specified characters.
        
        Args:
            s: Input string
            char_list: List of characters to split by
            
        Returns:
            List of split strings
        """
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)
        
    def _normalize_str(self, input_str: str, remove_punct: bool = True) -> str:
        """Normalize a string by removing whitespace and optionally punctuation.
        
        Args:
            input_str: Input string
            remove_punct: Whether to remove punctuation
            
        Returns:
            Normalized string
        """
        # Remove all whitespace
        no_spaces = re.sub(r"\s", "", input_str)
        
        # Remove punctuation if specified
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()
            
    def _question_scorer(
        self,
        model_answer: str,
        ground_truth: str,
    ) -> tuple[bool, str]:
        """Score a model's answer against the ground truth.
        
        Args:
            model_answer: Model's answer
            ground_truth: Ground truth answer
            
        Returns:
            Tuple of (is_correct, explanation)
        """
        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False
                
        # If ground truth is a number
        if is_float(ground_truth):
            normalized_answer = self._normalize_number_str(model_answer)
            return (
                normalized_answer == float(ground_truth),
                f"Evaluated {model_answer} as a number."
            )
            
        # If ground truth is a list
        elif any(char in ground_truth for char in [",", ";"]):
            gt_elems = self._split_string(ground_truth)
            ma_elems = self._split_string(model_answer)
            
            # Check length is the same
            if len(gt_elems) != len(ma_elems):
                return (
                    False,
                    f"Evaluated {model_answer} as a comma separated list, returned False because lists have different lengths."
                )
                
            # Compare each element as float or str
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self._normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    comparisons.append(
                        self._normalize_str(ma_elem, remove_punct=False)
                        == self._normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons), f"Evaluated {model_answer} as a comma separated list."
            
        # If ground truth is a string
        else:
            return (
                self._normalize_str(model_answer) == self._normalize_str(ground_truth),
                f"Evaluated {model_answer} as a string."
            )
            
    async def _generate_response(self, prompt: str) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Generated response text
        """
        # Generate response
        response = await self.model.generate_async(
            prompt=prompt,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        return response.text.strip()
    
    async def evaluate_task(self, task: GAIATask) -> Dict[str, Any]:
        """Evaluate a single GAIA task.
        
        Args:
            task: The GAIA task to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Build prompt
        file_info = ""
        if task.files:
            file_info = f"The following file is referenced in the question below and you will likely need to use it in order to find the correct answer: /shared_files/{task.files[0]}"
            
        prompt = self.TASK_PROMPT.format(
            file=file_info,
            question=task.question
        )
        
        # Generate response
        answer = await self._generate_response(prompt)
        
        # Score the answer
        is_correct, explanation = self._question_scorer(answer, task.answer)
        
        return {
            "task_id": task.task_id,
            "question": task.question,
            "generated_answer": answer,
            "gold_answer": task.answer,
            "is_correct": is_correct,
            "explanation": explanation,
            "level": task.level,
            "annotator_metadata": task.annotator_metadata
        }
        
    async def run_async(self) -> Dict[str, Any]:
        """Run the GAIA test asynchronously.
        
        Returns:
            Dict containing test results
        """
        # Load dataset
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config.get("num_test_cases", len(self.tasks)), len(self.tasks))
        test_cases = self.tasks[:num_cases]
        
        # Evaluate tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 1))
        
        async def eval_with_semaphore(task):
            async with semaphore:
                return await self.evaluate_task(task)
        
        results = await tqdm_asyncio.gather(
            *[eval_with_semaphore(task) for task in test_cases],
            desc="Evaluating GAIA tasks"
        )
        
        # Calculate overall metrics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r["is_correct"])
        accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculate metrics by level
        level_metrics = {}
        for level in ["level1", "level2", "level3"]:
            level_results = [r for r in results if r["level"] == level]
            if level_results:
                level_correct = sum(1 for r in level_results if r["is_correct"])
                level_metrics[level] = {
                    "total_tasks": len(level_results),
                    "correct_tasks": level_correct,
                    "accuracy": level_correct / len(level_results)
                }
        
        return {
            "total_tasks": total_tasks,
            "correct_tasks": correct_tasks,
            "accuracy": accuracy,
            "level_metrics": level_metrics,
            "results": results
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the GAIA test.
        
        Returns:
            Dict containing test results
        """
        return asyncio.run(self.run_async()) 