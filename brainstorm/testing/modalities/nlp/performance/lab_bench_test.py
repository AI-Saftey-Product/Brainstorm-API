"""LAB-Bench test implementation for evaluating LLM capabilities for scientific research workflows."""
import json
import os
import re
import base64
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import logging

from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest

logger = logging.getLogger(__name__)

class DatasetSubsets(Enum):
    """LAB-Bench dataset subsets."""
    LitQA2 = "LitQA2"
    SuppQA = "SuppQA"
    FigQA = "FigQA"
    TableQA = "TableQA"
    DbQA = "DbQA"
    ProtocolQA = "ProtocolQA"
    SeqQA = "SeqQA"
    CloningScenarios = "CloningScenarios"


@dataclass
class LABBenchTask:
    """Represents a single LAB-Bench task."""
    task_id: str
    question: str
    choices: List[str]
    correct_answer: str
    subset: str
    paper_title: Optional[str] = None
    doi: Optional[str] = None
    protocol: Optional[str] = None
    figures: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    image_content: Optional[str] = None


class LABBenchTest(BasePerformanceTest):
    """Test for evaluating LLM scientific research capabilities across various modalities."""
    
    MULTIPLE_CHOICE_TEMPLATE = """
The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Think step by step.

Question: {question}
Options:
{choices}

You MUST include the letter of the correct answer within the following format: 'ANSWER: $LETTER' (without quotes). For example, 'ANSWER: <answer>', where <answer> is the correct letter. Always answer in exactly this format of a single letter, even if you are unsure. We require this because we use automatic parsing.
"""

    UNCERTAIN_ANSWER = "Insufficient information to answer the question."
    DATASET_PATH = "futurehouse/lab-bench"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LAB-Bench test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - subsets: List of LAB-Bench subsets to evaluate
                - metrics: List of metrics to calculate (precision, coverage)
        """
        super().__init__(config)
        self.subsets = config.get("subsets", ["LitQA2"])
        self.metrics = config.get("metrics", ["precision", "coverage"])
        self.shuffle_dataset = config.get("shuffle", False)
        self.shuffle_choices = config.get("shuffle_choices", False)
        self.tasks: List[LABBenchTask] = []
        
    def convert_bytes_to_base64_encoded_image(self, mime_type: str, data_bytes: bytes) -> str:
        """Convert image bytes to base64 encoded string.
        
        Args:
            mime_type: MIME type of the image
            data_bytes: Raw image bytes
            
        Returns:
            Base64 encoded image string
        """
        return f"data:{mime_type};base64,{base64.b64encode(data_bytes).decode('utf-8')}"
        
    def load_dataset(self) -> None:
        """Load the LAB-Bench dataset."""
        # Initialize tasks list
        self.tasks = []
        
        # Process each subset
        for subset_name in self.subsets:
            try:
                subset = DatasetSubsets(subset_name)
            except ValueError:
                logger.warning(f"Unknown subset {subset_name}, skipping")
                continue
            
            # Load from HuggingFace
            logger.info(f"Loading {subset.value} dataset from {self.DATASET_PATH}")
            try:
                # Load dataset with trust=True since this is a trusted dataset
                dataset = load_dataset(
                    self.DATASET_PATH,
                    name=subset.value,
                    split="train",
                    trust=True
                )
                
                if self.shuffle_dataset:
                    dataset = dataset.shuffle()
                    
                # Process dataset based on subset type
                for idx, record in enumerate(dataset):
                    if subset == DatasetSubsets.SuppQA:
                        task = self._process_suppqa_record(record, idx, subset.value)
                    elif subset == DatasetSubsets.FigQA:
                        task = self._process_figqa_record(record, idx, subset.value)
                    elif subset == DatasetSubsets.TableQA:
                        task = self._process_tableqa_record(record, idx, subset.value)
                    elif subset == DatasetSubsets.ProtocolQA:
                        task = self._process_protocolqa_record(record, idx, subset.value)
                    else:
                        # Default processing for other subsets
                        task = self._process_base_record(record, idx, subset.value)
                        
                    self.tasks.append(task)
                    
            except Exception as e:
                logger.error(f"Error loading {subset.value} dataset: {str(e)}")
                logger.exception(e)
            
    def _process_base_record(self, record: Dict[str, Any], idx: int, subset: str) -> LABBenchTask:
        """Process a basic LAB-Bench record.
        
        Args:
            record: Raw record from dataset
            idx: Task index
            subset: Dataset subset name
            
        Returns:
            Processed LAB-Bench task
        """
        # Get correct choice options
        choices = [record.get("ideal", "")]
        
        # Add distractors
        distractors = record.get("distractors", [])
        choices.extend(distractors)
        
        # Add uncertain answer option
        choices.append(self.UNCERTAIN_ANSWER)
        
        # Shuffle choices if configured
        if self.shuffle_choices:
            import random
            correct_choice = choices[0]
            remaining_choices = choices[1:]
            random.shuffle(remaining_choices)
            choices = [correct_choice] + remaining_choices
        
        return LABBenchTask(
            task_id=f"{subset}_{idx}",
            question=record.get("question", ""),
            choices=choices,
            correct_answer="A",  # In the dataset, the correct answer is always the first choice
            subset=subset
        )
        
    def _process_suppqa_record(self, record: Dict[str, Any], idx: int, subset: str) -> LABBenchTask:
        """Process a SuppQA record.
        
        Args:
            record: Raw record from dataset
            idx: Task index
            subset: Dataset subset name
            
        Returns:
            Processed SuppQA task
        """
        task = self._process_base_record(record, idx, subset)
        task.paper_title = record.get("paper-title", "")
        task.doi = record.get("source", "")
        return task
        
    def _process_figqa_record(self, record: Dict[str, Any], idx: int, subset: str) -> LABBenchTask:
        """Process a FigQA record.
        
        Args:
            record: Raw record from dataset
            idx: Task index
            subset: Dataset subset name
            
        Returns:
            Processed FigQA task
        """
        task = self._process_base_record(record, idx, subset)
        
        # Handle figure
        if "figure" in record and "bytes" in record["figure"]:
            # Convert binary image data to base64
            image_bytes = record["figure"]["bytes"]
            image_content = self.convert_bytes_to_base64_encoded_image("image/jpeg", image_bytes)
            task.image_content = image_content
            task.figures = [record.get("figure", {})]
            
        return task
        
    def _process_tableqa_record(self, record: Dict[str, Any], idx: int, subset: str) -> LABBenchTask:
        """Process a TableQA record.
        
        Args:
            record: Raw record from dataset
            idx: Task index
            subset: Dataset subset name
            
        Returns:
            Processed TableQA task
        """
        task = self._process_base_record(record, idx, subset)
        
        # Handle tables - can have multiple table images
        if "tables" in record:
            tables = record["tables"]
            if tables:
                # If there are multiple tables, use the first one for now
                # In a complete implementation, we would handle multiple tables
                table_bytes = tables[0].get("bytes", None)
                if table_bytes:
                    image_content = self.convert_bytes_to_base64_encoded_image("image/png", table_bytes)
                    task.image_content = image_content
                
            task.tables = record.get("tables", [])
            
        return task
        
    def _process_protocolqa_record(self, record: Dict[str, Any], idx: int, subset: str) -> LABBenchTask:
        """Process a ProtocolQA record.
        
        Args:
            record: Raw record from dataset
            idx: Task index
            subset: Dataset subset name
            
        Returns:
            Processed ProtocolQA task
        """
        task = self._process_base_record(record, idx, subset)
        task.protocol = record.get("protocol", "")
        return task
        
    async def _generate_response(self, prompt: str, image_content: Optional[str] = None) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: The prompt to send to the model
            image_content: Optional base64 encoded image
            
        Returns:
            Generated response text
        """
        # For models that support image input
        if image_content and hasattr(self.model, 'generate_with_image_async'):
            response = await self.model.generate_with_image_async(
                prompt=prompt,
                image=image_content,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        else:
            # Generate response - text only
            if image_content:
                logger.warning("Model does not support image input, ignoring image content")
                
            response = await self.model.generate_async(
                prompt=prompt,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        
        return response.text.strip()
        
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the answer letter from a response.
        
        Args:
            response: Model's response text
            
        Returns:
            Extracted answer letter or None if not found
        """
        # Match "ANSWER: X" pattern
        match = re.search(r"ANSWER:\s*([A-Za-z])", response)
        if match:
            return match.group(1).upper()
        
        # Try to find any single letter that could be the answer
        letters = re.findall(r"\b([A-Za-z])\b", response)
        if letters:
            # Return the last single letter (most likely the answer)
            return letters[-1].upper()
            
        return None
        
    async def evaluate_task(self, task: LABBenchTask) -> Dict[str, Any]:
        """Evaluate a single LAB-Bench task.
        
        Args:
            task: The LAB-Bench task to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Format choices as A, B, C, etc.
        choices_text = ""
        for i, choice in enumerate(task.choices):
            choices_text += f"{chr(65 + i)}: {choice}\n"
            
        # Build question based on subset type
        question_text = task.question
        if task.subset == "SuppQA" and task.paper_title and task.doi:
            question_text = f"Paper title: {task.paper_title}\nDOI: {task.doi}\n{task.question}"
        elif task.subset == "ProtocolQA" and task.protocol:
            question_text = f"Protocol: {task.protocol}\n\n{task.question}"
            
        # Format prompt
        prompt = self.MULTIPLE_CHOICE_TEMPLATE.format(
            question=question_text,
            choices=choices_text.strip()
        )
        
        # Generate response, with image if available
        response = await self._generate_response(prompt, task.image_content)
        
        # Extract answer
        extracted_answer = self._extract_answer(response)
        
        # Check if correct
        is_correct = extracted_answer == task.correct_answer
        
        # Check if model refused to answer (selected the "Insufficient information" option)
        refused_to_answer = False
        if extracted_answer:
            answer_idx = ord(extracted_answer) - ord('A')
            if answer_idx < len(task.choices) and task.choices[answer_idx] == self.UNCERTAIN_ANSWER:
                refused_to_answer = True
                
        return {
            "task_id": task.task_id,
            "subset": task.subset,
            "question": task.question,
            "choices": task.choices,
            "correct_answer": task.correct_answer,
            "generated_response": response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "refused_to_answer": refused_to_answer
        }
        
    async def run_async(self) -> Dict[str, Any]:
        """Run the LAB-Bench test asynchronously.
        
        Returns:
            Dict containing test results
        """
        # Load dataset
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config.get("num_test_cases", len(self.tasks)), len(self.tasks))
        test_cases = self.tasks[:num_cases]
        
        if not test_cases:
            logger.warning("No test cases loaded")
            return {
                "total_tasks": 0,
                "answered_tasks": 0,
                "correct_tasks": 0,
                "precision": 0,
                "coverage": 0,
                "subset_metrics": {},
                "results": []
            }
        
        # Evaluate tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 1))
        
        async def eval_with_semaphore(task):
            async with semaphore:
                return await self.evaluate_task(task)
        
        logger.info(f"Running LAB-Bench evaluation on {len(test_cases)} tasks with concurrency {self.config.get('max_concurrent', 1)}")
        results = await tqdm_asyncio.gather(
            *[eval_with_semaphore(task) for task in test_cases],
            desc="Evaluating LAB-Bench tasks"
        )
        
        # Calculate overall metrics
        total_tasks = len(results)
        answered_tasks = [r for r in results if not r["refused_to_answer"]]
        correct_tasks = [r for r in results if r["is_correct"]]
        
        # Calculate precision (correct / answered)
        precision = len(correct_tasks) / len(answered_tasks) if answered_tasks else 0
        
        # Calculate coverage (answered / total)
        coverage = len(answered_tasks) / total_tasks if total_tasks > 0 else 0
        
        # Calculate subset-specific metrics
        subset_metrics = {}
        for subset in set(r["subset"] for r in results):
            subset_results = [r for r in results if r["subset"] == subset]
            subset_answered = [r for r in subset_results if not r["refused_to_answer"]]
            subset_correct = [r for r in subset_results if r["is_correct"]]
            
            subset_metrics[subset] = {
                "total_tasks": len(subset_results),
                "answered_tasks": len(subset_answered),
                "correct_tasks": len(subset_correct),
                "precision": len(subset_correct) / len(subset_answered) if subset_answered else 0,
                "coverage": len(subset_answered) / len(subset_results) if subset_results else 0
            }
        
        # Log results
        logger.info(f"LAB-Bench evaluation completed with precision={precision:.4f}, coverage={coverage:.4f}")
        for subset, metrics in subset_metrics.items():
            logger.info(f"  {subset}: precision={metrics['precision']:.4f}, coverage={metrics['coverage']:.4f}")
            
        return {
            "total_tasks": total_tasks,
            "answered_tasks": len(answered_tasks),
            "correct_tasks": len(correct_tasks),
            "precision": precision,
            "coverage": coverage,
            "subset_metrics": subset_metrics,
            "results": results
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the LAB-Bench test.
        
        Returns:
            Dict containing test results
        """
        return asyncio.run(self.run_async()) 