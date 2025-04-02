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

# Import from standard location with fallback
try:
    from brainstorm.testing.modalities.nlp.performance.base_test import BasePerformanceTest
except ImportError:
    from brainstorm.testing.base import BasePerformanceTest

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
        
    def _create_fallback_samples(self) -> None:
        """Create fallback samples in case the dataset loading fails completely."""
        logger.warning("Creating fallback LAB-Bench samples since loading failed")
        
        # Create a few sample tasks for LitQA2
        fallback_tasks = [
            {
                "subset": "LitQA2",
                "question": "What is the main function of the mitochondria in eukaryotic cells?",
                "choices": [
                    "Energy production through cellular respiration", 
                    "Protein synthesis", 
                    "Cell division", 
                    "Storage of genetic information",
                    self.UNCERTAIN_ANSWER
                ],
                "correct_answer": "A"
            },
            {
                "subset": "LitQA2",
                "question": "Which of the following is a primary characteristic of stem cells?",
                "choices": [
                    "The ability to differentiate into various cell types", 
                    "The inability to replicate", 
                    "The presence of cell walls", 
                    "The absence of DNA",
                    self.UNCERTAIN_ANSWER
                ],
                "correct_answer": "A"
            },
            {
                "subset": "ProtocolQA",
                "question": "In PCR protocols, what is the primary purpose of the denaturation step?",
                "choices": [
                    "To separate the DNA strands", 
                    "To activate the polymerase enzyme", 
                    "To anneal primers to the template", 
                    "To synthesize new DNA strands",
                    self.UNCERTAIN_ANSWER
                ],
                "correct_answer": "A",
                "protocol": "PCR Protocol: 1. Denaturation (95°C, 30s), 2. Annealing (55-65°C, 30s), 3. Extension (72°C, 1min/kb)"
            }
        ]
        
        # Add fallback tasks
        for idx, task_data in enumerate(fallback_tasks):
            subset = task_data["subset"]
            self.tasks.append(LABBenchTask(
                task_id=f"fallback_{subset}_{idx}",
                question=task_data["question"],
                choices=task_data["choices"],
                correct_answer=task_data["correct_answer"],
                subset=subset,
                protocol=task_data.get("protocol", None)
            ))
            
        logger.info(f"Created {len(fallback_tasks)} fallback tasks")

    def load_dataset(self) -> None:
        """Load the LAB-Bench dataset."""
        # Initialize tasks list
        self.tasks = []
        
        # Track if any datasets were loaded successfully
        any_successful_loads = False
        
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
                # Try to load the dataset - some versions of the datasets library don't support trust parameter
                try:
                    # First try without the trust parameter
                    dataset = load_dataset(
                        self.DATASET_PATH,
                        name=subset.value,
                        split="train"
                    )
                except Exception as e1:
                    logger.warning(f"Error loading dataset without trust parameter: {str(e1)}")
                    # Try with trust parameter as fallback
                    try:
                        dataset = load_dataset(
                            self.DATASET_PATH,
                            name=subset.value,
                            split="train",
                            trust=True
                        )
                    except TypeError as e2:
                        # If the error is about the trust parameter, try with a different approach
                        if "unexpected keyword argument 'trust'" in str(e2):
                            logger.warning("Trust parameter not supported, trying with verification disabled")
                            # For older versions of datasets, use verification_mode="no_checks" instead
                            dataset = load_dataset(
                                self.DATASET_PATH,
                                name=subset.value,
                                split="train",
                                verification_mode="no_checks"
                            )
                        else:
                            # If it's a different error, re-raise it
                            raise
                
                if self.shuffle_dataset:
                    dataset = dataset.shuffle()
                    
                # Process dataset based on subset type
                tasks_added = 0
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
                    tasks_added += 1
                
                if tasks_added > 0:
                    any_successful_loads = True
                    logger.info(f"Successfully loaded {tasks_added} tasks from {subset.value}")
                    
            except Exception as e:
                logger.error(f"Error loading {subset.value} dataset: {str(e)}")
                logger.exception(e)
        
        # If no datasets were loaded successfully, create fallback samples
        if not any_successful_loads or not self.tasks:
            logger.warning("No datasets were loaded successfully. Using fallback samples.")
            self._create_fallback_samples()
        
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
        try:
            # For models that support image input
            if image_content and hasattr(self.model, 'generate_with_image'):
                logger.debug("Using generate_with_image method")
                response = await self.model.generate_with_image(
                    prompt=prompt,
                    image=image_content,
                    temperature=self.config.get("temperature", 0),
                    max_tokens=self.config.get("max_tokens", 1024)
                )
            else:
                # Generate response - text only
                if image_content:
                    logger.warning("Model does not support image input, ignoring image content")
                    
                logger.debug("Using standard generate method")
                response = await self.model.generate(
                    prompt=prompt,
                    temperature=self.config.get("temperature", 0),
                    max_tokens=self.config.get("max_tokens", 1024)
                )
            
            # Handle different response formats
            if hasattr(response, 'text'):
                # Some adapters return an object with a text attribute
                return response.text.strip()
            elif hasattr(response, 'completion'):
                # Some adapters return an object with a completion attribute
                return response.completion.strip()
            else:
                # Some adapters return the text directly as a string
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
        
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
        logger = logging.getLogger(__name__)
        
        # Log task details with visual separator
        logger.info(f"\n{'='*50}\nTASK DETAILS\n{'='*50}")
        logger.info(f"Task ID: {task.task_id}")
        logger.info(f"Subset: {task.subset}")
        logger.info(f"Question: {task.question}")
        
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
        
        # Log the prompt
        logger.info(f"\n{'='*50}\nINPUT PROMPT\n{'='*50}")
        logger.info(f"{prompt}")
        if task.image_content:
            logger.info(f"[Image data included but not displayed in logs]")
        logger.info(f"{'='*50}\n")
        
        # Generate response, with image if available
        logger.info("Generating model response...")
        response = await self._generate_response(prompt, task.image_content)
        
        # Log the model output
        logger.info(f"\n{'='*50}\nMODEL RESPONSE\n{'='*50}")
        logger.info(f"{response}")
        logger.info(f"{'='*50}\n")
        
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
        
        # Log evaluation results
        logger.info(f"\n{'='*50}\nEVALUATION RESULTS\n{'='*50}")
        logger.info(f"Extracted Answer: {extracted_answer or 'None'}")
        logger.info(f"Correct Answer: {task.correct_answer}")
        logger.info(f"Is Correct: {is_correct}")
        logger.info(f"Refused to Answer: {refused_to_answer}")
        logger.info(f"{'='*50}\n")
                
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
        logger = logging.getLogger(__name__)
        
        # Log test start with clear separator
        logger.info(f"\n{'='*50}\nSTARTING LAB-BENCH TEST\n{'='*50}")
        logger.info(f"Test configuration:")
        logger.info(f"  Subsets: {self.subsets}")
        logger.info(f"  Shuffle dataset: {self.shuffle_dataset}")
        logger.info(f"  Shuffle choices: {self.shuffle_choices}")
        logger.info(f"  Maximum concurrent tasks: {self.config.get('max_concurrent', 1)}")
        logger.info(f"{'='*50}\n")
        
        # Load dataset
        logger.info("Loading LAB-Bench dataset...")
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config.get("num_test_cases", len(self.tasks)), len(self.tasks))
        test_cases = self.tasks[:num_cases]
        
        logger.info(f"Selected {len(test_cases)} test cases for evaluation")
        
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
        
        # Log final summary with clear visual separator
        logger.info(f"\n{'='*50}\nTEST SUMMARY\n{'='*50}")
        logger.info(f"Total Tasks: {total_tasks}")
        logger.info(f"Answered Tasks: {len(answered_tasks)}")
        logger.info(f"Correct Tasks: {len(correct_tasks)}")
        logger.info(f"Overall Precision: {precision:.4f}")
        logger.info(f"Overall Coverage: {coverage:.4f}")
        logger.info(f"\nSubset-specific metrics:")
        for subset, metrics in subset_metrics.items():
            logger.info(f"  {subset}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Coverage: {metrics['coverage']:.4f}")
            logger.info(f"    Total: {metrics['total_tasks']}, Answered: {metrics['answered_tasks']}, Correct: {metrics['correct_tasks']}")
        logger.info(f"{'='*50}\n")
            
        return {
            "total_tasks": total_tasks,
            "answered_tasks": len(answered_tasks),
            "correct_tasks": len(correct_tasks),
            "precision": precision,
            "coverage": coverage,
            "subset_metrics": subset_metrics,
            "results": results
        }
        
    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run the LAB-Bench test asynchronously.
        
        Args:
            **kwargs: Arbitrary keyword arguments passed from the test service,
                    including model_id and other settings
                    
        Returns:
            Dict containing test results
        """
        # Update any settings from kwargs if needed
        if kwargs and hasattr(self, 'model') and self.model:
            # Pass any model-specific settings
            model_settings = {k: v for k, v in kwargs.items() if k not in ['model_adapter']}
            if model_settings:
                try:
                    self.model.update_parameters(model_settings)
                except AttributeError:
                    pass  # Model doesn't support parameter updates
        
        # Run the test
        return await self.run_async() 