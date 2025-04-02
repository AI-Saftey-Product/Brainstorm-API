"""Mind2Web test implementation for evaluating web agent capabilities."""
import json
import os
import re
import string
from typing import Dict, Any, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download
from tqdm import tqdm

from brainstorm.testing.base import BasePerformanceTest
from brainstorm.testing.registry import register_test

ActionType = Literal["CLICK", "SELECT", "TYPE"]

class Mind2WebSplit(str, Enum):
    """Enum for Mind2Web dataset splits.

    Each split corresponds to a different generalization setting:
    - CROSS_TASK: Tests generalization to new tasks on websites seen during training
    - CROSS_WEBSITE: Tests generalization to new websites within domains seen during training
    - CROSS_DOMAIN: Tests generalization to entirely new domains and websites
    """
    TRAIN = "train"
    CROSS_TASK = "test_task"
    CROSS_WEBSITE = "test_website"
    CROSS_DOMAIN = "test_domain"

@dataclass
class Answer:
    """Answer for a Mind2Web task step."""
    letter: str
    action: Optional[ActionType] = None
    value: Optional[str] = None

@dataclass
class Mind2WebTask:
    """Represents a single Mind2Web task."""
    task_id: str
    annotation_id: str
    action_uid: str
    confirmed_task: str
    final_html: str
    previous_actions: str
    choices_text: str
    target_idx: int
    choices: List[Dict[str, Any]]
    answer: Answer
    domain: str
    website: str
    subdomain: str
    website_url: str
    metadata: Dict[str, Any]

class Mind2WebTest(BasePerformanceTest):
    """Test for evaluating web agent capabilities on real-world web tasks."""
    
    TASK_PROMPT = """'''
{final_html}
'''

Based on the HTML webpage above, try to complete the following task:
Task: {confirmed_task}

Previous actions:
{previous_actions}

What should be the next action? Please select from the following choices
(If the correct action is not in the page above, please select A. 'None of the above'):

{choices_text}
"""

    SYSTEM_PROMPT = """You are an expert web navigation assistant. 
Your task is to help users interact with web pages by selecting the appropriate elements and actions.
When given an HTML page and a task, analyze the page structure carefully and choose the most appropriate action.
Pay special attention to the IDs, classes, and text content of elements to identify the best match for the task.
"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Mind2Web test.
        
        Args:
            config: Test configuration containing:
                - epochs: Number of evaluation epochs
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - num_test_cases: Number of test cases to evaluate
                - challenges: Optional list of specific challenges to test
                - shuffle: Whether to shuffle test cases
                - max_concurrent: Maximum concurrent evaluations
                - split: Which Mind2Web split to evaluate (CROSS_TASK, CROSS_WEBSITE, CROSS_DOMAIN)
                - domain: Optional domain to filter tasks
                - subdomain: Optional subdomain to filter tasks
                - max_attempts: Maximum number of submission attempts
                - max_messages: Maximum number of messages before giving up
                - code_timeout: Timeout for code execution in seconds
        """
        super().__init__(config)
        self.split = Mind2WebSplit(config.get("split", "test_domain"))
        self.domain = config.get("domain")
        self.subdomain = config.get("subdomain")
        self.max_attempts = config.get("max_attempts", 1)
        self.max_messages = config.get("max_messages", 100)
        self.code_timeout = config.get("code_timeout", 180)
        self.tasks: List[Mind2WebTask] = []
        self.scores_data = None
        
    def load_dataset(self) -> None:
        """Load the Mind2Web dataset."""
        # Get the data directory path
        data_dir = Path("data") / "mind2web"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset if required
        if not os.path.exists(data_dir / "multimodal_mind2web"):
            try:
                snapshot_download(
                    repo_id="osunlp/Multimodal-Mind2Web",
                    repo_type="dataset",
                    local_dir=data_dir / "multimodal_mind2web",
                )
            except Exception as ex:
                raise ex
        
        # Download scores if required
        scores_path = data_dir / "scores.pkl"
        if not os.path.exists(scores_path):
            try:
                scores_url = "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EZllMua3lABAhXQnCN7-pr4BIP4YV8xPfbgyP5FXT18wag?e=yXkK8k&download=1"
                response = requests.get(scores_url, allow_redirects=True)
                if response.status_code == 200:
                    if b"<!DOCTYPE html>" in response.content[:20]:
                        # Parse the HTML to find the download link
                        soup = BeautifulSoup(response.content, "html.parser")
                        download_link = soup.find("a", {"data-download-url": True})
                        if download_link:
                            download_url = download_link["data-download-url"]
                            response = requests.get(download_url, allow_redirects=True)
                            if response.status_code == 200:
                                with open(scores_path, "wb") as f:
                                    f.write(response.content)
                            else:
                                raise Exception("Failed to download from extracted URL")
                        else:
                            raise Exception("Could not find download URL in SharePoint page")
                    else:
                        with open(scores_path, "wb") as f:
                            f.write(response.content)
                else:
                    raise Exception(f"Failed to download scores from SharePoint")
            except Exception as ex:
                raise ex
                
        # Load scores
        import pickle
        with open(scores_path, "rb") as f:
            self.scores_data = pickle.load(f)
            
        # Load dataset from HuggingFace
        from datasets import load_dataset
        dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=self.split.value)
        
        # Convert dataset to tasks
        self.tasks = []
        for record in dataset:
            # Skip if domain doesn't match
            if self.domain and record.get("domain") != self.domain:
                continue
                
            # Skip if subdomain doesn't match
            if self.subdomain and record.get("subdomain") != self.subdomain:
                continue
                
            # Format sample
            sample_id = f"{record['annotation_id']}_{record['action_uid']}"
            
            # Get previous actions
            target_index = int(record.get("target_action_index", "0"))
            prev_actions = self._get_previous_actions(record["action_reprs"], target_index)
            prev_actions_text = "\n".join(prev_actions[-5:]) if prev_actions else "None"
            
            # Process candidates for choices
            candidates = []
            candidates_text = []
            letter_index = 0
            
            # Always add "None of the above" option
            letter = chr(65 + letter_index)  # A
            candidates_text.append(f"({letter}) None of the above")
            candidates.append({
                "letter": letter,
                "action": None,
                "value": None,
                "is_correct": False,
            })
            letter_index += 1
            
            # Process positive candidates
            correct_letter = None
            correct_action = None
            correct_value = None
            
            for pos in record.get("pos_candidates", []):
                result = self._parse_candidate_json(pos)
                letter = chr(65 + letter_index)
                
                action_info = result["data"]
                action_type = action_info.get("action", {}).get("type", "CLICK")
                action_value = action_info.get("action", {}).get("value", "")
                
                choice_text = f"({letter}) {action_type}"
                if action_value and action_type in ["TYPE", "SELECT"]:
                    choice_text += f" with value: {action_value}"
                
                candidates_text.append(choice_text)
                candidates.append({
                    "letter": letter,
                    "action": action_type,
                    "value": action_value,
                    "is_correct": True,
                })
                
                # Save the first correct option as the answer
                if correct_letter is None:
                    correct_letter = letter
                    correct_action = action_type
                    correct_value = action_value
                    
                letter_index += 1
            
            # Process negative candidates
            for neg in record.get("neg_candidates", [])[:4]:  # Limit to 4 negative candidates
                result = self._parse_candidate_json(neg)
                letter = chr(65 + letter_index)
                
                action_info = result["data"]
                action_type = action_info.get("action", {}).get("type", "CLICK")
                action_value = action_info.get("action", {}).get("value", "")
                
                choice_text = f"({letter}) {action_type}"
                if action_value and action_type in ["TYPE", "SELECT"]:
                    choice_text += f" with value: {action_value}"
                
                candidates_text.append(choice_text)
                candidates.append({
                    "letter": letter,
                    "action": action_type,
                    "value": action_value,
                    "is_correct": False,
                })
                letter_index += 1
            
            # If no positive candidates, answer is "None of the above"
            if correct_letter is None:
                answer = Answer(letter="A", action=None, value=None)
            else:
                answer = Answer(letter=correct_letter, action=correct_action, value=correct_value)
            
            task = Mind2WebTask(
                task_id=record.get("task_id", ""),
                annotation_id=record["annotation_id"],
                action_uid=record["action_uid"],
                confirmed_task=record.get("confirmed_task", ""),
                final_html=record.get("html", "")[:200000],  # Truncate to avoid too long HTML
                previous_actions=prev_actions_text,
                choices_text="\n".join(candidates_text),
                target_idx=target_index,
                choices=candidates,
                answer=answer,
                domain=record.get("domain", ""),
                website=record.get("website", ""),
                subdomain=record.get("subdomain", ""),
                website_url=record.get("website_url", ""),
                metadata=record
            )
            self.tasks.append(task)
            
        # Shuffle if configured
        if self.config.get("shuffle", False):
            import random
            random.shuffle(self.tasks)
    
    def _get_previous_actions(self, action_reprs: List[str], target_index: int) -> List[str]:
        """Get previous actions up to the target index."""
        return action_reprs[:target_index]
    
    def _parse_candidate_json(self, json_str: str) -> Dict[str, Any]:
        """Parse a candidate JSON string and extract node ID."""
        # Fix JSON string formatting
        json_str = json_str.replace('" "', '", "')
        data = json.loads(json_str)
        
        # Try to get backend_node_id from attributes if it exists
        if "attributes" in data:
            attrs_str = data["attributes"]
            attrs = json.loads(attrs_str)
            node_id = attrs.get("backend_node_id")
        else:
            node_id = data.get("backend_node_id")
            
        return {"node_id": node_id, "data": data}
        
    def _extract_answer(self, completion: str) -> Answer:
        """Extract the answer from the model's completion."""
        # Clean up the completion
        completion = completion.strip()
        lines = [line.strip() for line in completion.split("\n")]
        completion = "\n".join(line for line in lines if line)
        
        # Try to extract letter
        letter_pattern = r"Answer:\s*(?:[\"'\[\(]|\bOption\b\s*)*([A-F])(?:\.|\)|\]|\"|'|\s|$)"
        letter_match = re.search(letter_pattern, completion, re.IGNORECASE)
        
        # If no letter found, try a more aggressive search
        if not letter_match:
            letter_match = re.search(
                r"(?:^|\s)([A-F])(?:\.|\)|\]|\"|'|\s|$)(?!\S)", completion, re.IGNORECASE
            )
            
        # If still no letter found, return empty Answer
        if not letter_match:
            return Answer(letter="", action=None, value=None)
            
        letter = letter_match.group(1).upper()
        
        # For "A" (None of the above), we don't need action/value
        if letter == "A":
            return Answer(letter=letter, action=None, value=None)
            
        action_match = re.search(r"Action:\s*([^\n]+)", completion, re.IGNORECASE)
        value_match = re.search(r"Value:\s*([^\n]+)", completion, re.IGNORECASE)
        
        action = None
        value = None
        
        # Add action if found
        if action_match:
            action_text = action_match.group(1).strip().upper()
            # Convert to ActionType if valid
            if action_text in ["CLICK", "SELECT", "TYPE"]:
                action = action_text
                
                # Add value if found and action is TYPE or SELECT
                if value_match and (action in ["TYPE", "SELECT"]):
                    value = value_match.group(1).strip()
                # For CLICK, we don't include value
                elif action == "CLICK":
                    pass
                # For other actions, include value if present
                elif value_match:
                    value = value_match.group(1).strip()
                    
        return Answer(letter=letter, action=action, value=value)
    
    def _calculate_text_similarity(self, pred: str, label: str) -> float:
        """Calculate F1 score between predicted and target action strings."""
        if not pred or not label:
            return 0.0 if bool(pred) != bool(label) else 1.0
            
        pred_words = set(pred.strip().split())
        label_words = set(label.strip().split())
        
        # Remove punctuation
        pred_words = set([x for x in pred_words if x not in string.punctuation])
        label_words = set([x for x in label_words if x not in string.punctuation])
        
        if len(pred_words) == 0 and len(label_words) == 0:
            return 1.0
        if len(pred_words) == 0 or len(label_words) == 0:
            return 0.0
            
        tp = len(pred_words & label_words)
        fp = len(pred_words - label_words)
        fn = len(label_words - pred_words)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    def _score_answer(self, pred_answer: Answer, target_answer: Answer) -> Tuple[float, float]:
        """Score an answer against the target.
        
        Returns:
            Tuple containing:
            - element_acc: 1.0 if correct element selected, 0.0 otherwise
            - action_f1: F1 score for action and value prediction
        """
        # Check if the element is correctly identified
        element_acc = 1.0 if pred_answer.letter == target_answer.letter else 0.0
        
        # For "None of the above" (option A), we only need to check the letter
        if target_answer.letter == "A" or pred_answer.letter == "A":
            action_f1 = 1.0 if pred_answer.letter == target_answer.letter else 0.0
            return element_acc, action_f1
            
        # Check action
        if pred_answer.action != target_answer.action:
            action_f1 = 0.0
            return element_acc, action_f1
            
        # For CLICK, we don't need to check value
        if target_answer.action == "CLICK":
            action_f1 = 1.0
            return element_acc, action_f1
            
        # Check value similarity for TYPE and SELECT
        if target_answer.action in ["TYPE", "SELECT"]:
            pred_value = pred_answer.value or ""
            target_value = target_answer.value or ""
            action_f1 = self._calculate_text_similarity(pred_value, target_value)
            return element_acc, action_f1
            
        # Default case
        action_f1 = 1.0 if pred_answer.action == target_answer.action else 0.0
        return element_acc, action_f1
        
    async def evaluate_task(self, task: Mind2WebTask) -> Dict[str, Any]:
        """Evaluate a single Mind2Web task.
        
        Args:
            task: The Mind2Web task to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        # Build prompt
        formatted_prompt = self.TASK_PROMPT.format(
            final_html=task.final_html,
            confirmed_task=task.confirmed_task,
            previous_actions=task.previous_actions,
            choices_text=task.choices_text
        )
        
        # Generate response
        response = await self._generate_response(
            prompt=formatted_prompt,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Extract answer from response
        pred_answer = self._extract_answer(response)
        
        # Score the answer
        element_acc, action_f1 = self._score_answer(pred_answer, task.answer)
        
        # Calculate step success (both element and action correct)
        step_success = float(element_acc == 1.0 and action_f1 == 1.0)
        
        return {
            "task_id": task.task_id,
            "annotation_id": task.annotation_id,
            "action_uid": task.action_uid,
            "confirmed_task": task.confirmed_task,
            "generated_answer": response,
            "extracted_answer": {
                "letter": pred_answer.letter,
                "action": pred_answer.action,
                "value": pred_answer.value
            },
            "target_answer": {
                "letter": task.answer.letter,
                "action": task.answer.action,
                "value": task.answer.value
            },
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "domain": task.domain,
            "website": task.website,
            "subdomain": task.subdomain,
            "website_url": task.website_url,
            "metadata": {
                "element_acc": element_acc,
                "action_f1": action_f1,
                "task_id": task.task_id
            }
        }
        
    async def run(self) -> Dict[str, Any]:
        """Run the Mind2Web test.
        
        Returns:
            Dict containing test results
        """
        # Load dataset
        self.load_dataset()
        
        # Select test cases
        num_cases = min(self.config.get("num_test_cases", len(self.tasks)), len(self.tasks))
        test_cases = self.tasks[:num_cases]
        
        # Evaluate tasks
        results = []
        for task in tqdm(test_cases, desc="Evaluating Mind2Web tasks"):
            result = await self.evaluate_task(task)
            results.append(result)
            
        # Calculate overall metrics
        element_acc = np.mean([r["element_acc"] for r in results])
        action_f1 = np.mean([r["action_f1"] for r in results])
        step_success_rate = np.mean([r["step_success"] for r in results])
        
        # Calculate task success rate (grouped by task_id)
        tasks_dict = defaultdict(list)
        for r in results:
            tasks_dict[r["task_id"]].append(r["step_success"])
            
        successful_tasks = sum(all(steps) for steps in tasks_dict.values())
        total_tasks = len(tasks_dict)
        task_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate domain-specific metrics
        domain_metrics = {}
        domains = set(r["domain"] for r in results)
        for domain in domains:
            domain_results = [r for r in results if r["domain"] == domain]
            domain_metrics[domain] = {
                "element_acc": np.mean([r["element_acc"] for r in domain_results]),
                "action_f1": np.mean([r["action_f1"] for r in domain_results]),
                "step_success_rate": np.mean([r["step_success"] for r in domain_results]),
                "count": len(domain_results)
            }
            
        # Calculate website-specific metrics
        website_metrics = {}
        websites = set((r["domain"], r["website"]) for r in results)
        for domain, website in websites:
            website_results = [r for r in results if r["domain"] == domain and r["website"] == website]
            website_metrics[f"{domain}/{website}"] = {
                "element_acc": np.mean([r["element_acc"] for r in website_results]),
                "action_f1": np.mean([r["action_f1"] for r in website_results]),
                "step_success_rate": np.mean([r["step_success"] for r in website_results]),
                "count": len(website_results)
            }
        
        return {
            "total_steps": len(results),
            "total_tasks": total_tasks,
            "metrics": {
                "element_acc": element_acc,
                "action_f1": action_f1,
                "step_success_rate": step_success_rate,
                "task_success_rate": task_success_rate
            },
            "domain_metrics": domain_metrics,
            "website_metrics": website_metrics,
            "results": results
        } 