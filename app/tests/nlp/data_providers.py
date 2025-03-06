"""Data providers for NLP tests to replace hardcoded examples."""

import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

import datasets
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Base class for dataset providers that supply test inputs."""
    
    @abstractmethod
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples for a specific task type."""
        pass

class HuggingFaceDataProvider(DataProvider):
    """Data provider that uses Hugging Face datasets."""
    
    TASK_TO_DATASET_MAPPING = {
        "text_classification": [
            ("imdb", "train", lambda x: {"text": x["text"], "expected": x["label"]}),
            ("sst2", "validation", lambda x: {"text": x["sentence"], "expected": x["label"]}),
            ("emotion", "train", lambda x: {"text": x["text"], "expected": x["label"]})
        ],
        "question_answering": [
            ("squad", "validation", lambda x: {
                "text": {"question": x["question"], "context": x["context"]}, 
                "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""
            }),
            ("adversarial_qa", "validation", lambda x: {
                "text": {"question": x["question"], "context": x["context"]}, 
                "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""
            })
        ],
        "summarization": [
            ("cnn_dailymail", "validation", lambda x: {"text": x["article"], "expected": x["highlights"]}),
            ("samsum", "train", lambda x: {"text": x["dialogue"], "expected": x["summary"]}),
        ],
        "generation": [
            ("wikihow", "test", lambda x: {"text": x["text"], "expected": x["headline"]}),
            ("eli5", "train", lambda x: {"text": x["question"], "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""})
        ]
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the provider with optional cache directory."""
        self.cache_dir = cache_dir
        self.loaded_datasets = {}  # Cache datasets after loading
    
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples for a specific task from HuggingFace datasets."""
        if task_type not in self.TASK_TO_DATASET_MAPPING:
            logger.warning(f"Task type {task_type} not mapped to datasets. Using default examples.")
            return []
            
        examples = []
        dataset_options = self.TASK_TO_DATASET_MAPPING[task_type]
        
        # Try datasets until we have enough examples
        for dataset_name, split, mapper in dataset_options:
            if len(examples) >= n_examples:
                break
                
            try:
                dataset_key = f"{dataset_name}_{split}"
                if dataset_key not in self.loaded_datasets:
                    logger.info(f"Loading dataset {dataset_name} ({split})")
                    self.loaded_datasets[dataset_key] = load_dataset(
                        dataset_name, split=split, cache_dir=self.cache_dir
                    )
                
                dataset = self.loaded_datasets[dataset_key]
                
                # Get random examples
                indices = random.sample(range(len(dataset)), min(n_examples * 2, len(dataset)))
                for idx in indices:
                    if len(examples) >= n_examples:
                        break
                    try:
                        item = dataset[idx]
                        example = mapper(item)
                        examples.append(example)
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Error mapping example: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                continue
                
        return examples

class AdversarialGLUEProvider(DataProvider):
    """Provider for AdvGLUE dataset focused on adversarial examples."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize provider."""
        self.cache_dir = cache_dir
        try:
            self.adv_dataset = load_dataset("adversarial_glue", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load AdvGLUE dataset: {e}")
            self.adv_dataset = None
    
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get adversarial examples from AdvGLUE."""
        if not self.adv_dataset:
            return []
            
        examples = []
        task_mapping = {
            "text_classification": ["sst2", "mnli", "qnli"],
            "question_answering": ["qnli"],
            "nli": ["mnli"]
        }
        
        if task_type not in task_mapping:
            return []
            
        subtasks = task_mapping[task_type]
        
        for subtask in subtasks:
            if len(examples) >= n_examples:
                break
                
            try:
                if subtask not in self.adv_dataset:
                    continue
                    
                adv_data = self.adv_dataset[subtask]
                indices = random.sample(range(len(adv_data)), min(n_examples, len(adv_data)))
                
                for idx in indices:
                    if len(examples) >= n_examples:
                        break
                        
                    item = adv_data[idx]
                    if subtask == "sst2":
                        examples.append({
                            "text": item["sentence"],
                            "expected": item["label"]
                        })
                    elif subtask == "mnli":
                        examples.append({
                            "text": {
                                "premise": item["premise"],
                                "hypothesis": item["hypothesis"]
                            },
                            "expected": item["label"]
                        })
                    elif subtask == "qnli":
                        examples.append({
                            "text": {
                                "question": item["question"],
                                "context": item["sentence"]
                            },
                            "expected": item["label"]
                        })
            except Exception as e:
                logger.warning(f"Error loading examples from {subtask}: {e}")
                
        return examples

class ToxicityDataProvider(DataProvider):
    """Provider for toxicity-focused datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize provider."""
        self.cache_dir = cache_dir
        try:
            # Load RealToxicityPrompts or similar datasets
            self.toxic_dataset = load_dataset("allenai/real-toxicity-prompts", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load toxicity dataset: {e}")
            self.toxic_dataset = None
    
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples focused on potential toxicity issues."""
        if not self.toxic_dataset or task_type not in ["generation", "text_classification"]:
            return []
            
        examples = []
        try:
            # Get prompts with high toxicity scores
            data = self.toxic_dataset["train"]
            # Sort by toxicity score if available
            if "prompt" in data.features and "toxicity" in data.features["prompt"]:
                sorted_indices = sorted(
                    range(len(data)), 
                    key=lambda i: data[i]["prompt"]["toxicity"] if data[i]["prompt"]["toxicity"] is not None else 0,
                    reverse=True
                )
                indices = sorted_indices[:min(n_examples * 3, len(sorted_indices))]
                # Take a random sample from the top toxic examples
                indices = random.sample(indices, min(n_examples, len(indices)))
            else:
                indices = random.sample(range(len(data)), min(n_examples * 2, len(data)))
                
            for idx in indices:
                if len(examples) >= n_examples:
                    break
                    
                item = data[idx]
                try:
                    # Adapt to the expected format
                    examples.append({
                        "text": item["prompt"]["text"],
                        "expected": "non-toxic response" if task_type == "generation" else 0
                    })
                except (KeyError, TypeError):
                    continue
                    
        except Exception as e:
            logger.warning(f"Error extracting toxicity examples: {e}")
            
        return examples

class AugmentedDataProvider(DataProvider):
    """Data provider that applies augmentations to existing examples."""
    
    def __init__(self, base_provider: DataProvider):
        """Initialize with a base provider to augment data from."""
        self.base_provider = base_provider
        
        try:
            import nlpaug.augmenter.word as naw
            import nlpaug.augmenter.sentence as nas
            
            # Word-level augmenters
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
            self.context_aug = naw.ContextualWordEmbsAug(
                model_path='roberta-base', action="substitute"
            )
            
            # Sentence-level augmenters
            self.back_translation_aug = nas.BackTranslationAug(
                from_model_name='Helsinki-NLP/opus-mt-en-de',
                to_model_name='Helsinki-NLP/opus-mt-de-en'
            )
            
        except ImportError:
            logger.warning("nlpaug not available, falling back to base provider")
            self.synonym_aug = None
            self.context_aug = None
            self.back_translation_aug = None
    
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get augmented examples from the base provider."""
        # Get base examples to augment
        base_examples = self.base_provider.get_examples(task_type, n_examples // 2)
        
        if not base_examples or not all([self.synonym_aug, self.context_aug, self.back_translation_aug]):
            return base_examples
            
        augmented_examples = []
        
        for example in base_examples:
            augmented_examples.append(example)  # Keep original
            
            if len(augmented_examples) >= n_examples:
                break
                
            # Create augmented versions
            try:
                if isinstance(example["text"], str):
                    text = example["text"]
                    # Choose a random augmentation method
                    aug_method = random.choice([
                        self.synonym_aug, 
                        self.context_aug, 
                        self.back_translation_aug
                    ])
                    augmented_text = aug_method.augment(text)[0]
                    
                    # Create new example with augmented text
                    augmented_examples.append({
                        "text": augmented_text,
                        "expected": example["expected"]  # Keep same expected output
                    })
                elif isinstance(example["text"], dict):
                    # Handle dict structures like QA pairs
                    aug_text = {}
                    for key, value in example["text"].items():
                        if isinstance(value, str):
                            aug_method = random.choice([self.synonym_aug, self.context_aug])
                            aug_text[key] = aug_method.augment(value)[0]
                        else:
                            aug_text[key] = value
                            
                    augmented_examples.append({
                        "text": aug_text,
                        "expected": example["expected"]
                    })
            except Exception as e:
                logger.warning(f"Error during augmentation: {e}")
                continue
                
            if len(augmented_examples) >= n_examples:
                break
                
        return augmented_examples

class DataProviderFactory:
    """Factory class for creating data providers."""
    
    @staticmethod
    def create(provider_type: str, **kwargs) -> DataProvider:
        """Create a data provider of the specified type."""
        providers = {
            "huggingface": HuggingFaceDataProvider,
            "adversarial_glue": AdversarialGLUEProvider,
            "toxicity": ToxicityDataProvider
        }
        
        if provider_type not in providers:
            logger.warning(f"Unknown provider type: {provider_type}. Using HuggingFace provider.")
            return HuggingFaceDataProvider(**kwargs)
            
        return providers[provider_type](**kwargs)
    
    @staticmethod
    def create_augmented(base_provider_type: str, **kwargs) -> DataProvider:
        """Create an augmented data provider."""
        base_provider = DataProviderFactory.create(base_provider_type, **kwargs)
        return AugmentedDataProvider(base_provider) 