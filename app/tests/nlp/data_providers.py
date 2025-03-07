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
            ("emotion", "train", lambda x: {"text": x["text"], "expected": x["label"]}),
            # Add more alternatives for text classification
            ("ag_news", "test", lambda x: {"text": x["text"], "expected": x["label"]}),
            ("yelp_review_full", "train", lambda x: {"text": x["text"], "expected": x["label"]}),
            ("banking77", "train", lambda x: {"text": x["text"], "expected": x["label"]}),
            ("silicone/emo", "train", lambda x: {"text": x["text"], "expected": x["emotion"]})
        ],
        "question_answering": [
            ("squad", "validation", lambda x: {
                "text": {"question": x["question"], "context": x["context"]}, 
                "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""
            }),
            ("adversarial_qa", "validation", lambda x: {
                "text": {"question": x["question"], "context": x["context"]}, 
                "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""
            }),
            # Add more alternatives for QA
            ("quoref", "validation", lambda x: {
                "text": {"question": x["question"], "context": x["context"]}, 
                "expected": x["answers"]["text"][0] if x["answers"]["text"] else ""
            }),
            ("coqa", "train", lambda x: {
                "text": {"question": x["questions"][0], "context": x["story"]}, 
                "expected": x["answers"]["input_text"][0] if len(x["answers"]["input_text"]) > 0 else ""
            }),
            ("triviaqa", "train", lambda x: {
                "text": {"question": x["question"], "context": " ".join(x["entity_pages"]["wiki_context"][:2]) if "entity_pages" in x and "wiki_context" in x["entity_pages"] else ""},
                "expected": x["answer"]["value"] if "answer" in x else ""
            })
        ],
        "summarization": [
            ("cnn_dailymail", "validation", lambda x: {"text": x["article"], "expected": x["highlights"]}),
            ("samsum", "train", lambda x: {"text": x["dialogue"], "expected": x["summary"]}),
            # Add more alternatives for summarization
            ("xsum", "train", lambda x: {"text": x["document"], "expected": x["summary"]}),
            ("multi_news", "train", lambda x: {"text": x["document"], "expected": x["summary"]}),
            ("billsum", "train", lambda x: {"text": x["text"], "expected": x["summary"]}),
            ("big_patent", "train", lambda x: {"text": x["description"], "expected": x["abstract"]})
        ],
        "generation": [
            # Replaced defunct datasets with reliable alternatives
            ("cnn_dailymail", "validation", lambda x: {"text": "Write an article about: " + x["highlights"], "expected": x["article"]}),
            ("samsum", "train", lambda x: {"text": "Create a dialogue based on this summary: " + x["summary"], "expected": x["dialogue"]}),
            ("squad", "validation", lambda x: {"text": "Generate a question about: " + x["context"], "expected": x["question"]}),
            ("xsum", "train", lambda x: {"text": "Generate content based on: " + x["summary"], "expected": x["document"]}),
            # Additional generation datasets
            ("wikitext", "train", lambda x: {"text": "Continue this text: " + x["text"][:200], "expected": x["text"][200:400]}),
            ("openai/webgpt_comparisons", "train", lambda x: {"text": x["question"], "expected": x["answer_0"] if random.random() > 0.5 else x["answer_1"]}),
            ("common_gen", "train", lambda x: {"text": "Write a sentence using these concepts: " + ", ".join(x["concepts"]), "expected": x["target"]})
        ]
    }
    
    # Fallback examples for when datasets fail to load
    FALLBACK_EXAMPLES = {
        "text_classification": [
            {"text": "This movie was fantastic and I enjoyed every minute of it.", "expected": 1},
            {"text": "The product was terrible and broke after one use.", "expected": 0},
            {"text": "I had a neutral experience, neither good nor bad.", "expected": 2}
        ],
        "question_answering": [
            {"text": {"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}, "expected": "Paris"},
            {"text": {"question": "Who invented the telephone?", "context": "Alexander Graham Bell is credited with inventing the first practical telephone."}, "expected": "Alexander Graham Bell"}
        ],
        "summarization": [
            {"text": "The quick brown fox jumps over the lazy dog. The dog was too tired to react, while the fox continued on its journey through the forest. At the edge of the forest, the fox encountered a river that it needed to cross to reach its den.", "expected": "A fox jumped over a dog and needed to cross a river to get home."},
            {"text": "Scientists have discovered a new species of deep-sea fish that can survive at extreme depths. The fish has specialized adaptations including pressure-resistant cells and unique vision capabilities that allow it to see in near-total darkness.", "expected": "New deep-sea fish species found with adaptations for extreme depths and darkness."}
        ],
        "generation": [
            {"text": "Write a short story about space exploration.", "expected": "Astronauts discovered a new planet with traces of ancient civilization."},
            {"text": "Describe the process of photosynthesis.", "expected": "Plants convert sunlight into energy through a chemical process in their cells."}
        ]
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the provider with optional cache directory."""
        self.cache_dir = cache_dir
        self.loaded_datasets = {}  # Cache datasets after loading
    
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples for a task type from HuggingFace datasets."""
        # Normalize task type
        task_type = task_type.lower().replace(" ", "_")
        
        if task_type not in self.TASK_TO_DATASET_MAPPING:
            logger.warning(f"No dataset mapping for task type: {task_type}")
            return self._get_fallback_examples(task_type, n_examples)
        
        # Try each dataset until we find one
        datasets_tried = 0
        for dataset_name, split, transform_fn in self.TASK_TO_DATASET_MAPPING[task_type]:
            try:
                logger.info(f"Loading dataset {dataset_name} ({split})")
                # Try with trust_remote_code=True to handle more complex datasets
                # Use additional parameters to help with dataset loading
                dataset = load_dataset(
                    dataset_name, 
                    split=split, 
                    trust_remote_code=True, 
                    use_auth_token=False,  # No token needed for public datasets
                    streaming=False,  # We need random access
                    num_proc=1  # Single process to avoid complex errors
                )
                datasets_tried += 1
                
                # Select random examples
                if isinstance(dataset, Dataset) and len(dataset) > 0:
                    # Ensure we don't request more examples than available
                    n = min(n_examples, len(dataset))
                    
                    # Get random indices
                    indices = random.sample(range(len(dataset)), n)
                    
                    # Transform examples to the expected format
                    examples = []
                    for idx in indices:
                        try:
                            example = transform_fn(dataset[idx])
                            examples.append(example)
                        except Exception as e:
                            logger.warning(f"Error transforming example from {dataset_name}: {e}")
                    
                    if examples:
                        logger.info(f"Successfully loaded {len(examples)} examples from {dataset_name}")
                        return examples
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {str(e)}")
        
        # If we've tried at least one dataset but all failed, use fallbacks
        if datasets_tried > 0:
            logger.warning(f"Failed to load any dataset for task type: {task_type}, using fallback examples")
            return self._get_fallback_examples(task_type, n_examples)
        else:
            # This should not happen unless there's a bug in the dataset mapping
            logger.error(f"No datasets were tried for task type: {task_type}")
            return self._get_fallback_examples(task_type, n_examples)
    
    def _get_fallback_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get fallback examples when datasets cannot be loaded."""
        if task_type not in self.FALLBACK_EXAMPLES:
            logger.warning(f"No fallback examples for task type: {task_type}")
            # Return some generic examples that might work for most tasks
            return [
                {"text": "This is a test example.", "expected": "Test response", "_is_fallback": True}
            ]
        
        # Return the available fallback examples, limited to requested number
        examples = self.FALLBACK_EXAMPLES[task_type]
        
        # Mark all examples as fallbacks
        for example in examples:
            example["_is_fallback"] = True
            
        if len(examples) > n_examples:
            return random.sample(examples, n_examples)
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
            
            # Back translation augmenter (now in word module, not sentence)
            self.back_translation_aug = naw.BackTranslationAug(
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
        
        # Remove parameters that aren't supported by the provider constructors
        kwargs_copy = kwargs.copy()
        if 'use_augmentation' in kwargs_copy:
            # This parameter is only used by create_augmented, not by the actual providers
            kwargs_copy.pop('use_augmentation')
            
        if provider_type not in providers:
            logger.warning(f"Unknown provider type: {provider_type}. Using HuggingFace provider.")
            return HuggingFaceDataProvider(**kwargs_copy)
            
        return providers[provider_type](**kwargs_copy)
    
    @staticmethod
    def create_augmented(base_provider_type: str, **kwargs) -> DataProvider:
        """Create an augmented data provider."""
        use_augmentation = kwargs.get('use_augmentation', False)
        
        # If augmentation is disabled, just return the base provider
        if not use_augmentation:
            base_provider = DataProviderFactory.create(base_provider_type, **kwargs)
            return base_provider
            
        # Otherwise, create the base provider and wrap it with augmentation
        base_provider = DataProviderFactory.create(base_provider_type, **kwargs)
        return AugmentedDataProvider(base_provider) 