from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, Json
from enum import Enum


class ModelModality(str, Enum):
    """
    Model modalities, commented out ones may not be implemented yet
    """
    NLP = "NLP"
    # VISION = "Vision"
    # AUDIO = "Audio"
    # MULTIMODAL = "Multimodal"
    # TABULAR = "Tabular"


class ModelSubType(str, Enum):
    """
    Model subtypes, commented out ones may not be implemented yet
    """
    # TEXT_CLASSIFICATION = "Text Classification"
    # TOKEN_CLASSIFICATION = "Token Classification"
    # TABLE_QUESTION_ANSWERING = "Table Question Answering"
    # QUESTION_ANSWERING = "Question Answering"
    # ZERO_SHOT_CLASSIFICATION = "Zero-Shot Classification"
    # TRANSLATION = "Translation"
    # SUMMARIZATION = "Summarization"
    # FEATURE_EXTRACTION = "Feature Extraction"
    TEXT_GENERATION = "Text Generation"
    # TEXT2TEXT_GENERATION = "Text2Text Generation"
    # FILL_MASK = "Fill-Mask"
    # SENTENCE_SIMILARITY = "Sentence Similarity"


class ModelProvider(str, Enum):
    """
    Providers. Commented ones may not be implemented yet
    """
    # HUGGINGFACE = "HuggingFace"
    OPENAI = "OpenAI"
    # ANTHROPIC = "Anthropic"


class ModelDefinition(BaseModel):
    """API schema for Model. Includes validation. Whether it should be 1:1 as data model is TBD."""
    model_id: str
    name: str
    description: Optional[str] = None

    modality: ModelModality
    sub_type: ModelSubType
    provider: ModelProvider
    provider_model: str  # todo: we should have some validation as to which provider models do we support

    endpoint_url: str
    api_key: str

    parameters: Json = Field(default_factory=dict)

    # todo: mapping for supported providers vs modalities vs subtypes

    class Config:
        orm_mode = True

ModelDefinition.model_rebuild()
