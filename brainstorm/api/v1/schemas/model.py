from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator
from enum import Enum


class ModelModality(str, Enum):
    NLP = "NLP"
    VISION = "Vision"
    AUDIO = "Audio"
    MULTIMODAL = "Multimodal"
    TABULAR = "Tabular"


class NLPModelType(str, Enum):
    TEXT_CLASSIFICATION = "Text Classification"
    TOKEN_CLASSIFICATION = "Token Classification"
    TABLE_QUESTION_ANSWERING = "Table Question Answering"
    QUESTION_ANSWERING = "Question Answering"
    ZERO_SHOT_CLASSIFICATION = "Zero-Shot Classification"
    TRANSLATION = "Translation"
    SUMMARIZATION = "Summarization"
    FEATURE_EXTRACTION = "Feature Extraction"
    TEXT_GENERATION = "Text Generation"
    TEXT2TEXT_GENERATION = "Text2Text Generation"
    FILL_MASK = "Fill-Mask"
    SENTENCE_SIMILARITY = "Sentence Similarity"


class ModelProvider(str, Enum):
    """Enum of supported model providers."""
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"  # For generic API endpoints


class ModelBase(BaseModel):
    """Base model schema."""
    name: str
    description: Optional[str] = None
    provider: str
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    target_category: Optional[str] = None
    target_subtype: Optional[str] = None
    target_parameters: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ModelCreate(ModelBase):
    """Schema for creating a new model."""
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    provider_id: Optional[str] = None  # Changed from model_id to provider_id
    target_id: str
    target_type: str
    target_category: str = "NLP"
    target_subtype: str
    target_parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_subtype")
    @classmethod
    def validate_sub_type(cls, v: str) -> str:
        """Validate the target subtype."""
        valid_subtypes = ["Text Generation", "Question Answering", "Text Classification", "Summarization"]
        if v not in valid_subtypes:
            raise ValueError(f"Invalid target subtype. Must be one of: {', '.join(valid_subtypes)}")
        return v

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ModelUpdate(BaseModel):
    """Schema for updating an existing model."""
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint_url: Optional[str] = None
    default_parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ModelResponse(ModelBase):
    """Schema for model response."""
    id: UUID
    endpoint_url: Optional[str] = None
    default_parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ModelList(BaseModel):
    """Schema for listing multiple models."""
    models: List[ModelResponse]
    count: int

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# New schemas for modalities endpoint
class SubTypeInfo(BaseModel):
    """Information about a model sub-type."""
    name: str
    description: Optional[str] = None


class ModalityInfo(BaseModel):
    """Information about a modality."""
    name: str
    description: str
    sub_types: List[str]


class ModalitiesResponse(BaseModel):
    """Response schema for modalities endpoint."""
    modalities: List[str]
    modality_info: Dict[str, ModalityInfo]
    count: int


# New schemas for test categories
class TestCategoriesResponse(BaseModel):
    """Response schema for test categories endpoint."""
    categories: List[str]
    count: int 