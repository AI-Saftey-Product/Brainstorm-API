from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
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


class ModelBase(BaseModel):
    """Base schema for model data."""
    name: str
    description: Optional[str] = None
    modality: ModelModality
    sub_type: str
    
    class Config:
        use_enum_values = True


class ModelCreate(ModelBase):
    """Schema for creating a new model."""
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    default_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('sub_type')
    def validate_sub_type(cls, v, values):
        modality = values.get('modality')
        if modality == ModelModality.NLP and v not in [t.value for t in NLPModelType]:
            raise ValueError(f"Invalid NLP model sub_type: {v}")
        # More validators for other modalities can be added here
        return v


class ModelUpdate(BaseModel):
    """Schema for updating an existing model."""
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint_url: Optional[str] = None
    default_parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ModelResponse(ModelBase):
    """Schema for model response."""
    id: UUID
    endpoint_url: Optional[str] = None
    default_parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        orm_mode = True


class ModelList(BaseModel):
    """Schema for listing multiple models."""
    models: List[ModelResponse]
    count: int 