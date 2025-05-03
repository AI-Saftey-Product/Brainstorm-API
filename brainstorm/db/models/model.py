import json
import uuid
from enum import Enum

from pydantic import validator, field_validator
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from sqlalchemy.orm import relationship

from brainstorm.db.base import Base
from sqlalchemy import Enum as SQLEnum


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
    TEXT_GENERATION = "TEXT_GENERATION"
    # TEXT2TEXT_GENERATION = "Text2Text Generation"
    # FILL_MASK = "Fill-Mask"
    # SENTENCE_SIMILARITY = "Sentence Similarity"


class ModelProvider(str, Enum):
    """
    Providers. Commented ones may not be implemented yet
    """
    HUGGINGFACE = "HUGGINGFACE"
    OPENAI = "OPENAI"
    # ANTHROPIC = "Anthropic"
    LLAMA = "LLAMA"
    GCP_MAAS = "GCP_MAAS"


class ModelDefinition(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "model_definitions"

    model_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    # Model type information
    modality = Column(SQLEnum(ModelModality, native_enum=False), nullable=False)
    sub_type = Column(SQLEnum(ModelSubType, native_enum=False), nullable=False)
    provider = Column(SQLEnum(ModelProvider, native_enum=False), nullable=False)
    provider_model = Column(String, nullable=False)

    # Connection information
    endpoint_url = Column(String, nullable=False)
    api_key = Column(String, nullable=False)  # For now just the key on its own

    # Configuration
    parameters = Column(JSON, nullable=True, default={})

    test_runs = relationship("TestRun", back_populates="model_definition")

    eval_definitions = relationship("EvalDefinition", back_populates="model_definition")
