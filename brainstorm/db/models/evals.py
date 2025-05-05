import json
import uuid
from enum import Enum

from pydantic import validator, field_validator
from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from datetime import datetime

from sqlalchemy.orm import relationship

from brainstorm.db.base import Base
from sqlalchemy import Enum as SQLEnum


class DatasetModality(str, Enum):
    """
    Model modalities, commented out ones may not be implemented yet
    """
    NLP = "NLP"
    # VISION = "Vision"
    # AUDIO = "Audio"
    # MULTIMODAL = "Multimodal"
    # TABULAR = "Tabular"


class DatasetEvaluator(str, Enum):
    """
    Providers. Commented ones may not be implemented yet
    """
    LLM = "LLM"


class DatasetAdapter(str, Enum):
    """
    Providers. Commented ones may not be implemented yet
    """
    TruthfulQA = "TruthfulQA"


class EvalDefinition(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "eval_definitions"

    eval_id = Column(String, primary_key=True)

    dataset_id = Column(String, ForeignKey('dataset_definitions.dataset_id'))
    dataset_definition = relationship("DatasetDefinition", back_populates="eval_definitions")

    model_id = Column(String, ForeignKey('model_definitions.model_id'))
    model_definition = relationship("ModelDefinition", back_populates="eval_definitions")

    scorer = Column(String, nullable=False)
    scorer_agg_functions = Column(JSON, nullable=False)
    scorer_agg_dimensions = Column(JSON, nullable=True)

    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    # Configuration
    parameters = Column(JSON, nullable=True, default={})

    results = Column(JSON, nullable=True, default=[])
