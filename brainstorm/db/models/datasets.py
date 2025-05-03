import json
import uuid
from enum import Enum

from pydantic import validator, field_validator
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from sqlalchemy.orm import relationship

from brainstorm.db.base import Base
from sqlalchemy import Enum as SQLEnum


class DatasetModality(str, Enum):
    """
    Model modalities, commented out ones may not be implemented yet
    """
    NLP = "NLP"

class DatasetAdapter(str, Enum):
    """
    Providers. Commented ones may not be implemented yet
    """
    TruthfulQA = "TruthfulQA"
    MuSR = "MuSR"
    BBQ = "BBQ"


class DatasetDefinition(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "dataset_definitions"

    dataset_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    # Model type information
    modality = Column(SQLEnum(DatasetModality, native_enum=False), nullable=False)
    dataset_adapter = Column(SQLEnum(DatasetAdapter, native_enum=False), nullable=False)
    sample_size = Column(Integer, nullable=True)

    # Configuration
    parameters = Column(JSON, nullable=True, default={})

    eval_definitions = relationship("EvalDefinition", back_populates="dataset_definition")
