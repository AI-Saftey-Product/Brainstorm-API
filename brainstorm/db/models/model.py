import uuid
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from brainstorm.db.base import Base


class ModelDefinitionDataModel(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "model_definitions"

    model_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    
    # Model type information
    modality = Column(String, nullable=False)  # NLP, Vision, etc.
    sub_type = Column(String, nullable=False)  # Text Generation, Image Classification, etc.
    provider = Column(String, nullable=False)
    provider_model = Column(String, nullable=False)

    # Connection information
    endpoint_url = Column(String, nullable=False)
    api_key = Column(String, nullable=False)  # For now just the key on its own
    
    # Configuration
    parameters = Column(JSON, nullable=False, default={})
