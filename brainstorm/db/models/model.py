import uuid
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from brainstorm.db.base import Base


class Model(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    
    # Model type information
    modality = Column(String, nullable=False)  # NLP, Vision, etc.
    sub_type = Column(String, nullable=False)  # Text Generation, Image Classification, etc.
    
    # Connection information
    endpoint_url = Column(String, nullable=True)
    api_key_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to securely stored key
    
    # Configuration
    default_parameters = Column(JSON, nullable=False, default={})
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True) 