import uuid
from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime

from sqlalchemy.orm import relationship

from brainstorm.db.base import Base

class TestRun(Base):
    """Database model for AI models registered in the system."""
    __tablename__ = "test_runs"

    test_run_id = Column(String, primary_key=True)
    test_ids = Column(ARRAY(String))
    test_parameters = Column(JSON, nullable=True, default={})
    test_run_results = Column(JSON, nullable=True, default={})

    # model_id = Column(String, ForeignKey('model_definitions.model_id'))
    # model_definition = relationship("ModelDefinition", back_populates="test_runs")
