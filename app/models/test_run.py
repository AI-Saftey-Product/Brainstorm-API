import uuid
from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime

from app.db.base import Base


class TestRun(Base):
    """Database model for test runs."""
    __tablename__ = "test_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, nullable=False)  # Changed from UUID to String to support frontend model ids
    
    # Tests being run
    test_ids = Column(ARRAY(String), nullable=False)
    
    # Run configuration
    model_parameters = Column(JSON, nullable=False, default={})
    test_parameters = Column(JSON, nullable=False, default={})
    
    # Status information
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    
    # Results (summary)
    summary_results = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow) 