import uuid
from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey, Float, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from app.db.base import Base


class TestResult(Base):
    """Database model for individual test results."""
    __tablename__ = "test_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_run_id = Column(UUID(as_uuid=True), ForeignKey("test_runs.id"), nullable=False)
    test_id = Column(String, nullable=False)
    
    # Test information
    test_category = Column(String, nullable=False)  # bias, toxicity, etc.
    test_name = Column(String, nullable=False)
    
    # Result data
    status = Column(String, nullable=False)  # success, failure, error
    score = Column(Float, nullable=True)  # Overall score if applicable
    metrics = Column(JSON, nullable=False, default={})  # Detailed metrics
    
    # Test inputs/outputs
    prompt = Column(Text, nullable=True)  # Test input/prompt
    response = Column(Text, nullable=True)  # Model response
    expected = Column(Text, nullable=True)  # Expected or baseline response
    
    # Analysis
    issues_found = Column(Integer, nullable=False, default=0)
    analysis = Column(JSON, nullable=True)  # Detailed analysis
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow) 