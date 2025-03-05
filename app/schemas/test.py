from typing import Optional, Dict, Any, List, Union, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class TestCategory(str, Enum):
    BIAS = "bias"
    TOXICITY = "toxicity"
    SECURITY = "security"
    HALLUCINATION = "hallucination"
    PRIVACY = "privacy"


class TestInfo(BaseModel):
    """Schema for test information."""
    id: str
    name: str
    description: str
    category: TestCategory
    compatible_modalities: List[str]
    compatible_sub_types: List[str]


class TestRunCreate(BaseModel):
    """Schema for creating a new test run."""
    model_id: UUID
    test_ids: List[str]
    model_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    test_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TestRunResponse(BaseModel):
    """Schema for test run response."""
    id: UUID
    model_id: UUID
    test_ids: List[str]
    status: TestStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    summary_results: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class TestRunList(BaseModel):
    """Schema for listing multiple test runs."""
    test_runs: List[TestRunResponse]
    count: int


class TestResultResponse(BaseModel):
    """Schema for test result response."""
    id: UUID
    test_run_id: UUID
    test_id: str
    test_category: str
    test_name: str
    status: ResultStatus
    score: Optional[float] = None
    metrics: Dict[str, Any]
    prompt: Optional[str] = None
    response: Optional[str] = None
    expected: Optional[str] = None
    issues_found: int
    analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        orm_mode = True


class TestResultList(BaseModel):
    """Schema for listing multiple test results."""
    results: List[TestResultResponse]
    count: int 