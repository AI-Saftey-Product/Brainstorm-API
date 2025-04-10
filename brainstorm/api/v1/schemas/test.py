from typing import Optional, Dict, Any, List, Union, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, field_validator, ConfigDict
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


class TestResultResponse(BaseModel):
    """Schema for test result response."""
    id: UUID
    test_run_id: UUID
    input: Dict[str, Any]
    output: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TestRunCreate(BaseModel):
    """Schema for creating a new test run."""
    test_ids: List[str]
    model_settings: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    test_run_id: Optional[str] = None
    
    @field_validator("model_settings")
    @classmethod
    def validate_model_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the model settings."""
        if not v.get("model_id"):
            raise ValueError("model_id is required in model_settings")
        return v
    
    @field_validator("test_ids")
    @classmethod
    def validate_test_ids(cls, v: List[str]) -> List[str]:
        """Validate the test IDs."""
        if not v:
            raise ValueError("At least one test ID is required")
        return v
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class TestRunResponse(BaseModel):
    """Schema for test run response."""
    id: UUID
    target_id: str
    target_type: str
    target_parameters: Dict[str, Any]
    results: List[TestResultResponse]
    created_at: datetime
    updated_at: datetime
    status: str
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class TestRunList(BaseModel):
    """Schema for listing multiple test runs."""
    test_runs: List[TestRunResponse]
    count: int


class TestResultList(BaseModel):
    """Schema for listing multiple test results."""
    results: List[TestResultResponse]
    count: int


class TestCategoriesResponse(BaseModel):
    """Response schema for test categories endpoint."""
    categories: List[str]
    count: int


class TestsResponse(BaseModel):
    """Response schema for tests endpoint."""
    tests: Dict[str, Any]
    count: int


class TestRegistryResponse(BaseModel):
    """Response schema for test registry endpoint."""
    tests: Dict[str, Any]
    count: int
    filters: Dict[str, Optional[str]]


class ModelSpecificTestsResponse(BaseModel):
    """Response schema for model-specific tests endpoint."""
    tests: Dict[str, Any]
    count: int
    target_info: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())