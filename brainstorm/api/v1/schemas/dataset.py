import json
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, Json, field_serializer
from enum import Enum

from brainstorm.db.models.model import ModelModality, ModelSubType, ModelProvider


class PydanticDatasetDefinition(BaseModel):
    """API schema for Model. Includes validation. Whether it should be 1:1 as data model is TBD."""
    dataset_id: str
    name: str
    description: Optional[str] = None

    modality: str  # todo: add validation
    dataset_adapter: str  # todo: add validation
    sample_size: Optional[int] = None

    parameters: Optional[Dict[str, Any]] = {}

    class Config:
        orm_mode = True

    @field_validator('parameters', mode='before')
    @classmethod
    def parse_json_if_needed(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        return v


PydanticDatasetDefinition.model_rebuild()
