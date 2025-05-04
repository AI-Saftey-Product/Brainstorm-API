import json
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, Json, field_serializer
from enum import Enum

from brainstorm.db.models.model import ModelModality, ModelSubType, ModelProvider


class PydanticEvalDefinition(BaseModel):
    """API schema for Model. Includes validation. Whether it should be 1:1 as data model is TBD."""
    eval_id: str
    dataset_id: str
    model_id: str

    scorer: str
    scorer_agg_functions: List[str] = []
    scorer_agg_dimensions: List[str] = []

    name: str
    description: Optional[str] = None

    parameters: Optional[Dict[str, Any]] = {}
    results: Optional[List[Dict[str, Any]]] = None

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


PydanticEvalDefinition.model_rebuild()
