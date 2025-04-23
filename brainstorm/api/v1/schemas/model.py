import json
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, Json, field_serializer
from enum import Enum

from brainstorm.db.models.model import ModelModality, ModelSubType, ModelProvider


class PydanticModelDefinition(BaseModel):
    """API schema for Model. Includes validation. Whether it should be 1:1 as data model is TBD."""
    model_id: str
    name: str
    description: Optional[str] = None

    modality: ModelModality
    sub_type: ModelSubType
    provider: ModelProvider
    provider_model: str  # todo: we should have some validation as to which provider models do we support

    endpoint_url: str
    api_key: str

    parameters: Optional[Dict[str, Any]] = {}

    # todo: mapping for supported providers vs modalities vs subtypes

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


PydanticModelDefinition.model_rebuild()
