from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator, model_validator


class NLPBaseParameters(BaseModel):
    """Base parameters for NLP models."""
    model_id: str


class TextGenerationParameters(NLPBaseParameters):
    """Parameters for text generation models."""
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1024, gt=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    stop_sequences: Optional[List[str]] = Field(default_factory=list)


class ClassificationParameters(NLPBaseParameters):
    """Parameters for classification models."""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    multi_label: Optional[bool] = False
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class QuestionAnsweringParameters(NLPBaseParameters):
    """Parameters for question answering models."""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(256, gt=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    context_window: Optional[int] = Field(4096, gt=0)


class SummarizationParameters(NLPBaseParameters):
    """Parameters for summarization models."""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, gt=0)
    min_length: Optional[int] = Field(50, ge=0)
    max_length: Optional[int] = Field(500, gt=0)
    
    @model_validator(mode='after')
    def validate_length(self) -> 'SummarizationParameters':
        """Validate that min_length is less than or equal to max_length."""
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError("min_length must be less than or equal to max_length")
        return self


# Factory function to get the right parameter schema based on model type
def get_nlp_parameter_schema(sub_type: str) -> type:
    """Get the appropriate parameter schema for an NLP model sub-type."""
    schema_map = {
        "Text Generation": TextGenerationParameters,
        "Text2Text Generation": TextGenerationParameters,
        "Question Answering": QuestionAnsweringParameters,
        "Table Question Answering": QuestionAnsweringParameters,
        "Text Classification": ClassificationParameters,
        "Token Classification": ClassificationParameters,
        "Zero-Shot Classification": ClassificationParameters,
        "Summarization": SummarizationParameters,
    }
    
    return schema_map.get(sub_type, NLPBaseParameters) 