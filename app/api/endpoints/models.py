"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.schemas.model import (
    ModelCreate, ModelUpdate, ModelResponse, ModelList,
    ModelModality, NLPModelType
)
from app.services import model_service


router = APIRouter()


@router.post("/", response_model=ModelResponse, status_code=201)
async def create_model(
    model_data: ModelCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new model.
    """
    try:
        db_model = await model_service.create_model(db, model_data)
        return db_model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")


@router.get("/", response_model=ModelList)
async def get_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get a list of models.
    """
    models = await model_service.get_models(db, skip, limit)
    return {"models": models, "count": len(models)}


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a model by ID.
    """
    db_model = await model_service.get_model(db, model_id)
    if not db_model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    return db_model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: UUID,
    model_data: ModelUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a model.
    """
    try:
        db_model = await model_service.update_model(db, model_id, model_data)
        if not db_model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        return db_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a model.
    """
    try:
        success = await model_service.delete_model(db, model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


@router.post("/{model_id}/validate", status_code=200)
async def validate_model_connection(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Validate the connection to a model.
    """
    try:
        is_valid = await model_service.validate_model_connection(model_id, db)
        return {"valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating model connection: {str(e)}")


@router.get("/modalities")
async def get_model_modalities():
    """
    Get all available model modalities.
    
    This endpoint returns a list of all modalities supported by the system,
    such as NLP, Vision, etc.
    """
    try:
        # Hard-coded response for now
        return {
            "modalities": ["NLP"],
            "modality_info": {
                "NLP": {
                    "name": "Natural Language Processing", 
                    "description": "Models that process and generate text",
                    "sub_types": [
                        "Text Generation",
                        "Text2Text Generation",
                        "Question Answering",
                        "Text Classification",
                        "Zero-Shot Classification",
                        "Summarization"
                    ]
                }
            },
            "count": 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model modalities: {str(e)}") 