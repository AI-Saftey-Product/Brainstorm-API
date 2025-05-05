"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
import asyncio
import logging
from fastapi import Response

from fastapi.responses import JSONResponse

from brainstorm.api.v1.schemas.eval import PydanticEvalDefinition
from brainstorm.db.base import get_db
from brainstorm.api.v1.schemas.model import (
    PydanticModelDefinition
)
from brainstorm.core.adapters import get_model_adapter
from brainstorm.db.models.model import ModelDefinition

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/validate_model")
async def validate_model(
        model_id: str,
        db: Session = Depends(get_db)
):
    stmt = select(ModelDefinition).filter_by(model_id=model_id)
    model_definition = db.execute(stmt).scalars().first()

    adapter = get_model_adapter(model_definition)

    try:
        await adapter.generate("Hi")
        return Response(
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=e
        )


@router.post("/create_or_update_model")
async def create_or_update_model(
        model_definition: PydanticModelDefinition,
        db: Session = Depends(get_db)
):
    """
    Create or update a model definition.

    Currently:
    - We have parity between SQL model and API model
    - We always receive full model definition
    - We let frontend generate unique model IDs
    - If model with a given ID already exists we simply update it with definition we received
    """
    try:
        logger.info(model_definition)

        db_model = ModelDefinition(**model_definition.dict())

        stmt = select(ModelDefinition).filter_by(model_id=model_definition.model_id)
        existing = db.execute(stmt).scalar_one_or_none()

        if not existing:
            db.add(db_model)
        else:
            db.merge(db_model)

        db.commit()
        return Response(
            status_code=200,
        )
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a model due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Failed to create or update a model due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/get_models", response_model=List[PydanticModelDefinition])
async def get_models(
        model_id: Optional[str] = None,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(ModelDefinition)

    if model_id:
        stmt = stmt.filter_by(model_id=model_id)

    models = db.execute(stmt).scalars().all()
    return models


@router.get("/get_model_evals", response_model=List[PydanticEvalDefinition])
async def get_model_evals(
        model_id: str,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(ModelDefinition)

    if model_id:
        stmt = stmt.filter_by(model_id=model_id)

    model = db.execute(stmt).scalar_one_or_none()

    return model.eval_definitions


@router.post("/delete_models")
async def delete_models(
        model_ids: List[str],
        db: Session = Depends(get_db)
):
    """
    Delete models.
    """
    try:
        stmt = delete(ModelDefinition).where(ModelDefinition.model_id.in_(model_ids))
        db.execute(stmt)
        db.commit()

        # todo: handle if some (or all) requested model IDs where not found and therefore not deleted

    # todo: this duplicates creat or update code, should refactor or may be there are some constructs for that in the library
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a model due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=e
        )
    except Exception as e:
        logger.exception("Failed to create or update a model due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=e
        )

