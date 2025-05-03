"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
import logging
from fastapi import Response

from brainstorm.api.v1.schemas.dataset import PydanticDatasetDefinition
from brainstorm.db.base import get_db
from brainstorm.db.models.datasets import DatasetDefinition

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/create_or_update_dataset")
async def create_or_update_model(
        dataset_definition: PydanticDatasetDefinition,
        db: Session = Depends(get_db)
):
    """
    """
    try:
        # todo: this is basically same as with models - django generates these itself so may there is a way to
        #  generate or at least re-use these generic CRUD endpoints here too?
        logger.info(dataset_definition)

        db_model = DatasetDefinition(**dataset_definition.dict())

        stmt = select(DatasetDefinition).filter_by(dataset_id=dataset_definition.dataset_id)
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
        logger.exception("Failed to create or update a dataset due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Failed to create or update a dataset due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/get_datasets", response_model=List[PydanticDatasetDefinition])
async def get_datasets(
        dataset_id: Optional[str] = None,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(DatasetDefinition)

    if dataset_id:
        stmt = stmt.filter_by(dataset_id=dataset_id)

    models = db.execute(stmt).scalars().all()
    return models


@router.post("/delete_datasets")
async def delete_datasets(
        dataset_ids: List[str],
        db: Session = Depends(get_db)
):
    """
    Delete models.
    """
    try:
        stmt = delete(DatasetDefinition).where(DatasetDefinition.dataset_id.in_(dataset_ids))
        db.execute(stmt)
        db.commit()

        # todo: handle if some (or all) requested model IDs where not found and therefore not deleted

    # todo: this duplicates creat or update code, should refactor or may be there are some constructs for that in the library
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a datasets due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=e
        )
    except Exception as e:
        logger.exception("Failed to create or update a datasets due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=e
        )

