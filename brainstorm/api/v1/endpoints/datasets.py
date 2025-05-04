"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
import logging
from fastapi import Response

from brainstorm.api.v1.schemas.dataset import PydanticDatasetDefinition
from brainstorm.api.v1.schemas.eval import PydanticEvalDefinition
from brainstorm.datasets.initial_datasets import DATASETS_MAP
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


@router.get("/get_dataset_evals", response_model=List[PydanticEvalDefinition])
async def get_dataset_evals(
        dataset_id: str,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(DatasetDefinition)

    if dataset_id:
        stmt = stmt.filter_by(dataset_id=dataset_id)

    model = db.execute(stmt).scalar_one_or_none()

    return model.eval_definitions

@router.get("/get_dataset_suggested_scorers_and_dimensions")
async def get_dataset_suggested_scorers_and_dimensions(
        dataset_id: str,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(DatasetDefinition)

    if dataset_id:
        stmt = stmt.filter_by(dataset_id=dataset_id)

    model = db.execute(stmt).scalar_one_or_none()

    dataset_instance = DATASETS_MAP[model.dataset_adapter]

    # todo: datasets can suggested multiple scorers but currently Eval and runner only support 1 scorer
    return {
        "suggested_scorers": dataset_instance.suggested_scorers[0],
        "suggested_agg_dimensions": dataset_instance.suggested_agg_dimensions,
        "suggested_agg_functions": dataset_instance.suggested_agg_functions,
    }


@router.get("/get_dataset_preview")
async def get_dataset_preview(
        dataset_id: str,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(DatasetDefinition)

    if dataset_id:
        stmt = stmt.filter_by(dataset_id=dataset_id)

    models = db.execute(stmt).scalars().all()

    dataset_instance = DATASETS_MAP[models[0].dataset_adapter](dataset_definition=models[0])

    preview = {}
    for split_key in dataset_instance.keys():
        split_len = len(dataset_instance[split_key])
        rows = dataset_instance[split_key].select(range(min(split_len, 10)))
        preview[split_key] = [dict(row) for row in rows]

    # split -> [rows] where each row is a dict like field: value
    return preview


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

