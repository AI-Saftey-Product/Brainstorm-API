"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
import logging
from fastapi import Response
from starlette.responses import StreamingResponse

from brainstorm.api.v1.schemas.eval import PydanticEvalDefinition
from brainstorm.db.base import get_db
from brainstorm.db.models.evals import EvalDefinition
from brainstorm.evals.runner import run_evals

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/run_eval")
async def run_eval(
        eval_id: str,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """

    result_generator = await run_evals(eval_id=eval_id, db=db)

    return StreamingResponse(result_generator(), media_type="application/json")


@router.post("/create_or_update_eval")
async def create_or_update_eval(
        eval_definition: PydanticEvalDefinition,
        db: Session = Depends(get_db)
):
    """
    """
    try:
        # todo: this is basically same as with models - django generates these itself so may there is a way to
        #  generate or at least re-use these generic CRUD endpoints here too?
        logger.info(eval_definition)

        db_model = EvalDefinition(**eval_definition.dict())

        stmt = select(EvalDefinition).filter_by(eval_id=eval_definition.eval_id)
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
        logger.exception("Failed to create or update a eval due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Failed to create or update a eval due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/get_evals", response_model=List[PydanticEvalDefinition])
async def get_evals(
        eval_id: Optional[str] = None,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(EvalDefinition)

    if eval_id:
        stmt = stmt.filter_by(eval_id=eval_id)

    models = db.execute(stmt).scalars().all()
    return models


@router.post("/delete_evals")
async def delete_evals(
        eval_ids: List[str],
        db: Session = Depends(get_db)
):
    """
    Delete models.
    """
    try:
        stmt = delete(EvalDefinition).where(EvalDefinition.eval_id.in_(eval_ids))
        db.execute(stmt)
        db.commit()

        # todo: handle if some (or all) requested model IDs where not found and therefore not deleted

    # todo: this duplicates creat or update code, should refactor or may be there are some constructs for that in the library
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a evals due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=e
        )
    except Exception as e:
        logger.exception("Failed to create or update a evals due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=e
        )

