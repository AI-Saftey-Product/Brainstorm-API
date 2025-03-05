"""API endpoints for test management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.schemas.test import TestRunCreate, TestRunResponse, TestRunList, TestResultResponse, TestResultList, TestInfo
from app.services import test_service


router = APIRouter()


@router.get("/registry", response_model=Dict[str, Any])
async def get_test_registry(
    modality: Optional[str] = None,
    sub_type: Optional[str] = None
):
    """
    Get the test registry, optionally filtered by modality and sub-type.
    """
    try:
        tests = await test_service.get_available_tests(modality, sub_type)
        return {"tests": tests, "count": len(tests)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving test registry: {str(e)}")


@router.post("/runs", response_model=TestRunResponse, status_code=201)
async def create_test_run(
    test_run_data: TestRunCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new test run.
    """
    try:
        db_test_run = await test_service.create_test_run(db, test_run_data)
        return db_test_run
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating test run: {str(e)}")


@router.get("/runs", response_model=TestRunList)
async def get_test_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get a list of test runs.
    """
    test_runs = await test_service.get_test_runs(db, skip, limit)
    return {"test_runs": test_runs, "count": len(test_runs)}


@router.get("/runs/{test_run_id}", response_model=TestRunResponse)
async def get_test_run(
    test_run_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a test run by ID.
    """
    db_test_run = await test_service.get_test_run(db, test_run_id)
    if not db_test_run:
        raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
    return db_test_run


@router.get("/runs/{test_run_id}/results", response_model=TestResultList)
async def get_test_results(
    test_run_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get test results for a test run.
    """
    # First check if the test run exists
    db_test_run = await test_service.get_test_run(db, test_run_id)
    if not db_test_run:
        raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
    
    results = await test_service.get_test_results(db, test_run_id)
    return {"results": results, "count": len(results)} 