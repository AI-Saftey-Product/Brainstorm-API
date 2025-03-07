"""API endpoints for test management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.schemas.test import (
    TestRunCreate, TestRunResponse, TestRunList, 
    TestResultResponse, TestResultList, TestInfo,
    TestCategoriesResponse, TestsResponse, TestRegistryResponse,
    ModelSpecificTestsResponse
)
from app.services import test_service


router = APIRouter()


@router.get("/registry")
async def get_test_registry(
    modality: Optional[str] = Query(None, description="Filter tests by modality (e.g., 'NLP', 'Vision')"),
    model_type: Optional[str] = Query(None, description="Filter tests by model type/sub-type (e.g., 'Text Generation', 'Question Answering')"),
    category: Optional[str] = Query(None, description="Filter tests by category (e.g., 'bias', 'toxicity', 'robustness')"),
    include_config: bool = Query(False, description="Include default configuration for each test")
):
    """
    Get the test registry, optionally filtered by modality, model type, and category.
    
    - **modality**: Filter tests by modality (e.g., 'NLP', 'Vision')
    - **model_type**: Filter tests by model type/sub-type (e.g., 'Text Generation', 'Question Answering')
    - **category**: Filter tests by category (e.g., 'bias', 'toxicity', 'robustness')
    - **include_config**: Whether to include default configuration in the response
    
    The frontend should call this endpoint with both modality and model_type to get
    tests specifically applicable to a particular model.
    """
    try:
        tests = await test_service.get_available_tests(modality, model_type, category)
        
        # If include_config is False, remove the default_config and parameter_schema from the response
        if not include_config:
            for test_id in tests:
                tests[test_id].pop("default_config", None)
                tests[test_id].pop("parameter_schema", None)
        
        return {
            "tests": tests, 
            "count": len(tests),
            "filters": {
                "modality": modality,
                "model_type": model_type,
                "category": category
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving test registry: {str(e)}")


@router.get("/model-tests")
async def get_model_specific_tests(
    modality: str = Query(..., description="Model modality (e.g., 'NLP', 'Vision')"),
    model_type: str = Query(..., description="Model type (e.g., 'Text Generation', 'Question Answering')"),
    include_config: bool = Query(True, description="Include default configuration for each test"),
    category: Optional[str] = Query(None, description="Filter by test category (e.g., 'bias', 'toxicity', 'robustness')")
):
    """
    Get tests specifically applicable to a model based on its modality and type.
    
    This endpoint is optimized for frontend use to retrieve only the tests that
    are relevant for a specific model being tested.
    
    All parameters except category are required:
    - **modality**: Model modality (e.g., 'NLP', 'Vision')
    - **model_type**: Model type (e.g., 'Text Generation', 'Question Answering')
    - **include_config**: Whether to include default configuration in the response
    - **category**: Optional filter by test category
    """
    try:
        # Get tests that match all specified criteria
        tests = await test_service.get_available_tests(modality, model_type, category)
            
        # If include_config is False, remove the default_config from the response
        if not include_config:
            for test_id in tests:
                tests[test_id].pop("default_config", None)
                tests[test_id].pop("parameter_schema", None)
        
        return {
            "tests": tests,
            "count": len(tests),
            "model_info": {
                "modality": modality,
                "type": model_type,
                "category_filter": category
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model-specific tests: {str(e)}")


@router.post("/run", status_code=201)
async def create_test_run(
    test_run_data: TestRunCreate
):
    """
    Create a new test run.
    """
    try:
        test_run = await test_service.create_test_run(test_run_data)
        
        # Format the response with the expected structure for the frontend
        return {
            "task_id": str(test_run["id"]),
            "status": test_run["status"],
            "message": "Test run created successfully",
            "test_run": test_run
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating test run: {str(e)}")


@router.get("/runs")
async def get_test_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    """
    Get a list of test runs.
    """
    test_runs = await test_service.get_test_runs(skip, limit)
    return {"test_runs": test_runs, "count": len(test_runs)}


@router.get("/runs/{test_run_id}")
async def get_test_run(
    test_run_id: UUID
):
    """
    Get a test run by ID.
    """
    test_run = await test_service.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
    return test_run


@router.get("/runs/{test_run_id}/results")
async def get_test_results(
    test_run_id: UUID
):
    """
    Get test results for a test run.
    """
    # First check if the test run exists
    test_run = await test_service.get_test_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
    
    results = await test_service.get_test_results(test_run_id)
    return {"results": results, "count": len(results)}


@router.get("/results/{test_run_id}")
async def get_test_results_alias(
    test_run_id: UUID
):
    """
    Get test results for a test run (alias for frontend compatibility).
    """
    return await get_test_results(test_run_id)


@router.get("/categories")
async def get_test_categories():
    """
    Get all available test categories.
    
    This endpoint returns a list of all test categories available in the system,
    which can be used for filtering tests.
    """
    try:
        # Get all tests from the registry
        all_tests = await test_service.get_available_tests()
        
        # Extract unique categories
        categories = set()
        for test_id, test in all_tests.items():
            if "category" in test:
                categories.add(test["category"])
        
        # Convert to list and sort
        category_list = sorted(list(categories))
        
        return {
            "categories": category_list,
            "count": len(category_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving test categories: {str(e)}")


@router.get("/")
async def get_all_tests(
    include_config: bool = Query(False, description="Include default configuration for each test")
):
    """
    Get all available tests without any filtering.
    
    This is a convenience endpoint for the frontend to get all tests at once.
    """
    try:
        tests = await test_service.get_available_tests()
        
        # If include_config is False, remove the default_config and parameter_schema from the response
        if not include_config:
            for test_id in tests:
                tests[test_id].pop("default_config", None)
                tests[test_id].pop("parameter_schema", None)
        
        return {
            "tests": tests,
            "count": len(tests)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tests: {str(e)}")


@router.get("/status/{test_run_id}")
async def get_test_status(
    test_run_id: UUID
):
    """
    Simple placeholder for test status that returns a minimal response.
    """
    # Return a minimal static response without doing any real work
    return {
        "task_id": str(test_run_id),
        "status": "completed",
        "progress": 100,
        "message": "Status check disabled",
        "results_count": 0,
        "initialization_complete": True
    }


@router.get("/debug/runs")
async def debug_get_all_test_runs():
    """
    Debug endpoint to get all test runs in memory.
    This helps diagnose issues with test execution.
    """
    from app.services.test_service import test_runs
    
    # Create a safe copy of test runs to avoid serialization issues
    safe_runs = {}
    for run_id, run in test_runs.items():
        try:
            safe_run = {
                "id": str(run.get("id", "")),
                "model_id": str(run.get("model_id", "")),
                "test_ids": run.get("test_ids", []),
                "status": run.get("status", "unknown"),
                "start_time": str(run.get("start_time", "")),
                "end_time": str(run.get("end_time", "")),
                "created_at": str(run.get("created_at", "")),
                "updated_at": str(run.get("updated_at", "")),
                "summary_results": run.get("summary_results", {})
            }
            safe_runs[str(run_id)] = safe_run
        except Exception as e:
            safe_runs[str(run_id)] = {"error": f"Failed to serialize: {str(e)}"}
    
    return {
        "count": len(safe_runs),
        "test_runs": safe_runs
    } 