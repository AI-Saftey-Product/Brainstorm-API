"""API endpoints for test management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
import traceback
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from brainstorm.core.services import test_service

logger = logging.getLogger(__name__)

from brainstorm.db.base import get_db
from brainstorm.api.v1.schemas.test import (
    TestRunCreate, TestRunResponse, TestRunList, 
    TestResultResponse, TestResultList, TestInfo,
    TestCategoriesResponse, TestsResponse, TestRegistryResponse,
    ModelSpecificTestsResponse
)
from brainstorm.core.services.test_service import (
    get_test_registry as get_registry,
    get_model_specific_tests as get_model_tests,
    create_test_run as create_run,
    get_test_runs as get_runs,
    get_test_run as get_run,
    get_test_results as get_results,
    get_test_categories as get_categories,
    get_all_tests as get_tests
)
from brainstorm.testing.registry import test_registry


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
        logger.debug(f"Getting test registry with filters:")
        logger.debug(f"  modality: {modality}")
        logger.debug(f"  model_type: {model_type}")
        logger.debug(f"  category: {category}")
        logger.debug(f"  include_config: {include_config}")
        
        tests = await get_registry(modality, model_type, category)
        logger.debug(f"Found {len(tests)} tests in registry")
        logger.debug(f"Test IDs: {list(tests.keys())}")
        
        # If include_config is False, remove the default_config and parameter_schema from the response
        if not include_config:
            logger.debug("Removing config and schema from response")
            for test_id in tests:
                tests[test_id].pop("default_config", None)
                tests[test_id].pop("parameter_schema", None)
        
        response = {
            "tests": tests, 
            "count": len(tests),
            "filters": {
                "modality": modality,
                "model_type": model_type,
                "category": category
            }
        }
        
        logger.debug(f"Returning response with {len(tests)} tests")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving test registry: {str(e)}")
        logger.error(traceback.format_exc())
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
        # Log request parameters
        logger.debug(f"Getting model-specific tests with params:")
        logger.debug(f"  modality: {modality}")
        logger.debug(f"  model_type: {model_type}")
        logger.debug(f"  category: {category}")
        logger.debug(f"  include_config: {include_config}")
        
        # Convert inputs to lowercase for registry lookup
        modality_key = modality.lower()
        model_type_key = model_type.lower()  # Keep spaces in model type
        category_key = category.lower() if category else None
        
        logger.debug(f"Converted keys:")
        logger.debug(f"  modality_key: {modality_key}")
        logger.debug(f"  model_type_key: {model_type_key}")
        logger.debug(f"  category_key: {category_key}")
        
        # Get tests that match all specified criteria
        tests = test_service.get_model_specific_tests(
            modality=modality_key,
            model_type=model_type_key,
            include_config=include_config,
            category=category_key
        )
        
        logger.debug(f"Found {len(tests)} matching tests")
        logger.debug(f"Test IDs: {list(tests.keys())}")
        
        response = {
            "tests": tests,
            "count": len(tests),
            "model_info": {
                "modality": modality,  # Return original case for display
                "type": model_type,    # Return original case for display
                "category_filter": category
            }
        }
        
        logger.debug(f"Returning response with {len(tests)} tests")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving model-specific tests: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving model-specific tests: {str(e)}")


@router.post("/run", status_code=201)
async def create_test_run(
    test_run_data: TestRunCreate
):
    """
    Create a new test run.
    
    Expects a request body with the following structure:
    ```json
    {
      "test_run_id": "uuid-from-websocket-connection",
      "test_ids": ["prompt_injection_test"],
      "model_settings": {
        "model_id": "gpt2",
        "modality": "NLP",
        "sub_type": "Text Generation", 
        "source": "huggingface",
        "api_key": "your-api-key"
      },
      "parameters": {
        "prompt_injection_test": {
          "max_samples": 100
        }
      }
    }
    ```
    
    - `test_run_id`: Required field containing the UUID received from WebSocket connection
    - `test_ids`: Required array of test IDs to run
    - `model_settings`: Required object with model configuration
      - `model_id`: Required field identifying the model
      - Other fields are optional but recommended
    - `parameters`: Optional object with test-specific parameters
    
    Important: You MUST connect to the WebSocket endpoint at `/ws/tests` first to obtain
    a test_run_id, then provide that ID when creating a test run.
    """
    try:
        # Check if test_run_id is provided
        if not test_run_data.test_run_id:
            raise ValueError("test_run_id is required. Connect to WebSocket endpoint at /ws/tests first to obtain an ID.")
            
        test_run = await create_run(test_run_data)
        
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
    test_runs = await get_runs(skip, limit)
    return {"test_runs": test_runs, "count": len(test_runs)}


@router.get("/runs/{test_run_id}")
async def get_test_run(
    test_run_id: UUID
):
    """
    Get a test run by ID.
    """
    test_run = await get_run(test_run_id)
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
    test_run = await get_run(test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
    
    results = await get_results(test_run_id)
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
        all_tests = await get_registry()
        
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
        tests = await get_tests()
        
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
    from brainstorm.core.services.test_service import debug_get_all_test_runs
    
    return await debug_get_all_test_runs()


@router.get("/debug/websocket/{test_run_id}")
async def debug_websocket_connection(test_run_id: str):
    """
    Debug endpoint to test WebSocket connections.
    Sends a test notification to all clients connected to a test run.
    """
    from brainstorm.core.services.test_service import debug_websocket_connection
    
    return await debug_websocket_connection(test_run_id)


@router.get("/debug/run-status/{test_run_id}")
async def debug_test_run_status(test_run_id: str):
    """
    Debug endpoint to check the status of a test run and its execution.
    """
    from brainstorm.core.services.test_service import debug_test_run_status
    
    return await debug_test_run_status(test_run_id)


@router.get("/debug/available-tests")
async def debug_available_tests():
    """
    Debug endpoint to get all available test IDs and their details.
    This helps developers see what test IDs are available for testing.
    """
    from brainstorm.core.services.test_service import debug_available_tests as get_debug_tests
    
    return await get_debug_tests() 