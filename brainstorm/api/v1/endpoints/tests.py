"""API endpoints for test management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
import traceback
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from brainstorm.core.services import test_service
from brainstorm.db.models.model import ModelDefinition
from brainstorm.db.models.test_run import TestRun

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
        logger.exception(f"Error retrieving model-specific tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model-specific tests: {str(e)}")


@router.post("/run", status_code=201)
async def create_test_run(
    test_run_data: TestRunCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new test run.
    
    Expects a request body with the following structure:
    ```json
    {
      "test_run_id": "uuid-from-websocket-connection",
      "test_ids": ["prompt_injection_test"],
      "model_id": "model_id",
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
        stmt = select(ModelDefinition).filter_by(model_id=test_run_data.model_id)
        model_definition = db.execute(stmt).scalars().first()

        test_run = await create_run(
            test_run_data=test_run_data,
            model_definition=model_definition
        )

        test_run_results = {
            "task_id": str(test_run["id"]),
            "status": test_run["status"],
            "message": "Test run created successfully",
            "test_run": test_run
        }

        db_test_run = TestRun(
            test_run_id=test_run_data.test_run_id,
            test_ids=test_run_data.test_ids,
            test_parameters=test_run_data.parameters,
            test_run_results=test_run_results,
            model_id=test_run_data.model_id,
        )
        db.add(db_test_run)
        db.commit()

        # Format the response with the expected structure for the frontend
        return test_run_results
    except ValueError as e:
        logger.exception()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error creating test run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating test run: {str(e)}")


@router.get("/get_test_runs")
async def get_test_runs(
        test_run_id: Optional[str] = None,
        db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(TestRun)

    if test_run_id:
        stmt = stmt.filter_by(test_run_id=test_run_id)

    test_runs = db.execute(stmt).scalars().all()
    return test_runs
