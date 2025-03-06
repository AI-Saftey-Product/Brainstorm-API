"""Service for handling test operations."""
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import asyncio

from app.models.test_run import TestRun
from app.models.test_result import TestResult
from app.schemas.test import TestRunCreate
from app.test_registry import test_registry
from app.model_adapters import get_model_adapter
from app.services.model_service import get_model
from app.tests.nlp.adversarial_robustness_test import AdversarialRobustnessTest


logger = logging.getLogger(__name__)


async def create_test_run(db: Session, test_run_data: TestRunCreate) -> TestRun:
    """
    Create a new test run.
    
    Args:
        db: Database session
        test_run_data: Test run creation data
        
    Returns:
        The created test run
    """
    try:
        # Validate model exists
        model = await get_model(db, test_run_data.model_id)
        if not model:
            raise ValueError(f"Model with ID {test_run_data.model_id} not found")
        
        # Validate tests exist
        for test_id in test_run_data.test_ids:
            test = test_registry.get_test(test_id)
            if not test:
                raise ValueError(f"Test with ID {test_id} not found")
            
            # Check if test is compatible with model
            if model.modality not in test["compatible_modalities"]:
                raise ValueError(f"Test {test_id} is not compatible with modality {model.modality}")
            
            if model.sub_type not in test["compatible_sub_types"]:
                raise ValueError(f"Test {test_id} is not compatible with model sub-type {model.sub_type}")
        
        # Create test run
        db_test_run = TestRun(
            model_id=test_run_data.model_id,
            test_ids=test_run_data.test_ids,
            model_parameters=test_run_data.model_parameters or {},
            test_parameters=test_run_data.test_parameters or {},
            status="pending"
        )
        
        db.add(db_test_run)
        db.commit()
        db.refresh(db_test_run)
        
        # Start test run asynchronously
        # In a production environment, this would be handled by a task queue
        asyncio.create_task(run_tests(db_test_run.id, db))
        
        return db_test_run
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating test run: {str(e)}")
        raise


async def get_test_run(db: Session, test_run_id: UUID) -> Optional[TestRun]:
    """
    Get a test run by ID.
    
    Args:
        db: Database session
        test_run_id: Test run ID
        
    Returns:
        The test run if found, None otherwise
    """
    return db.query(TestRun).filter(TestRun.id == test_run_id).first()


async def get_test_runs(db: Session, skip: int = 0, limit: int = 100) -> List[TestRun]:
    """
    Get a list of test runs.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of test runs
    """
    return db.query(TestRun).order_by(TestRun.created_at.desc()).offset(skip).limit(limit).all()


async def get_test_results(db: Session, test_run_id: UUID) -> List[TestResult]:
    """
    Get test results for a test run.
    
    Args:
        db: Database session
        test_run_id: Test run ID
        
    Returns:
        List of test results
    """
    return db.query(TestResult).filter(TestResult.test_run_id == test_run_id).all()


async def run_tests(test_run_id: UUID, db: Session) -> None:
    """
    Run tests for a test run.
    
    Args:
        test_run_id: Test run ID
        db: Database session
    """
    try:
        # Get test run
        db_test_run = await get_test_run(db, test_run_id)
        if not db_test_run:
            logger.error(f"Test run with ID {test_run_id} not found")
            return
        
        # Update status to running
        db_test_run.status = "running"
        db_test_run.start_time = datetime.utcnow()
        db.commit()
        
        # Get model
        model = await get_model(db, db_test_run.model_id)
        if not model:
            logger.error(f"Model with ID {db_test_run.model_id} not found")
            db_test_run.status = "failed"
            db.commit()
            return
        
        # Get model adapter
        model_config = {
            "id": str(model.id),
            "modality": model.modality,
            "sub_type": model.sub_type,
            "endpoint_url": model.endpoint_url,
            # In a real implementation, you would securely retrieve the API key
            "api_key": None  # This would be retrieved from a secure storage
        }
        
        adapter = get_model_adapter(model.modality, model_config)
        await adapter.initialize(model_config)
        
        # Merge model parameters
        model_parameters = {**model.default_parameters, **db_test_run.model_parameters}
        
        # Run each test
        results = []
        summary = {
            "total_tests": len(db_test_run.test_ids),
            "completed": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        for test_id in db_test_run.test_ids:
            try:
                # Get test details
                test = test_registry.get_test(test_id)
                if not test:
                    logger.error(f"Test with ID {test_id} not found")
                    continue
                
                # Get test parameters for this test
                test_parameters = db_test_run.test_parameters.get(test_id, {})
                
                # Run the appropriate test based on test_id
                if test_id == "nlp_adversarial_robustness_test":
                    # Run adversarial robustness test
                    test_result = await run_adversarial_robustness_test(
                        model_adapter=adapter,
                        test_id=test_id,
                        test_category=test["category"],
                        test_name=test["name"],
                        model_parameters=model_parameters,
                        test_parameters=test_parameters
                    )
                else:
                    # For other tests, use placeholder as before
                    test_result = TestResult(
                        test_run_id=test_run_id,
                        test_id=test_id,
                        test_category=test["category"],
                        test_name=test["name"],
                        status="success",  # Placeholder
                        score=0.95,  # Placeholder
                        metrics={"accuracy": 0.95},  # Placeholder
                        prompt="Test prompt",  # Placeholder
                        response="Test response",  # Placeholder
                        issues_found=0  # Placeholder
                    )
                
                db.add(test_result)
                db.commit()
                
                # Update summary
                summary["completed"] += 1
                summary["passed"] += 1
                
            except Exception as e:
                logger.error(f"Error running test {test_id}: {str(e)}")
                
                # Create error result
                test_result = TestResult(
                    test_run_id=test_run_id,
                    test_id=test_id,
                    test_category=test["category"] if test else "unknown",
                    test_name=test["name"] if test else "Unknown Test",
                    status="error",
                    metrics={"error": str(e)},
                    issues_found=1
                )
                
                db.add(test_result)
                db.commit()
                
                # Update summary
                summary["completed"] += 1
                summary["errors"] += 1
        
        # Update test run status
        db_test_run.status = "completed"
        db_test_run.end_time = datetime.utcnow()
        db_test_run.summary_results = summary
        db.commit()
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        
        # Update test run status
        try:
            db_test_run = await get_test_run(db, test_run_id)
            if db_test_run:
                db_test_run.status = "failed"
                db_test_run.end_time = datetime.utcnow()
                db.commit()
        except Exception as inner_e:
            logger.error(f"Error updating test run status: {str(inner_e)}")


async def get_available_tests(
    modality: Optional[str] = None, 
    model_type: Optional[str] = None, 
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get available tests, optionally filtered by modality, model type, and category.
    
    Args:
        modality: Optional modality filter (e.g., 'NLP', 'Vision')
        model_type: Optional model type filter (e.g., 'Text Generation', 'Question Answering')
        category: Optional category filter (e.g., 'bias', 'toxicity', 'robustness')
        
    Returns:
        Dictionary of available tests that match the specified filters
    """
    # First get tests filtered by modality and model type
    if modality and model_type:
        tests = test_registry.get_tests_by_sub_type(modality, model_type)
    elif modality:
        tests = test_registry.get_tests_by_modality(modality)
    else:
        tests = test_registry.get_all_tests()
    
    # Then filter by category if specified
    if category and tests:
        return {
            test_id: test for test_id, test in tests.items()
            if test.get("category") == category
        }
        
    return tests


async def run_adversarial_robustness_test(
    model_adapter: Any,
    test_id: str,
    test_category: str,
    test_name: str,
    model_parameters: Dict[str, Any],
    test_parameters: Dict[str, Any]
) -> TestResult:
    """
    Run the adversarial robustness test.
    
    Args:
        model_adapter: Model adapter
        test_id: Test ID
        test_category: Test category
        test_name: Test name
        model_parameters: Model parameters
        test_parameters: Test parameters
        
    Returns:
        Test result
    """
    try:
        # Initialize the test
        test = AdversarialRobustnessTest(test_parameters.get("config"))
        
        # Prepare parameters
        parameters = {
            "model_type": model_adapter.model_type,
            "model_parameters": model_parameters,
            "test_inputs": test_parameters.get("test_inputs")
        }
        
        # Run the test
        results = await test.run(model_adapter, parameters)
        
        # Extract metrics
        robustness_score = results.get("robustness_score", 0)
        
        # Create test result
        test_result = TestResult(
            test_id=test_id,
            test_category=test_category,
            test_name=test_name,
            status="success" if robustness_score >= 0.5 else "failure",
            score=robustness_score,
            metrics=results,
            # Include some examples in the results
            prompt=str(results.get("attack_examples", [])[:2]),
            response=str(results.get("interpretation", {})),
            issues_found=len([a for a in results.get("attack_results", []) 
                             if a.get("success_rate", 0) > 0.5])
        )
        
        return test_result
    
    except Exception as e:
        logger.error(f"Error running adversarial robustness test: {str(e)}")
        
        # Create error result
        return TestResult(
            test_id=test_id,
            test_category=test_category,
            test_name=test_name,
            status="error",
            metrics={"error": str(e)},
            issues_found=1
        ) 