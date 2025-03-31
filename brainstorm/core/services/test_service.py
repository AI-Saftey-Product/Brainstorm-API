"""Service for handling test operations."""
from __future__ import annotations

import uuid
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
import asyncio

from brainstorm.core.adapters.nlp_adapter import HuggingFaceNLPAdapter
from brainstorm.core.adapters.base_adapter import ModelAdapter
from brainstorm.core.websocket import manager as websocket_manager
from brainstorm.testing.registry.registry import test_registry
from brainstorm.testing.modalities.nlp.bias.occupational_test import OccupationalBiasTest
from brainstorm.testing.modalities.nlp.prompt_injection_test import PromptInjectionTest
from brainstorm.testing.modalities.nlp.security.jailbreak_test import JailbreakTest
from brainstorm.testing.modalities.nlp.security.data_extraction_test import DataExtractionTest
from brainstorm.testing.modalities.nlp.optimized_robustness_test import OptimizedRobustnessTest
from brainstorm.testing.modalities.nlp.bias.evaluators import (
    IntersectionalEvaluator,
    QAEvaluator
)

logger = logging.getLogger(__name__)

# Keep track of active test tasks to prevent garbage collection
active_test_tasks: Set[asyncio.Task] = set()

# Initialize test registry
REGISTRY = test_registry
TEST_REGISTRY = REGISTRY._tests

# Import all test classes
def _import_test_class(path: str) -> type:
    """Dynamically import a test class from its string path."""
    try:
        logger.debug(f"Attempting to import test class from path: {path}")
        module_path, class_name = path.rsplit('.', 1)
        logger.debug(f"  Module path: {module_path}")
        logger.debug(f"  Class name: {class_name}")
        
        module = __import__(module_path, fromlist=[class_name])
        test_class = getattr(module, class_name)
        logger.debug(f"Successfully imported {class_name}")
        return test_class
    except Exception as e:
        logger.error(f"Failed to import test class {path}: {e}")
        logger.error(traceback.format_exc())
        return None

# Convert string paths to actual classes in registry
logger.info("Initializing test registry...")
logger.debug(f"Found {len(TEST_REGISTRY)} tests in registry")

for test_id, test_info in TEST_REGISTRY.items():
    logger.debug(f"\nProcessing test {test_id}:")
    if isinstance(test_info.get("test_class"), str):
        logger.debug(f"Found string test class path: {test_info['test_class']}")
        test_class = _import_test_class(test_info["test_class"])
        if test_class:
            TEST_REGISTRY[test_id]["test_class"] = test_class
            logger.debug(f"Successfully imported and stored test class")
        else:
            logger.error(f"Failed to import test class for {test_id}")
    else:
        logger.debug(f"Test class is already imported")

logger.info(f"Test registry initialization complete with {len(TEST_REGISTRY)} tests")

def get_test_registry(
    modality: Optional[str] = None,
    model_type: Optional[str] = None,
    category: Optional[str] = None,
    include_config: bool = False
) -> Dict[str, Any]:
    """Get the test registry filtered by modality, model type, and category."""
    filtered_registry = {}
    
    # Get the actual tests dictionary from the registry
    registry_tests = TEST_REGISTRY
    
    logger.debug(f"Filtering registry with modality={modality}, model_type={model_type}, category={category}")
    logger.debug(f"Initial registry has {len(registry_tests)} tests")
    logger.debug(f"Registry keys: {list(registry_tests.keys())}")
    logger.debug(f"First test info: {next(iter(registry_tests.values()))}")
    
    for test_id, test_info in registry_tests.items():
        logger.debug(f"\nChecking test {test_id}:")
        logger.debug(f"Test info: {test_info}")
        
        # Check if modality matches
        if modality:
            compatible_modalities = test_info.get("compatible_modalities", [])
            logger.debug(f"  Compatible modalities: {compatible_modalities}")
            if not compatible_modalities or modality.upper() not in [m.upper() for m in compatible_modalities]:
                logger.debug(f"  Skipping due to modality mismatch")
                continue
                
        # Check if model type matches
        if model_type:
            compatible_sub_types = test_info.get("compatible_sub_types", [])
            logger.debug(f"  Compatible sub types: {compatible_sub_types}")
            if not compatible_sub_types or not any(sub_type.lower() == model_type.lower() for sub_type in compatible_sub_types):
                logger.debug(f"  Skipping due to model type mismatch")
                continue
                
        # Check if category matches
        if category:
            test_category = test_info.get("category", "")
            logger.debug(f"  Test category: {test_category}")
            if test_category.lower() != category.lower():
                logger.debug(f"  Skipping due to category mismatch")
                continue
            
        logger.debug(f"  Test passed all filters")
        filtered_registry[test_id] = test_info
    
    logger.debug(f"Final filtered registry has {len(filtered_registry)} tests")
    logger.debug(f"Filtered test IDs: {list(filtered_registry.keys())}")
    return filtered_registry

def get_model_specific_tests(
    modality: str,
    model_type: str,
    include_config: bool = True,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """Get tests specific to a model's modality and type."""
    registry = get_test_registry(modality, model_type, category)
    
    if not registry:
        return {}
    
    # Define user-friendly names and descriptions
    display_info = {
        "bias_test": {
            "display_name": "Bias Test",
            "description": "Tests for biases in how the model answers questions about different groups."
        },
        "crows_pairs": {
            "display_name": "Stereotype Analysis",
            "description": "Evaluates model behavior with stereotypical vs. anti-stereotypical statements."
        },
        "intersect_bench": {
            "display_name": "Intersectional Benchmark",
            "description": "Comprehensive benchmark for testing intersectional fairness across multiple dimensions."
        }
    }
        
    tests = {}
    for test_id, test_info in registry.items():
        # Import test class if needed
        if isinstance(test_info.get("test_class"), str):
            test_class = _import_test_class(test_info["test_class"])
            if test_class:
                test_info["test_class"] = test_class
            else:
                logger.error(f"Failed to import test class for {test_id}")
                continue
                
        test_class = test_info["test_class"]
        test_category = test_info.get("category", "")
        
        # Get user-friendly display info
        display = display_info.get(test_id, {
            "display_name": test_id.replace("_", " ").title(),
            "description": getattr(test_class, "__doc__", "") or "No description available."
        })
        
        tests[test_id] = {
            "id": test_id,  # Use original test ID consistently
            "name": display["display_name"],  # User-friendly display name
            "category": test_category.title(),  # Capitalize category
            "description": display["description"],
            "modality": modality,
            "model_type": model_type
        }
        
        if include_config:
            try:
                # Handle different types of test classes
                if hasattr(test_class, "get_default_config"):
                    # For BaseTest subclasses
                    test_instance = test_class({})
                    tests[test_id]["default_config"] = test_instance.get_default_config()
                elif test_category == "bias":
                    # For bias evaluators
                    tests[test_id]["default_config"] = {
                        "max_samples": 100,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "test_types": [test_id],
                        "batch_size": 8
                    }
                else:
                    # Default configuration
                    tests[test_id]["default_config"] = {
                        "max_samples": 100,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
            except Exception as e:
                logger.error(f"Error getting default config for {test_id}: {e}")
                tests[test_id]["default_config"] = {}
    
    return tests

def get_all_tests(include_config: bool = False) -> Dict[str, Any]:
    """Get all available tests without any filtering."""
    all_tests = {}
    
    for test_id, test_info in TEST_REGISTRY.items():
        test_class = test_info["test_class"]
        test_modality = test_info.get("modality")
        test_model_type = test_info.get("model_type")
        test_category = test_info.get("category")
        
        test_info = {
            "name": test_id,
            "category": test_category,
            "description": test_class.__doc__ or "",
        }
        if include_config:
            # Add default configuration if available
            try:
                test_instance = test_class({})
                test_info["default_config"] = test_instance.get_default_config()
            except:
                test_info["default_config"] = {}
                
        all_tests[test_id] = test_info
    
    return all_tests

async def get_test_status(test_run_id: str) -> Dict[str, Any]:
    """Get status of a test run."""
    # This would normally query a database
    return {
        "id": test_run_id,
        "status": "unknown",
        "progress": 0,
        "total_tests": 0,
        "completed_tests": 0
    }

async def debug_get_all_test_runs() -> List[Dict[str, Any]]:
    """Debug endpoint to get all test runs in memory."""
    return [{
        "id": str(uuid.uuid4()),
        "status": "debug",
        "active_tasks": len(active_test_tasks)
    }]

async def debug_websocket_connection(test_run_id: str) -> Dict[str, Any]:
    """Debug endpoint to test WebSocket connections."""
    # Check if there are active connections
    active_connections = websocket_manager.active_connections.get(test_run_id, set())
    connection_count = len(active_connections)
    
    # Get last connection time
    last_connection = websocket_manager.last_connection_time.get(test_run_id)
    
    # Send a test message
    test_message = {
        "type": "debug_message",
        "test_run_id": test_run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "connection_info": {
            "active_connections": connection_count,
            "last_connection": last_connection,
            "has_connections": test_run_id in websocket_manager.active_connections
        }
    }
    
    try:
        await websocket_manager.send_notification(test_run_id, test_message)
        message_sent = True
    except Exception as e:
        message_sent = False
        logger.error(f"Error sending debug message: {e}")
    
    return {
        "status": "success",
        "message": f"WebSocket connection test for run {test_run_id}",
        "timestamp": datetime.utcnow().isoformat(),
        "connection_info": {
            "active_connections": connection_count,
            "last_connection": last_connection,
            "has_connections": test_run_id in websocket_manager.active_connections,
            "message_sent": message_sent
        }
    }

async def debug_test_run_status(test_run_id: str) -> Dict[str, Any]:
    """Debug endpoint to check test run status."""
    return {
        "id": test_run_id,
        "status": "debug",
        "active_tasks": len(active_test_tasks),
        "task_details": [str(task) for task in active_test_tasks],
        "timestamp": datetime.utcnow().isoformat()
    }

async def debug_available_tests() -> Dict[str, Any]:
    """Debug endpoint to get all available test IDs."""
    available_tests = {}
    
    for test_id, test_info in TEST_REGISTRY.items():
        test_class = test_info["test_class"]
        test_modality = test_info.get("modality")
        test_model_type = test_info.get("model_type")
        test_category = test_info.get("category")
        
        available_tests[test_id] = {
            "modality": test_modality,
            "model_type": test_model_type,
            "category": test_category,
            "description": test_class.__doc__ or "",
            "class": test_class.__name__
        }
    
    return available_tests

# Initialize test data providers
bias_data_provider = OccupationalBiasTest(config={"test_type": "bias", "test_category": "bias"})

async def run_bias_test(
    model_adapter: ModelAdapter,
    test_id: str,
    test_parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run a specific bias test."""
    try:
        # Configure the test
        config = {
            "max_samples": test_parameters.get("max_samples", 100),
            "temperature": test_parameters.get("temperature", 0.7),
            "top_p": test_parameters.get("top_p", 0.9)
        }
        
        # Create and run test with OccupationalBiasTest as default
        test = OccupationalBiasTest(config)
        results = await test.run_test(model_adapter, test_parameters or {})
        
        return {
            "id": str(uuid.uuid4()),
            "test_run_id": None,
            "test_id": test_id,
            "test_category": "bias",
            "test_name": "occupational_bias",
            "status": "success",
            "score": results.get("score", 0),
            "metrics": results.get("metrics", {}),
            "issues_found": results.get("issues_found", 0),
            "analysis": results.get("analysis", {}),
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running bias test: {str(e)}")
        return {
            "id": str(uuid.uuid4()),
            "test_run_id": None,
            "test_id": test_id,
            "test_category": "bias",
            "test_name": "occupational_bias",
            "status": "error",
            "score": 0,
            "metrics": {},
            "issues_found": 1,
            "analysis": {"error": str(e)},
            "created_at": datetime.utcnow().isoformat()
        }

from pydantic import BaseModel

class TestRunCreate(BaseModel):
    test_run_id: Optional[str]
    test_ids: List[str]
    model_settings: Dict[str, Any]
    parameters: Optional[Dict[str, Any]]

def _serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects in a dictionary or list."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj

async def create_test_run(test_run_data: TestRunCreate) -> Dict[str, Any]:
    """Create a new test run."""
    try:
        # Generate test run ID if not provided
        test_run_id = test_run_data.test_run_id or str(uuid.uuid4())
        
        # Initialize test run
        test_run = {
            "id": test_run_id,
            "status": "running",
            "progress": 0,
            "total_tests": len(test_run_data.test_ids),
            "completed_tests": 0,
            "failed_tests": 0,
            "results": [],
            "created_at": datetime.utcnow().isoformat(),
            "model_settings": test_run_data.model_settings,
            "parameters": test_run_data.parameters or {}
        }
        
        # Create model adapter based on settings
        model_settings = test_run_data.model_settings
        if model_settings["modality"].upper() == "NLP":
            # Always use HuggingFace adapter
            model_adapter = HuggingFaceNLPAdapter(model_settings)
            await model_adapter.initialize(model_settings)
        else:
            raise ValueError(f"Unsupported modality: {model_settings['modality']}")
        
        # Track failed tests
        failed_tests = 0
        
        # Execute tests asynchronously
        for test_id in test_run_data.test_ids:
            try:
                # Get test info directly from registry
                test_info = test_registry.get_test(test_id)
                
                if not test_info:
                    error_msg = f"Test not found: {test_id}"
                    logger.error(error_msg)
                    test_run["results"].append({
                        "id": str(uuid.uuid4()),
                        "test_run_id": test_run_id,
                        "test_id": test_id,
                        "status": "error",
                        "error": error_msg,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    failed_tests += 1
                    continue
                
                # Create test instance
                test_class = test_info["test_class"]
                if isinstance(test_class, str):
                    test_class = _import_test_class(test_class)
                
                if not test_class:
                    error_msg = f"Failed to import test class for {test_id}"
                    logger.error(error_msg)
                    test_run["results"].append({
                        "id": str(uuid.uuid4()),
                        "test_run_id": test_run_id,
                        "test_id": test_id,
                        "status": "error",
                        "error": error_msg,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    failed_tests += 1
                    continue
                
                # Initialize test with configuration
                test_config = {
                    **test_info.get("default_config", {}),
                    **test_run_data.parameters.get(test_id, {}),
                    "websocket_manager": websocket_manager,
                    "test_id": test_id
                }
                
                test_instance = test_class(test_config)
                
                # Send test start notification
                start_message = {
                    "type": "test_started",
                    "test_run_id": test_run_id,
                    "test_id": test_id,
                    "test_name": test_info["name"],
                    "test_category": test_info["category"],
                    "status": "running",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket_manager.send_notification(test_run_id, start_message)
                
                # Run test with both model_adapter and model_parameters
                result = await test_instance.run_test(model_adapter, test_run_data.model_settings)
                
                # Serialize datetime objects in the result
                serialized_result = _serialize_datetime(result)
                
                # Add result to test run
                test_run["results"].append({
                    "id": str(uuid.uuid4()),
                    "test_run_id": test_run_id,
                    "test_id": test_id,
                    "test_category": test_info["category"],
                    "test_name": test_info["name"],
                    "status": "completed",
                    "result": serialized_result,
                    "created_at": datetime.utcnow().isoformat()
                })
                
                test_run["completed_tests"] += 1
                if result.get("passed", False):
                    test_run["completed_tests"] += 1
                else:
                    failed_tests += 1
                
                # Update progress
                test_run["progress"] = (test_run["completed_tests"] / test_run["total_tests"]) * 100
                
                # Send progress update
                progress_message = {
                    "type": "test_progress",
                    "test_run_id": test_run_id,
                    "progress": test_run["progress"],
                    "completed_tests": test_run["completed_tests"],
                    "total_tests": test_run["total_tests"],
                    "current_test": test_id,
                    "status": "running"
                }
                await websocket_manager.send_notification(test_run_id, progress_message)
                
            except Exception as e:
                error_msg = f"Error running test {test_id}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                test_run["results"].append({
                    "id": str(uuid.uuid4()),
                    "test_run_id": test_run_id,
                    "test_id": test_id,
                    "test_category": test_info.get("category", "unknown"),
                    "test_name": test_info.get("name", test_id),
                    "status": "error",
                    "error": error_msg,
                    "created_at": datetime.utcnow().isoformat()
                })
                
                failed_tests += 1
                test_run["completed_tests"] += 1
                
                # Update progress
                test_run["progress"] = (test_run["completed_tests"] / test_run["total_tests"]) * 100
        
        # Update final status and send completion message
        test_run["status"] = "completed"
        test_run["updated_at"] = datetime.utcnow().isoformat()
        
        # Format results to match frontend expectations
        formatted_results = {}
        
        for result in test_run.get("results", []):
            try:
                if result.get("status") == "completed":
                    test_id = result.get("test_id")
                    if not test_id:
                        logger.warning(f"Skipping result with missing test_id: {result}")
                        continue
                    
                    # Extract test metadata
                    test_info_from_registry = test_registry.get_test_info(test_id) or {}
                    
                    formatted_results[test_id] = {
                        "test": {
                            "id": test_id,
                            "name": result.get("test_name", "Unknown Test"),
                            "category": result.get("test_category", "unknown"),
                            "description": test_info_from_registry.get("description", ""),
                            "severity": test_info_from_registry.get("severity", "medium")
                        },
                        "result": {
                            "pass": result.get("result", {}).get("issues_found", 0) == 0,
                            "score": result.get("result", {}).get("score", 0),
                            "message": f"Completed {result.get('test_name', 'test')} with score {result.get('result', {}).get('score', 0)}",
                            "details": result.get("result", {}).get("analysis", {}),
                            "issues_found": result.get("result", {}).get("issues_found", 0),
                            "metrics": result.get("result", {}).get("metrics", {}),
                            "timestamp": result.get("created_at", datetime.utcnow().isoformat())
                        }
                    }
            except Exception as e:
                logger.error(f"Error formatting result {result.get('test_id', 'unknown')}: {str(e)}")
        
        # Create completion message with formatted results
        completion_message = {
            "type": "test_complete",
            "test_run_id": test_run_id,
            "status": "completed",
            "results": {
                "total_tests": test_run["total_tests"],
                "completed_tests": test_run["completed_tests"],
                "failed_tests": failed_tests,
                "test_results": formatted_results
            }
        }
        
        # Send completion message with formatted results
        await websocket_manager.send_notification(test_run_id, completion_message)
        
        logger.info(f"Completed test run {test_run_id} with {len(formatted_results)} formatted test results")
        
        # Serialize the test run before returning
        serialized_test_run = _serialize_datetime(test_run)
        return serialized_test_run
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error creating test run: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Create and serialize error message
        error_message = {
            "type": "test_failed",
            "error": error_msg,
            "details": {
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Send serialized error message
        await websocket_manager.send_notification(test_run_id, _serialize_datetime(error_message))
        
        raise

async def get_test_runs(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Get a list of test runs."""
    # This would normally query a database
    return []

async def get_test_run(test_run_id: str) -> Optional[Dict[str, Any]]:
    """Get a test run by ID."""
    # This would normally query a database
    return None

async def get_test_results(test_run_id: str) -> List[Dict[str, Any]]:
    """Get test results for a test run."""
    # This would normally query a database
    return []

def get_test_categories() -> List[str]:
    """Get all available test categories."""
    categories = set()
    
    for test_info in TEST_REGISTRY.items():
        categories.add(test_info[1].get("category"))
    
    return sorted(list(categories))