"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import asyncio
import logging

from app.db.base import get_db
from app.schemas.model import (
    ModelCreate, ModelUpdate, ModelResponse, ModelList,
    ModelModality, NLPModelType
)
from app.services import model_service
from app.model_adapters import get_model_adapter


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=ModelResponse, status_code=201)
async def create_model(
    model_data: ModelCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new model.
    """
    try:
        db_model = await model_service.create_model(db, model_data)
        return db_model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")


@router.get("/", response_model=ModelList)
async def get_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get a list of models.
    """
    models = await model_service.get_models(db, skip, limit)
    return {"models": models, "count": len(models)}


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a model by ID.
    """
    db_model = await model_service.get_model(db, model_id)
    if not db_model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    return db_model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: UUID,
    model_data: ModelUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a model.
    """
    try:
        db_model = await model_service.update_model(db, model_id, model_data)
        if not db_model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        return db_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a model.
    """
    try:
        success = await model_service.delete_model(db, model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


@router.post("/{model_id}/validate", status_code=200)
async def validate_model_connection(
    model_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Validate the connection to a model.
    """
    try:
        is_valid = await model_service.validate_model_connection(model_id, db)
        return {"valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating model connection: {str(e)}")


@router.get("/modalities")
async def get_model_modalities():
    """
    Get all available model modalities.
    
    This endpoint returns a list of all modalities supported by the system,
    such as NLP, Vision, etc.
    """
    try:
        # Hard-coded response for now
        return {
            "modalities": ["NLP"],
            "modality_info": {
                "NLP": {
                    "name": "Natural Language Processing", 
                    "description": "Models that process and generate text",
                    "sub_types": [
                        "Text Generation",
                        "Text2Text Generation",
                        "Question Answering",
                        "Text Classification",
                        "Zero-Shot Classification",
                        "Summarization"
                    ]
                }
            },
            "count": 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model modalities: {str(e)}")


@router.post("/validate")
async def validate_model_connection(model_config: Dict[str, Any]):
    """
    Validate a model connection and readiness.
    
    This endpoint allows testing a model configuration before starting actual tests
    to ensure the model is accessible and properly initialized.
    
    Returns:
        Dict with status and message
    """
    try:
        # Get appropriate adapter based on model configuration
        adapter = get_model_adapter(model_config)
        await adapter.initialize(model_config)
        
        # Validate connection
        connection_valid = await adapter.validate_connection()
        if not connection_valid:
            error_detail = {
                "stage": "connection_validation",
                "model_id": model_config.get("id", "Unknown"),
                "model_type": model_config.get("sub_type", "Unknown"),
                "api_key_provided": bool(model_config.get("api_key"))
            }
            
            return {
                "status": "error",
                "message": "Connection validation failed. The model may be temporarily unavailable or still loading.",
                "details": error_detail
            }
            
        # Perform a warm-up test
        warm_up_prompt = "Hello, this is a test."
        try:
            warm_up_result = await adapter.generate(warm_up_prompt)
            if warm_up_result.startswith("Error:"):
                error_detail = {
                    "stage": "warm_up",
                    "model_id": model_config.get("id", "Unknown"),
                    "model_type": model_config.get("sub_type", "Unknown"),
                    "prompt": warm_up_prompt,
                    "error_message": warm_up_result
                }
                
                return {
                    "status": "error", 
                    "message": f"Warm-up test failed: {warm_up_result}",
                    "details": error_detail
                }
            
            return {
                "status": "success",
                "message": "Model connection validated successfully and ready for testing.",
                "sample_output": warm_up_result,
                "model_info": {
                    "id": model_config.get("id", "Unknown"),
                    "type": model_config.get("sub_type", "Unknown"),
                    "modality": model_config.get("modality", "Unknown")
                }
            }
        except Exception as e:
            error_detail = {
                "stage": "warm_up",
                "model_id": model_config.get("id", "Unknown"),
                "model_type": model_config.get("sub_type", "Unknown"),
                "prompt": warm_up_prompt,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            
            return {
                "status": "error",
                "message": f"Error during warm-up test: {str(e)}",
                "details": error_detail
            }
            
    except Exception as e:
        error_detail = {
            "stage": "initialization",
            "model_id": model_config.get("id", "Unknown"),
            "model_type": model_config.get("sub_type", "Unknown"),
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        
        return {
            "status": "error",
            "message": f"Error initializing model: {str(e)}",
            "details": error_detail
        }


@router.post("/validate/huggingface")
async def validate_huggingface_model(model_config: Dict[str, Any]):
    """
    Validate a HuggingFace model connection with extensive warm-up.
    
    This endpoint is specifically designed for HuggingFace models,
    implementing multiple warm-up attempts to ensure the model is fully loaded.
    
    Returns:
        Dict with status, message, and detailed loading information
    """
    try:
        # Ensure it's a HuggingFace model
        if model_config.get("source") != "huggingface":
            model_config["source"] = "huggingface"
        
        # Get appropriate adapter
        adapter = get_model_adapter(model_config)
        await adapter.initialize(model_config)
        
        # First validate basic connection
        connection_valid = await adapter.validate_connection()
        if not connection_valid:
            return {
                "status": "error",
                "message": "Initial connection validation failed. The model may be unavailable.",
                "details": {
                    "stage": "connection_validation",
                    "model_id": model_config.get("model_id", "Unknown"),
                    "api_key_provided": bool(model_config.get("api_key"))
                }
            }
        
        # Warm-up sequence with multiple attempts
        max_attempts = 5
        warm_up_history = []
        model_ready = False
        
        for attempt in range(max_attempts):
            try:
                # Use a simple prompt
                warm_up_prompt = "Hello"
                start_time = asyncio.get_event_loop().time()
                warm_up_result = await adapter.generate(warm_up_prompt)
                duration = asyncio.get_event_loop().time() - start_time
                
                # Record attempt results
                attempt_info = {
                    "attempt": attempt + 1,
                    "duration": f"{duration:.2f}s",
                    "success": not warm_up_result.startswith("Error:"),
                    "response": warm_up_result[:100] + ("..." if len(warm_up_result) > 100 else "")
                }
                warm_up_history.append(attempt_info)
                
                if not warm_up_result.startswith("Error:"):
                    model_ready = True
                    break
                
                # If model is still loading, wait and retry
                if "temporarily unavailable" in warm_up_result or "still loading" in warm_up_result:
                    wait_time = (attempt + 1) * 5  # Increasing wait time
                    if attempt < max_attempts - 1:  # Don't wait on the last attempt
                        await asyncio.sleep(wait_time)
            except Exception as e:
                # Record error
                attempt_info = {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "success": False
                }
                warm_up_history.append(attempt_info)
                
                # Wait before retrying
                if attempt < max_attempts - 1:  # Don't wait on the last attempt
                    wait_time = (attempt + 1) * 5
                    await asyncio.sleep(wait_time)
        
        # Return results
        if model_ready:
            return {
                "status": "success",
                "message": f"HuggingFace model is available and ready after {len(warm_up_history)} attempts",
                "warm_up_history": warm_up_history,
                "model_info": {
                    "id": model_config.get("model_id", "Unknown"),
                    "type": model_config.get("sub_type", "Unknown")
                }
            }
        else:
            return {
                "status": "error",
                "message": f"HuggingFace model failed to respond after {max_attempts} warm-up attempts",
                "warm_up_history": warm_up_history,
                "details": {
                    "model_id": model_config.get("model_id", "Unknown"),
                    "type": model_config.get("sub_type", "Unknown"),
                    "suggestion": "Try a different model or check your API key permissions"
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during HuggingFace model validation: {str(e)}",
            "details": {
                "error_type": type(e).__name__,
                "model_id": model_config.get("model_id", "Unknown")
            }
        }


@router.post("/initialize")
async def initialize_model(model_config: Dict[str, Any], try_alternatives: bool = False):
    """
    Initialize a model and confirm it's ready for testing.
    
    This endpoint explicitly loads the model and performs a simple test to ensure
    it's ready for use. It provides a clear status message to confirm readiness.
    
    If the primary model fails and try_alternatives is True, it will attempt to
    initialize alternative smaller models of the same type.
    
    Returns:
        Dict with initialization status and details
    """
    # Track original model config in case we need to try alternatives
    original_config = model_config.copy()
    original_model_id = model_config.get("model_id", "Unknown")
    
    try:
        init_start_time = asyncio.get_event_loop().time()
        
        # Get appropriate adapter based on model configuration
        adapter = get_model_adapter(model_config)
        await adapter.initialize(model_config)
        
        # First validate basic connection
        connection_valid = await adapter.validate_connection()
        if not connection_valid:
            # If connection failed and we should try alternatives
            if try_alternatives and model_config.get("source") == "huggingface":
                # Get the model type to find appropriate alternatives
                model_type = model_config.get("sub_type", "")
                
                # Map of alternative models by type
                alt_models = {
                    "Text Generation": ["distilgpt2", "sshleifer/tiny-gpt2", "distilbert-base-uncased"],
                    "Text2Text Generation": ["sshleifer/distilbart-cnn-6-6", "t5-small", "facebook/bart-base"],
                    "Question Answering": ["distilbert-base-cased-distilled-squad", "mrm8488/bert-tiny-5-finetuned-squadv2"],
                    "Text Classification": ["distilbert-base-uncased-finetuned-sst-2-english", "prajjwal1/bert-tiny"],
                    "Zero-Shot Classification": ["facebook/bart-large-mnli", "typeform/distilbert-base-uncased-mnli"]
                }
                
                # Get alternatives for this model type
                alternatives = alt_models.get(model_type, [])
                
                # If we have alternatives, try them
                if alternatives:
                    alt_results = []
                    for i, alt_model_id in enumerate(alternatives):
                        try:
                            logger.info(f"Trying alternative model {i+1}/{len(alternatives)}: {alt_model_id}")
                            
                            # Create a new config with the alternative model
                            alt_config = model_config.copy()
                            alt_config["model_id"] = alt_model_id
                            
                            # Try to initialize and validate
                            alt_adapter = get_model_adapter(alt_config)
                            await alt_adapter.initialize(alt_config)
                            alt_valid = await alt_adapter.validate_connection()
                            
                            if alt_valid:
                                # Test with a simple prompt
                                test_prompt = "Hello, this is a test."
                                test_result = await alt_adapter.generate(test_prompt)
                                
                                if not test_result.startswith("Error:"):
                                    # Success! Return with information about the fallback
                                    return {
                                        "status": "success",
                                        "initialized": True,
                                        "fallback_used": True,
                                        "original_model": original_model_id,
                                        "message": f"Alternative model successfully initialized and ready for testing",
                                        "details": {
                                            "model_id": alt_model_id,
                                            "source": "huggingface",
                                            "time_taken": f"{asyncio.get_event_loop().time() - init_start_time:.2f}s",
                                            "sample_output": test_result[:100] + ("..." if len(test_result) > 100 else "")
                                        }
                                    }
                            
                            # Record this attempt
                            alt_results.append({
                                "model_id": alt_model_id,
                                "success": alt_valid
                            })
                            
                        except Exception as alt_e:
                            logger.error(f"Error with alternative model {alt_model_id}: {str(alt_e)}")
                            alt_results.append({
                                "model_id": alt_model_id,
                                "success": False,
                                "error": str(alt_e)
                            })
                    
                    # If we get here, all alternatives failed
                    return {
                        "status": "error",
                        "initialized": False,
                        "message": "Primary model and all alternatives failed to initialize",
                        "details": {
                            "model_id": original_model_id,
                            "source": model_config.get("source", "Unknown"),
                            "time_taken": f"{asyncio.get_event_loop().time() - init_start_time:.2f}s",
                            "alternative_attempts": alt_results
                        }
                    }
                
            # No alternatives or alternatives not enabled
            return {
                "status": "error",
                "initialized": False,
                "message": "Model initialization failed: Cannot connect to model API",
                "details": {
                    "model_id": model_config.get("model_id", "Unknown"),
                    "source": model_config.get("source", "Unknown"),
                    "time_taken": f"{asyncio.get_event_loop().time() - init_start_time:.2f}s"
                }
            }
        
        # Perform a simple generation test to confirm the model is ready
        test_prompt = "Hello, this is a test."
        try:
            test_result = await adapter.generate(test_prompt)
            
            # Calculate total initialization time
            init_time = asyncio.get_event_loop().time() - init_start_time
            
            if test_result.startswith("Error:"):
                # If generation failed and we should try alternatives
                if try_alternatives and model_config.get("source") == "huggingface":
                    # Similar alternative logic as above
                    # This is a simplified version to avoid code duplication
                    return {
                        "status": "error",
                        "initialized": False,
                        "message": f"Model initialization failed: {test_result}",
                        "details": {
                            "model_id": model_config.get("model_id", "Unknown"),
                            "source": model_config.get("source", "Unknown"),
                            "time_taken": f"{init_time:.2f}s",
                            "suggestion": "Try enabling alternative models with try_alternatives=true"
                        }
                    }
                
                # Regular error with no alternatives
                return {
                    "status": "error",
                    "initialized": False,
                    "message": f"Model initialization failed: {test_result}",
                    "details": {
                        "model_id": model_config.get("model_id", "Unknown"),
                        "source": model_config.get("source", "Unknown"),
                        "time_taken": f"{init_time:.2f}s"
                    }
                }
            
            # Success case
            return {
                "status": "success",
                "initialized": True,
                "message": "Model successfully initialized and ready for testing",
                "details": {
                    "model_id": model_config.get("model_id", "Unknown"),
                    "source": model_config.get("source", "Unknown"),
                    "time_taken": f"{init_time:.2f}s",
                    "sample_output": test_result[:100] + ("..." if len(test_result) > 100 else "")
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "initialized": False,
                "message": f"Model initialization failed during generation test: {str(e)}",
                "details": {
                    "model_id": model_config.get("model_id", "Unknown"),
                    "source": model_config.get("source", "Unknown"),
                    "time_taken": f"{asyncio.get_event_loop().time() - init_start_time:.2f}s",
                    "error": str(e)
                }
            }
            
    except Exception as e:
        return {
            "status": "error",
            "initialized": False,
            "message": f"Model initialization failed: {str(e)}",
            "details": {
                "error_type": type(e).__name__,
                "error": str(e)
            }
        }


@router.post("/debug/adapter")
async def debug_create_adapter(model_config: dict):
    """
    Debug endpoint to test model adapter creation.
    This helps identify issues with model adapter initialization.
    """
    try:
        from app.model_adapters import get_model_adapter
        
        adapter = get_model_adapter(model_config)
        adapter_type = type(adapter).__name__
        
        # Test basic functionality
        test_result = None
        connection_status = None
        
        try:
            # Test connection
            connection_status = await adapter.validate_connection()
            
            # Test generation with a simple prompt
            if hasattr(adapter, 'generate'):
                test_result = await adapter.generate("Hello, this is a test.")
        except Exception as e:
            test_result = f"Error testing adapter: {str(e)}"
        
        return {
            "success": True,
            "adapter_type": adapter_type,
            "connection_status": connection_status,
            "test_result": test_result
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        } 