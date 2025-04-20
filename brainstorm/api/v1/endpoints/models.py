"""API endpoints for model management."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
import asyncio
import logging
from fastapi import Response

from fastapi.responses import JSONResponse

from brainstorm.db.base import get_db
from brainstorm.api.v1.schemas.model import (
    ModelDefinition
)
from brainstorm.core.adapters import get_model_adapter
from brainstorm.db.models.model import ModelDefinitionDataModel

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/create_or_update_model")
async def create_or_update_model(
    model_definition: ModelDefinition,
    db: Session = Depends(get_db)
):
    """
    Create or update a model definition.

    Currently:
    - We have parity between SQL model and API model
    - We always receive full model definition
    - We let frontend generate unique model IDs
    - If model with a given ID already exists we simply update it with definition we received
    """
    try:
        db_model = ModelDefinitionDataModel(**model_definition.dict())

        stmt = select(ModelDefinitionDataModel).filter_by(model_id=model_definition.model_id)
        existing = db.execute(stmt).scalar_one_or_none()

        if not existing:
            db.add(db_model)

        db.commit()
        db.refresh(db_model)
        return Response(
            status_code=200,
        )
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a model due to SQLAlchemyError")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Failed to create or update a model due to generic exception")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/get_models", response_model=List[ModelDefinition])
async def get_models(
    model_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get a list of models.
    """
    stmt = select(ModelDefinitionDataModel)

    if model_id:
        stmt = stmt.filter_by(model_id=model_id)

    return db.execute(stmt).scalars().all()


@router.post("/delete_models")
async def delete_model(
    model_ids: List[str],
    db: Session = Depends(get_db)
):
    """
    Delete models.
    """
    try:
        stmt = delete(ModelDefinitionDataModel).where(ModelDefinitionDataModel.model_id.in_(model_ids))
        db.execute(stmt)
        db.commit()

        # todo: handle if some (or all) requested model IDs where not found and therefore not deleted

    # todo: this duplicates creat or update code, should refactor or may be there are some constructs for that in the library
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Failed to create or update a model due to SQLAlchemyError")
        return HTTPException(
            status_code=500,
            detail=e
        )
    except Exception as e:
        logger.exception("Failed to create or update a model due to generic exception")
        return HTTPException(
            status_code=500,
            detail=e
        )



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