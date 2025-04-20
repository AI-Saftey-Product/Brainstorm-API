"""Service for handling model operations."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging

from brainstorm.db.models.model import Model
from brainstorm.api.v1.schemas.model import ModelCreate, ModelUpdate
from brainstorm.core.adapters import get_model_adapter


logger = logging.getLogger(__name__)





async def get_models(db: Session, skip: int = 0, limit: int = 100) -> List[Model]:
    """
    Get a list of models.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of models
    """
    return db.query(Model).offset(skip).limit(limit).all()


async def update_model(db: Session, model_id: UUID, model_data: ModelUpdate) -> Optional[Model]:
    """
    Update a model.
    
    Args:
        db: Database session
        model_id: Model ID
        model_data: Model update data
        
    Returns:
        The updated model if found, None otherwise
    """
    try:
        db_model = await get_model(db, model_id)
        if not db_model:
            return None
        
        # Update fields
        update_data = model_data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_model, key, value)
        
        db.commit()
        db.refresh(db_model)
        return db_model
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating model: {str(e)}")
        raise


async def delete_model(db: Session, model_id: UUID) -> bool:
    """
    Delete a model.
    
    Args:
        db: Database session
        model_id: Model ID
        
    Returns:
        True if deleted, False if not found
    """
    try:
        db_model = await get_model(db, model_id)
        if not db_model:
            return False
        
        db.delete(db_model)
        db.commit()
        return True
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error deleting model: {str(e)}")
        raise


async def validate_model_connection(model_id: UUID, db: Session) -> bool:
    """
    Validate the connection to a model.
    
    Args:
        model_id: Model ID
        db: Database session
        
    Returns:
        True if connection is valid, False otherwise
    """
    try:
        db_model = await get_model(db, model_id)
        if not db_model:
            return False
        
        # Get model adapter
        model_config = {
            "id": str(db_model.id),
            "modality": db_model.modality,
            "sub_type": db_model.sub_type,
            "endpoint_url": db_model.endpoint_url,
            # In a real implementation, you would securely retrieve the API key
            "api_key": None  # This would be retrieved from a secure storage
        }
        
        adapter = get_model_adapter(db_model.modality, model_config)
        await adapter.initialize(model_config)
        
        # Validate connection
        is_valid = await adapter.validate_connection()
        return is_valid
    except Exception as e:
        logger.error(f"Error validating model connection: {str(e)}")
        return False 