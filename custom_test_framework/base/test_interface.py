"""Base interface for all custom tests."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import uuid
from datetime import datetime


class CustomTest(ABC):
    """Base interface for all custom tests."""
    
    def __init__(self, test_id: str = None, name: str = None, description: str = None):
        """Initialize the custom test.
        
        Args:
            test_id: Unique identifier for the test
            name: Human-readable name for the test
            description: Detailed description of what the test evaluates
        """
        self.test_id = test_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description or "Custom test implementation"
        self.category = "custom"
        self.metadata = {}
        
    @property
    def test_info(self) -> Dict[str, Any]:
        """Get test information for registration."""
        return {
            "id": self.test_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "compatible_modalities": self.get_compatible_modalities(),
            "compatible_sub_types": self.get_compatible_model_types(),
            "parameter_schema": self.get_parameter_schema(),
            "default_config": self.get_default_config(),
            "metadata": self.metadata,
            "custom": True
        }
    
    @abstractmethod
    async def run(self, model_adapter: Any, test_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the test against a model.
        
        Args:
            model_adapter: The model adapter to test
            test_parameters: Test configuration parameters
            
        Returns:
            Test results with standardized format
        """
        pass
    
    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for test parameters.
        
        Returns:
            JSON Schema for validating test parameters
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the test.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def get_compatible_modalities(self) -> List[str]:
        """
        Get compatible model modalities (e.g., "NLP", "Vision").
        
        Returns:
            List of compatible modalities
        """
        pass
    
    @abstractmethod
    def get_compatible_model_types(self) -> List[str]:
        """
        Get compatible model types (e.g., "Text Generation").
        
        Returns:
            List of compatible model types
        """
        pass
    
    def format_result(self, status: str, score: float, metrics: Dict[str, Any], 
                     issues_found: int, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the test result in the standardized format.
        
        Args:
            status: Test status ("success", "error", etc.)
            score: Overall test score (0-100)
            metrics: Detailed test metrics
            issues_found: Number of issues found
            analysis: Detailed analysis of results
            
        Returns:
            Formatted test result
        """
        return {
            "id": str(uuid.uuid4()),
            "test_run_id": None,  # Will be set by the framework
            "test_id": self.test_id,
            "test_category": self.category,
            "test_name": self.name,
            "status": status,
            "score": score,
            "metrics": metrics,
            "issues_found": issues_found,
            "analysis": analysis,
            "created_at": datetime.utcnow()
        } 