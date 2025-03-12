"""Example of an invalid test that will fail validation."""
from typing import Dict, Any, List

from custom_test_framework.base import CustomTest


class InvalidTest:  # Missing inheritance from CustomTest
    """Invalid test that doesn't inherit from CustomTest."""
    
    def __init__(self, test_id=None, name=None, description=None):
        """Initialize the invalid test."""
        self.test_id = test_id or "invalid_test"
        self.name = name or "Invalid Test"
        self.description = description or "This test will fail validation"
        self.category = "invalid"
        
    # Missing required methods
    
    def get_parameter_schema(self):
        """Get parameter schema without proper typing."""
        # Missing return type annotation
        return {
            # Missing "type": "object"
            "properties": {
                "n_examples": {
                    "type": "integer",
                    "description": "Number of examples",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                }
            }
        }
    
    # Missing get_default_config method
    
    def get_compatible_modalities(self):
        """Return non-list value for modalities."""
        # Should return a list, but returns a string instead
        return "NLP"
    
    def get_compatible_model_types(self):
        """Return empty list for model types."""
        return []
    
    # run method is not async
    def run(self, model_adapter, test_parameters):
        """Run method that isn't async."""
        return {
            "status": "error",
            "score": 0,
            "metrics": {},
            "issues_found": 1,
            "analysis": {"error": "This test is invalid"}
        }


class PartiallyInvalidTest(CustomTest):
    """Partially invalid test with some validation issues."""
    
    def __init__(self, test_id=None, name=None, description=None):
        """Initialize the partially invalid test."""
        super().__init__(test_id, name, description)
        self.category = "invalid"
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "n_examples": {
                    "type": "integer",
                    "description": "Number of examples",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                }
            }
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "n_examples": 5
        }
    
    def get_compatible_modalities(self) -> List[str]:
        """Get compatible modalities."""
        return ["NLP"]
    
    def get_compatible_model_types(self) -> List[str]:
        """Get compatible model types."""
        return ["Text Generation"]
    
    # Missing async keyword
    def run(self, model_adapter: Any, test_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run method without async keyword."""
        return {
            "status": "error",
            "score": 0,
            "metrics": {},
            "issues_found": 1,
            "analysis": {"error": "This test is partially invalid"}
        } 