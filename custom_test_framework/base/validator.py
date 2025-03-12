"""Validator for custom tests."""
from typing import Dict, Any, Optional, List, Callable, Type
import inspect
import logging

from custom_test_framework.base.test_interface import CustomTest

logger = logging.getLogger(__name__)


class TestValidator:
    """Validates custom tests for compatibility with the existing framework."""
    
    @staticmethod
    def validate_test_class(test_class: Type) -> Dict[str, Any]:
        """
        Validate a test class for compatibility with the existing framework.
        
        Args:
            test_class: Class to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 1. Check inheritance
        if not inspect.isclass(test_class):
            validation_results["valid"] = False
            validation_results["errors"].append("Test class must be a class")
            return validation_results
            
        if not issubclass(test_class, CustomTest):
            validation_results["valid"] = False
            validation_results["errors"].append("Test class must inherit from CustomTest")
        
        # 2. Check required methods
        required_methods = [
            "run", "get_parameter_schema", "get_default_config",
            "get_compatible_modalities", "get_compatible_model_types"
        ]
        
        for method_name in required_methods:
            if not hasattr(test_class, method_name):
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required method: {method_name}")
            elif not inspect.isfunction(getattr(test_class, method_name)):
                validation_results["valid"] = False
                validation_results["errors"].append(f"{method_name} must be a method")
        
        # 3. Check method signatures
        method_signatures = {
            "run": ["self", "model_adapter", "test_parameters"],
            "get_parameter_schema": ["self"],
            "get_default_config": ["self"],
            "get_compatible_modalities": ["self"],
            "get_compatible_model_types": ["self"],
            "format_result": ["self", "status", "score", "metrics", "issues_found", "analysis"]
        }
        
        for method_name, expected_params in method_signatures.items():
            method = getattr(test_class, method_name, None)
            if method and inspect.isfunction(method):
                try:
                    sig = inspect.signature(method)
                    param_names = list(sig.parameters.keys())
                    missing_params = [param for param in expected_params if param not in param_names]
                    
                    if missing_params:
                        validation_results["valid"] = False
                        validation_results["errors"].append(
                            f"Method {method_name} is missing required parameters: {', '.join(missing_params)}"
                        )
                except Exception as e:
                    validation_results["warnings"].append(
                        f"Could not check signature for {method_name}: {str(e)}"
                    )
        
        # 4. Check if run method is async
        run_method = getattr(test_class, "run", None)
        if run_method and inspect.isfunction(run_method):
            if not inspect.iscoroutinefunction(run_method):
                validation_results["valid"] = False
                validation_results["errors"].append("run() method must be async (use 'async def')")
        
        # 5. Create an instance to check instance properties
        try:
            instance = test_class()
            
            # Check test_info structure
            test_info = instance.test_info
            required_fields = ["id", "name", "description", "category", 
                             "compatible_modalities", "compatible_sub_types", 
                             "parameter_schema", "default_config"]
            
            missing_fields = [field for field in required_fields if field not in test_info]
            if missing_fields:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"test_info missing required fields: {', '.join(missing_fields)}"
                )
            
            # Check parameter schema structure
            schema = instance.get_parameter_schema()
            if not isinstance(schema, dict):
                validation_results["valid"] = False
                validation_results["errors"].append("get_parameter_schema must return a dictionary")
            elif "type" not in schema or schema["type"] != "object":
                validation_results["warnings"].append(
                    "Parameter schema should have 'type': 'object'"
                )
            elif "properties" not in schema:
                validation_results["warnings"].append(
                    "Parameter schema should have a 'properties' field"
                )
            
            # Check default config
            config = instance.get_default_config()
            if not isinstance(config, dict):
                validation_results["valid"] = False
                validation_results["errors"].append("get_default_config must return a dictionary")
            
            # Check compatible modalities and types
            modalities = instance.get_compatible_modalities()
            if not isinstance(modalities, list) or not modalities:
                validation_results["warnings"].append(
                    "get_compatible_modalities should return a non-empty list"
                )
            
            model_types = instance.get_compatible_model_types()
            if not isinstance(model_types, list) or not model_types:
                validation_results["warnings"].append(
                    "get_compatible_model_types should return a non-empty list"
                )
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Error instantiating test class: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_test_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a test result for compatibility with the existing framework.
        
        Args:
            result: Test result to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = [
            "id", "test_id", "test_category", "test_name", 
            "status", "score", "metrics", "issues_found", "analysis"
        ]
        
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Result is missing required fields: {', '.join(missing_fields)}"
            )
        
        # Check field types
        type_checks = {
            "id": str,
            "test_id": str,
            "test_category": str,
            "test_name": str,
            "status": str,
            "score": (int, float),
            "metrics": dict,
            "issues_found": int,
            "analysis": dict
        }
        
        for field, expected_type in type_checks.items():
            if field in result and not isinstance(result[field], expected_type):
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Field '{field}' must be of type {expected_type.__name__}, but got {type(result[field]).__name__}"
                )
        
        # Check status values
        if "status" in result and result["status"] not in ["success", "error", "pending"]:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Invalid status value: {result['status']}. Must be 'success', 'error', or 'pending'"
            )
        
        # Check score range
        if "score" in result:
            score = result["score"]
            if isinstance(score, (int, float)) and (score < 0 or score > 100):
                validation_results["warnings"].append(
                    f"Score {score} is outside the recommended range of 0-100"
                )
        
        # Check metrics structure
        if "metrics" in result and isinstance(result["metrics"], dict):
            metrics = result["metrics"]
            if "test_results" not in metrics:
                validation_results["warnings"].append(
                    "Metrics should include a 'test_results' field for consistency with built-in tests"
                )
            
            if "test_results" in metrics and isinstance(metrics["test_results"], dict):
                if "performance_metrics" not in metrics["test_results"]:
                    validation_results["warnings"].append(
                        "test_results should include 'performance_metrics' field"
                    )
        
        return validation_results 