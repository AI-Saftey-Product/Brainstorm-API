"""Registry for custom tests."""
from typing import Dict, Any, List, Optional, Type
import logging
import importlib
import inspect
import json
import os
from pathlib import Path

from custom_test_framework.base.test_interface import CustomTest
from custom_test_framework.base.validator import TestValidator

logger = logging.getLogger(__name__)


class TestRegistry:
    """Registry for custom test implementations."""
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the custom test registry.
        
        Args:
            registry_path: Optional path to save/load registry data
        """
        self._tests = {}
        self.registry_path = registry_path
        
        # Load existing registry if path is provided
        if registry_path:
            self.load_registry()
    
    def register_test(self, test_instance: CustomTest, validate: bool = True) -> Dict[str, Any]:
        """
        Register a test instance in the registry.
        
        Args:
            test_instance: Instance of a custom test
            validate: Whether to validate the test before registering
            
        Returns:
            Registration result with validation info if applicable
        """
        result = {
            "status": "success",
            "test_id": test_instance.test_id,
            "test_name": test_instance.name,
            "validation": None
        }
        
        # Validate if requested
        if validate:
            validation = TestValidator.validate_test_class(test_instance.__class__)
            result["validation"] = validation
            
            if not validation["valid"]:
                result["status"] = "error"
                result["message"] = "Test validation failed"
                logger.error(f"Test validation failed for {test_instance.test_id}: {validation['errors']}")
                return result
        
        # Register the test
        test_info = test_instance.test_info
        test_id = test_info["id"]
        
        if test_id in self._tests:
            result["message"] = f"Test with ID {test_id} already registered, overwriting"
            logger.warning(f"Test with ID {test_id} already registered, overwriting")
            
        self._tests[test_id] = {
            "info": test_info,
            "class": test_instance.__class__.__name__,
            "module": test_instance.__class__.__module__
        }
        
        logger.info(f"Registered custom test: {test_id} - {test_info['name']}")
        
        # Save registry if path is provided
        if self.registry_path:
            self.save_registry()
        
        return result
    
    def register_test_class(self, 
                          test_class: Type[CustomTest], 
                          test_id: Optional[str] = None, 
                          name: Optional[str] = None,
                          description: Optional[str] = None,
                          validate: bool = True) -> Dict[str, Any]:
        """
        Register a test class in the registry.
        
        Args:
            test_class: Custom test class
            test_id: Optional test ID (defaults to class name)
            name: Optional test name (defaults to class name)
            description: Optional test description
            validate: Whether to validate the test before registering
            
        Returns:
            Registration result with validation info if applicable
        """
        try:
            # Create an instance and register it
            instance = test_class(test_id=test_id, name=name, description=description)
            return self.register_test(instance, validate=validate)
        except Exception as e:
            logger.error(f"Error registering test class {test_class.__name__}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error registering test class: {str(e)}",
                "test_id": test_id or getattr(test_class, "__name__", "unknown"),
                "validation": None
            }
    
    def import_and_register(self, 
                           module_path: str, 
                           class_name: str,
                           test_id: Optional[str] = None,
                           name: Optional[str] = None,
                           description: Optional[str] = None,
                           validate: bool = True) -> Dict[str, Any]:
        """
        Import and register a test class from a module.
        
        Args:
            module_path: Path to the module containing the test class
            class_name: Name of the test class
            test_id: Optional test ID
            name: Optional test name
            description: Optional test description
            validate: Whether to validate the test before registering
            
        Returns:
            Registration result
        """
        try:
            # Import the module and class
            module = importlib.import_module(module_path)
            test_class = getattr(module, class_name)
            
            # Validate it's a subclass of CustomTest
            if not inspect.isclass(test_class) or not issubclass(test_class, CustomTest):
                return {
                    "status": "error",
                    "message": f"Class {class_name} is not a valid custom test class",
                    "test_id": test_id or class_name,
                    "validation": None
                }
            
            # Register the class
            return self.register_test_class(
                test_class, 
                test_id=test_id, 
                name=name, 
                description=description,
                validate=validate
            )
        except Exception as e:
            logger.error(f"Error importing and registering test: {str(e)}")
            return {
                "status": "error",
                "message": f"Error importing and registering test: {str(e)}",
                "test_id": test_id or class_name,
                "validation": None
            }
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all registered tests.
        
        Returns:
            Dictionary of test information indexed by test ID
        """
        return {test_id: test_data["info"] for test_id, test_data in self._tests.items()}
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get test information by ID.
        
        Args:
            test_id: Test ID to retrieve
            
        Returns:
            Test information or None if not found
        """
        test_data = self._tests.get(test_id)
        return test_data["info"] if test_data else None
    
    def get_test_class(self, test_id: str) -> Optional[Type[CustomTest]]:
        """
        Get the test class for a registered test.
        
        Args:
            test_id: Test ID to retrieve
            
        Returns:
            Test class or None if not found
        """
        test_data = self._tests.get(test_id)
        if not test_data:
            return None
        
        try:
            module = importlib.import_module(test_data["module"])
            return getattr(module, test_data["class"])
        except Exception as e:
            logger.error(f"Error getting test class for {test_id}: {str(e)}")
            return None
    
    def get_test_instance(self, test_id: str) -> Optional[CustomTest]:
        """
        Get a new instance of a registered test.
        
        Args:
            test_id: Test ID to instantiate
            
        Returns:
            New test instance or None if not found
        """
        test_class = self.get_test_class(test_id)
        if not test_class:
            return None
        
        try:
            test_info = self._tests[test_id]["info"]
            return test_class(
                test_id=test_info["id"],
                name=test_info["name"],
                description=test_info["description"]
            )
        except Exception as e:
            logger.error(f"Error instantiating test for {test_id}: {str(e)}")
            return None
    
    def unregister_test(self, test_id: str) -> bool:
        """
        Unregister a test from the registry.
        
        Args:
            test_id: Test ID to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        if test_id in self._tests:
            del self._tests[test_id]
            logger.info(f"Unregistered test: {test_id}")
            
            # Save registry if path is provided
            if self.registry_path:
                self.save_registry()
                
            return True
        
        logger.warning(f"Test with ID {test_id} not found in registry")
        return False
    
    def save_registry(self) -> bool:
        """
        Save the registry to the specified path.
        
        Returns:
            True if successfully saved, False otherwise
        """
        if not self.registry_path:
            logger.warning("No registry path specified, cannot save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            
            # Prepare serializable data
            serializable = {}
            for test_id, test_data in self._tests.items():
                serializable[test_id] = {
                    "info": test_data["info"],
                    "class": test_data["class"],
                    "module": test_data["module"]
                }
            
            # Write to file
            with open(self.registry_path, 'w') as f:
                json.dump(serializable, f, indent=2)
                
            logger.info(f"Registry saved to {self.registry_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            return False
    
    def load_registry(self) -> bool:
        """
        Load the registry from the specified path.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.registry_path:
            logger.warning("No registry path specified, cannot load")
            return False
        
        try:
            # Check if file exists
            if not os.path.exists(self.registry_path):
                logger.info(f"Registry file {self.registry_path} not found, starting with empty registry")
                return False
            
            # Read from file
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            # Load data
            self._tests = {}
            for test_id, test_data in data.items():
                self._tests[test_id] = test_data
                
            logger.info(f"Registry loaded from {self.registry_path} with {len(self._tests)} tests")
            return True
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            return False


# Create default registry instance
registry = TestRegistry() 