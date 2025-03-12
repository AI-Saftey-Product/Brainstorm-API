"""Command-line interface for the custom testing framework."""
import argparse
import sys
import os
import logging
import importlib.util
import inspect
import json
from typing import Dict, Any, List, Optional
import asyncio
from pathlib import Path
import time

from custom_test_framework.base import CustomTest, TestValidator, registry

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to use verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_test_command(args) -> int:
    """
    Validate a test file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load the Python file
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return 1
        
        # Get module and class names
        if args.module and args.class_name:
            module_name = args.module
            class_name = args.class_name
        else:
            # Try to infer from file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            # We'll try to find the class later
            class_name = args.class_name
        
        # Import the module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            print(f"Error: Could not load module from {file_path}")
            return 1
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the test class if not specified
        if not class_name:
            # Look for classes that inherit from CustomTest
            test_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, CustomTest) and 
                    obj != CustomTest):
                    test_classes.append((name, obj))
            
            if not test_classes:
                print(f"Error: No custom test classes found in {file_path}")
                return 1
            elif len(test_classes) > 1:
                print(f"Multiple custom test classes found in {file_path}. Please specify one with --class:")
                for name, _ in test_classes:
                    print(f"  - {name}")
                return 1
            else:
                class_name, test_class = test_classes[0]
                print(f"Found test class: {class_name}")
        else:
            # Get the specified class
            if not hasattr(module, class_name):
                print(f"Error: Class {class_name} not found in {file_path}")
                return 1
                
            test_class = getattr(module, class_name)
            
            # Verify it's a custom test
            if not inspect.isclass(test_class) or not issubclass(test_class, CustomTest):
                print(f"Error: {class_name} is not a CustomTest subclass")
                return 1
        
        # Validate the test class
        print(f"Validating test class: {class_name}")
        validation = TestValidator.validate_test_class(test_class)
        
        # Display results
        if validation["valid"]:
            print("\n✅ Test class is valid!")
        else:
            print("\n❌ Test class validation failed")
        
        # Display errors
        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        # Display warnings
        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        # Show test info if validation passed
        if validation["valid"]:
            test_instance = test_class()
            test_info = test_instance.test_info
            
            print("\nTest Information:")
            print(f"  ID: {test_info['id']}")
            print(f"  Name: {test_info['name']}")
            print(f"  Description: {test_info['description']}")
            print(f"  Category: {test_info['category']}")
            print(f"  Compatible Modalities: {', '.join(test_info['compatible_modalities'])}")
            print(f"  Compatible Model Types: {', '.join(test_info['compatible_sub_types'])}")
            
            # Show schema summary
            schema = test_info["parameter_schema"]
            if "properties" in schema:
                print("\nParameters:")
                for param_name, param_info in schema["properties"].items():
                    default = param_info.get("default", "Not specified")
                    desc = param_info.get("description", "No description")
                    print(f"  {param_name}: {desc} (default: {default})")
        
        return 0 if validation["valid"] else 1
    
    except Exception as e:
        print(f"Error validating test: {str(e)}")
        logger.exception("Validation error")
        return 1


def register_test_command(args) -> int:
    """
    Register a test with the registry.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set up registry path if provided
        if args.registry_path:
            registry.registry_path = args.registry_path
        
        # Load the Python file
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return 1
        
        # Get module and class names
        if args.module and args.class_name:
            module_name = args.module
            class_name = args.class_name
        else:
            # Try to infer from file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            # We'll try to find the class later
            class_name = args.class_name
        
        # Import the module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            print(f"Error: Could not load module from {file_path}")
            return 1
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the test class if not specified
        if not class_name:
            # Look for classes that inherit from CustomTest
            test_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, CustomTest) and 
                    obj != CustomTest):
                    test_classes.append((name, obj))
            
            if not test_classes:
                print(f"Error: No custom test classes found in {file_path}")
                return 1
            elif len(test_classes) > 1:
                print(f"Multiple custom test classes found in {file_path}. Please specify one with --class:")
                for name, _ in test_classes:
                    print(f"  - {name}")
                return 1
            else:
                class_name, test_class = test_classes[0]
                print(f"Found test class: {class_name}")
        else:
            # Get the specified class
            if not hasattr(module, class_name):
                print(f"Error: Class {class_name} not found in {file_path}")
                return 1
                
            test_class = getattr(module, class_name)
            
            # Verify it's a custom test
            if not inspect.isclass(test_class) or not issubclass(test_class, CustomTest):
                print(f"Error: {class_name} is not a CustomTest subclass")
                return 1
        
        # Register the test class
        print(f"Registering test class: {class_name}")
        result = registry.register_test_class(
            test_class,
            test_id=args.test_id,
            name=args.name,
            description=args.description,
            validate=not args.skip_validation
        )
        
        # Display results
        if result["status"] == "success":
            print(f"\n✅ Test {result['test_id']} registered successfully!")
            
            # Show validation results if available
            if result.get("validation"):
                validation = result["validation"]
                
                # Display warnings
                if validation.get("warnings"):
                    print("\nWarnings:")
                    for warning in validation["warnings"]:
                        print(f"  - {warning}")
        else:
            print(f"\n❌ Test registration failed: {result.get('message', 'Unknown error')}")
            
            # Show validation errors if available
            if result.get("validation") and result["validation"].get("errors"):
                print("\nValidation errors:")
                for error in result["validation"]["errors"]:
                    print(f"  - {error}")
        
        return 0 if result["status"] == "success" else 1
    
    except Exception as e:
        print(f"Error registering test: {str(e)}")
        logger.exception("Registration error")
        return 1


def list_tests_command(args) -> int:
    """
    List registered tests.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set up registry path if provided
        if args.registry_path:
            registry.registry_path = args.registry_path
            registry.load_registry()
        
        # Get tests
        tests = registry.get_all_tests()
        
        if not tests:
            print("No tests registered.")
            return 0
        
        print(f"Found {len(tests)} registered tests:")
        print("")
        
        for test_id, test_info in tests.items():
            print(f"Test ID: {test_id}")
            print(f"  Name: {test_info['name']}")
            print(f"  Description: {test_info['description']}")
            print(f"  Category: {test_info['category']}")
            print(f"  Modalities: {', '.join(test_info['compatible_modalities'])}")
            print(f"  Model Types: {', '.join(test_info['compatible_sub_types'])}")
            print("")
        
        return 0
    
    except Exception as e:
        print(f"Error listing tests: {str(e)}")
        logger.exception("List error")
        return 1


def run_test_command(args) -> int:
    """
    Run a test against a specified model by communicating with the API service.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Get the test information from registry for display purposes
        if args.registry_path:
            registry.registry_path = args.registry_path
            registry.load_registry()
        
        test_id = args.test_id
        
        # Just for display purposes in CLI
        test_info = registry.get_test(test_id)
        if test_info:
            print(f"Running test: {test_info['name']} (ID: {test_id})")
        else:
            print(f"Running test with ID: {test_id}")
        
        # Set up API client configuration
        api_base_url = args.api_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        api_key = args.api_key or os.environ.get("API_KEY")
        
        # Validate API connection
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Check API connection
        try:
            print(f"Connecting to API at {api_base_url}")
            client = httpx.Client(base_url=api_base_url, headers=headers, timeout=10.0)
            response = client.get("/api/v1/health")
            
            if response.status_code != 200:
                print(f"Error: API service not available at {api_base_url}")
                print(f"Status code: {response.status_code}")
                return 1
                
            print("API connection successful")
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            print(f"Make sure the API service is running at {api_base_url}")
            return 1
        
        # Set up test parameters
        test_parameters = {}
        
        # Load parameters from file if specified
        if args.parameter_file:
            try:
                with open(args.parameter_file, 'r') as f:
                    test_parameters = json.load(f)
                print(f"Loaded parameters from {args.parameter_file}")
            except Exception as e:
                print(f"Error loading parameters from {args.parameter_file}: {str(e)}")
                return 1
        
        # Load parameters from command line if specified
        if args.parameters:
            try:
                cli_parameters = json.loads(args.parameters)
                # Update parameters with CLI values (overriding file values if both provided)
                test_parameters.update(cli_parameters)
                print("Applied parameters from command line")
            except Exception as e:
                print(f"Error parsing parameters from command line: {str(e)}")
                return 1
        
        # Set up model configuration
        model_config = {
            "model_id": args.model,
            "modality": "NLP",  # Default
            "sub_type": "Text Generation"  # Default
        }
        
        # Add additional model config if provided
        if args.model_config:
            try:
                additional_config = json.loads(args.model_config)
                model_config.update(additional_config)
                print("Applied additional model configuration")
            except Exception as e:
                print(f"Error parsing model configuration: {str(e)}")
                return 1
        
        # Prepare request data for the API
        request_data = {
            "test_id": test_id,
            "target_id": args.model,
            "target_type": "model",
            "target_parameters": model_config,
            "test_parameters": test_parameters
        }
        
        print(f"Sending test run request to API for model: {args.model}")
        
        # Create test run through API
        try:
            response = client.post("/api/v1/tests/runs", json=request_data)
            if response.status_code != 201 and response.status_code != 200:
                print(f"Error creating test run: {response.text}")
                return 1
                
            run_data = response.json()
            run_id = run_data.get("id")
            
            if not run_id:
                print("Error: No run ID returned from API")
                return 1
                
            print(f"Test run created with ID: {run_id}")
            
            # Poll for test results
            max_attempts = 60  # 5 minutes with 5-second intervals
            attempts = 0
            
            print("Waiting for test to complete...")
            
            while attempts < max_attempts:
                response = client.get(f"/api/v1/tests/runs/{run_id}")
                
                if response.status_code != 200:
                    print(f"Error checking test status: {response.text}")
                    return 1
                    
                run_status = response.json()
                status = run_status.get("status")
                
                if status == "completed":
                    print("Test completed successfully!")
                    break
                elif status == "failed":
                    print("Test failed to complete")
                    return 1
                elif status == "error":
                    print(f"Test encountered an error: {run_status.get('error', 'Unknown error')}")
                    return 1
                
                # Still running, wait and check again
                print(f"Test still running... (Status: {status})")
                time.sleep(5)
                attempts += 1
            
            if attempts >= max_attempts:
                print("Timeout waiting for test to complete")
                return 1
            
            # Get test results
            response = client.get(f"/api/v1/tests/runs/{run_id}/results")
            
            if response.status_code != 200:
                print(f"Error retrieving test results: {response.text}")
                return 1
                
            results = response.json()
            
            # Format and display results
            if "results" in results and len(results["results"]) > 0:
                result = results["results"][0]  # Get first result
                
                # Print result summary
                print("\nTest Result:")
                print(f"  Status: {result.get('status', 'unknown')}")
                print(f"  Score: {result.get('score', 'N/A')}")
                print(f"  Issues Found: {result.get('issues_found', 'N/A')}")
                
                # Save results to file if requested
                if args.output:
                    try:
                        with open(args.output, 'w') as f:
                            # Format based on requested output format
                            if args.format == 'json':
                                json.dump(result, f, indent=2)
                            elif args.format == 'yaml':
                                import yaml
                                yaml.dump(result, f)
                            else:
                                # Default to JSON
                                json.dump(result, f, indent=2)
                        print(f"\nResults saved to {args.output}")
                    except Exception as e:
                        print(f"Error saving results to {args.output}: {str(e)}")
                        return 1
                
                # Print detailed results if requested
                if args.verbose:
                    print("\nDetailed Analysis:")
                    if "analysis" in result and result["analysis"]:
                        for key, value in result["analysis"].items():
                            print(f"  {key}: {value}")
                    
                    print("\nMetrics:")
                    if "metrics" in result and result["metrics"]:
                        for key, value in result["metrics"].items():
                            print(f"  {key}: {value}")
            else:
                print("No test results returned from API")
                
            return 0
            
        except Exception as e:
            print(f"Error communicating with API: {str(e)}")
            logger.exception("API communication error")
            return 1
    
    except Exception as e:
        print(f"Error running test: {str(e)}")
        logger.exception("Run error")
        return 1


def list_api_models_command(args) -> int:
    """
    List models available in the API service.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set up API client configuration
        api_base_url = args.api_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        api_key = args.api_key or os.environ.get("API_KEY")
        
        # Set up API client
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Connect to API
        try:
            print(f"Connecting to API at {api_base_url}")
            client = httpx.Client(base_url=api_base_url, headers=headers, timeout=10.0)
            
            # Get models list
            response = client.get("/api/v1/models")
            
            if response.status_code != 200:
                print(f"Error fetching models: {response.text}")
                return 1
                
            models_data = response.json()
            models = models_data.get("models", [])
            
            if not models:
                print("No models found in the API service.")
                return 0
                
            # Display models
            print(f"Found {len(models)} models in the API service:")
            print("")
            
            for model in models:
                print(f"Model ID: {model.get('id')}")
                print(f"  Name: {model.get('name')}")
                print(f"  Description: {model.get('description', 'No description')}")
                print(f"  Modality: {model.get('modality')}")
                print(f"  Type: {model.get('sub_type')}")
                print("")
            
            return 0
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            print(f"Make sure the API service is running at {api_base_url}")
            return 1
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        logger.exception("API models list error")
        return 1


def list_api_tests_command(args) -> int:
    """
    List tests available in the API service.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set up API client configuration
        api_base_url = args.api_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        api_key = args.api_key or os.environ.get("API_KEY")
        
        # Set up API client
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Connect to API
        try:
            print(f"Connecting to API at {api_base_url}")
            client = httpx.Client(base_url=api_base_url, headers=headers, timeout=10.0)
            
            # Get tests list
            response = client.get("/api/v1/tests/registry")
            
            if response.status_code != 200:
                print(f"Error fetching tests from API: {response.text}")
                return 1
                
            tests_data = response.json()
            tests = tests_data.get("tests", [])
            
            if not tests:
                print("No tests found in the API service.")
                return 0
                
            # Display tests
            print(f"Found {len(tests)} tests in the API service:")
            print("")
            
            for test_id, test_info in tests.items():
                print(f"Test ID: {test_id}")
                print(f"  Name: {test_info.get('name')}")
                print(f"  Description: {test_info.get('description', 'No description')}")
                print(f"  Category: {test_info.get('category')}")
                print(f"  Modalities: {', '.join(test_info.get('compatible_modalities', []))}")
                print(f"  Model Types: {', '.join(test_info.get('compatible_sub_types', []))}")
                print("")
            
            return 0
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            print(f"Make sure the API service is running at {api_base_url}")
            return 1
    except Exception as e:
        print(f"Error listing API tests: {str(e)}")
        logger.exception("API tests list error")
        return 1


def register_with_api_command(args) -> int:
    """
    Register a local test with the API service.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # First load test from local registry
        if args.registry_path:
            registry.registry_path = args.registry_path
            registry.load_registry()
        
        test_id = args.test_id
        test_info = registry.get_test(test_id)
        
        if not test_info:
            print(f"Error: Test with ID '{test_id}' not found in local registry")
            print("Use 'custom-test list' to see available tests in local registry")
            return 1
        
        print(f"Found test in local registry: {test_info['name']} (ID: {test_id})")
        
        # Set up API client configuration
        api_base_url = args.api_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        api_key = args.api_key or os.environ.get("API_KEY")
        
        # Set up API client
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        # Connect to API
        try:
            print(f"Connecting to API at {api_base_url}")
            client = httpx.Client(base_url=api_base_url, headers=headers, timeout=10.0)
            
            # Prepare test data for API
            api_test_data = {
                "id": test_id,
                "name": test_info["name"],
                "description": test_info["description"],
                "category": test_info["category"],
                "compatible_modalities": test_info["compatible_modalities"],
                "compatible_sub_types": test_info["compatible_sub_types"],
                "parameter_schema": test_info["parameter_schema"],
                "default_config": test_info.get("default_config", {})
            }
            
            # Register test with API
            print(f"Registering test with API: {test_id}")
            response = client.post("/api/v1/tests/registry", json=api_test_data)
            
            if response.status_code not in (200, 201):
                print(f"Error registering test with API: {response.text}")
                return 1
                
            print(f"✅ Test {test_id} successfully registered with API service")
            return 0
            
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            print(f"Make sure the API service is running at {api_base_url}")
            return 1
    
    except Exception as e:
        print(f"Error registering test with API: {str(e)}")
        logger.exception("API registration error")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for the custom testing framework."
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Command to execute", 
        required=True
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", 
        help="Validate a custom test implementation"
    )
    validate_parser.add_argument(
        "file", 
        help="Path to the Python file containing the test class"
    )
    validate_parser.add_argument(
        "--module", "-m", 
        help="Module name (defaults to filename)"
    )
    validate_parser.add_argument(
        "--class", "-c", 
        dest="class_name",
        help="Class name to validate (defaults to first CustomTest subclass found)"
    )
    
    # Register command
    register_parser = subparsers.add_parser(
        "register", 
        help="Register a custom test implementation in local registry"
    )
    register_parser.add_argument(
        "file", 
        help="Path to the Python file containing the test class"
    )
    register_parser.add_argument(
        "--module", "-m", 
        help="Module name (defaults to filename)"
    )
    register_parser.add_argument(
        "--class", "-c", 
        dest="class_name",
        help="Class name to register (defaults to first CustomTest subclass found)"
    )
    register_parser.add_argument(
        "--test-id", "-i", 
        help="Custom test ID (defaults to class-defined ID)"
    )
    register_parser.add_argument(
        "--name", "-n", 
        help="Custom test name (defaults to class-defined name)"
    )
    register_parser.add_argument(
        "--description", "-d", 
        help="Custom test description (defaults to class-defined description)"
    )
    register_parser.add_argument(
        "--registry-path", "-r", 
        help="Path to store the registry data"
    )
    register_parser.add_argument(
        "--skip-validation", 
        action="store_true", 
        help="Skip test validation"
    )
    
    # List command
    list_parser = subparsers.add_parser(
        "list", 
        help="List tests in local registry"
    )
    list_parser.add_argument(
        "--registry-path", "-r", 
        help="Path to the registry data file"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run", 
        help="Run a registered test against a specified model"
    )
    run_parser.add_argument(
        "test_id", 
        help="ID of the registered test to run"
    )
    run_parser.add_argument(
        "--model", "-m", 
        required=True,
        help="Model name/ID to test against"
    )
    run_parser.add_argument(
        "--api-key", "-k", 
        help="API key for model access"
    )
    run_parser.add_argument(
        "--api-url", "-u",
        help="Base URL for the API service (default: http://localhost:8000)"
    )
    run_parser.add_argument(
        "--parameters", "-p", 
        help="Test parameters as JSON string"
    )
    run_parser.add_argument(
        "--parameter-file", "-f", 
        help="Path to JSON file with test parameters"
    )
    run_parser.add_argument(
        "--model-config", "-c", 
        help="Additional model configuration as JSON string"
    )
    run_parser.add_argument(
        "--output", "-o", 
        help="Path to save test results"
    )
    run_parser.add_argument(
        "--format", 
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format (defaults to json)"
    )
    run_parser.add_argument(
        "--registry-path", "-r", 
        help="Path to the registry data file"
    )
    
    # API Models command
    models_parser = subparsers.add_parser(
        "models", 
        help="List models available in the API service"
    )
    models_parser.add_argument(
        "--api-url", "-u",
        help="Base URL for the API service (default: http://localhost:8000)"
    )
    models_parser.add_argument(
        "--api-key", "-k", 
        help="API key for authentication"
    )
    
    # API Tests command
    api_tests_parser = subparsers.add_parser(
        "api-tests", 
        help="List tests available in the API service"
    )
    api_tests_parser.add_argument(
        "--api-url", "-u",
        help="Base URL for the API service (default: http://localhost:8000)"
    )
    api_tests_parser.add_argument(
        "--api-key", "-k", 
        help="API key for authentication"
    )
    
    # Register with API command
    register_api_parser = subparsers.add_parser(
        "register-api", 
        help="Register a local test with the API service"
    )
    register_api_parser.add_argument(
        "test_id", 
        help="ID of the test in local registry to register with API"
    )
    register_api_parser.add_argument(
        "--api-url", "-u",
        help="Base URL for the API service (default: http://localhost:8000)"
    )
    register_api_parser.add_argument(
        "--api-key", "-k", 
        help="API key for authentication"
    )
    register_api_parser.add_argument(
        "--registry-path", "-r", 
        help="Path to the local registry data file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == "validate":
        return validate_test_command(args)
    elif args.command == "register":
        return register_test_command(args)
    elif args.command == "list":
        return list_tests_command(args)
    elif args.command == "run":
        return run_test_command(args)
    elif args.command == "models":
        return list_api_models_command(args)
    elif args.command == "api-tests":
        return list_api_tests_command(args)
    elif args.command == "register-api":
        return register_with_api_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 