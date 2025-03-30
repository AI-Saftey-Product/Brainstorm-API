"""
Example script to demonstrate running a custom test programmatically.

This example shows how to use the API client to execute tests against models.
"""
import json
import logging
import time
from typing import Dict, Any, Optional
import httpx
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(
    test_id: str,
    model_name: str,
    parameters: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    api_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    timeout: int = 300  # 5 minutes max wait time
) -> Dict[str, Any]:
    """
    Run a custom test against a model through the API.
    
    Args:
        test_id: ID of the registered test to run
        model_name: Name/ID of the model to test
        parameters: Custom test parameters (optional)
        model_config: Additional model configuration (optional)
        api_url: Base URL for the API service
        api_key: API key for authentication (if required)
        timeout: Maximum time to wait for test completion in seconds
        
    Returns:
        Test result dictionary
    """
    logger.info(f"Running test '{test_id}' against model '{model_name}'")
    
    # Set up HTTP client
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    client = httpx.Client(base_url=api_url, headers=headers, timeout=30.0)
    
    # Check API connection
    try:
        logger.info(f"Connecting to API at {api_url}")
        response = client.get("/api/v1/health")
        if response.status_code != 200:
            raise ValueError(f"API not available at {api_url}: Status {response.status_code}")
        logger.info("API connection successful")
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")
        raise
    
    # Prepare request data
    if not model_config:
        model_config = {}
        
    request_data = {
        "test_id": test_id,
        "target_id": model_name,
        "target_type": "model",
        "target_parameters": {
            "model_id": model_name,
            "modality": "NLP",
            "sub_type": "Text Generation",
            **model_config
        },
        "test_parameters": parameters or {}
    }
    
    # Create test run
    logger.info("Creating test run...")
    response = client.post("/api/v1/tests/runs", json=request_data)
    if response.status_code not in (200, 201):
        error_msg = f"Failed to create test run: {response.text}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    run_data = response.json()
    run_id = run_data.get("id")
    
    if not run_id:
        raise ValueError("No run ID returned from API")
    
    logger.info(f"Test run created with ID: {run_id}")
    
    # Poll for completion
    poll_interval = 5  # seconds
    max_attempts = timeout // poll_interval
    attempts = 0
    
    logger.info("Waiting for test to complete...")
    
    while attempts < max_attempts:
        response = client.get(f"/api/v1/tests/runs/{run_id}")
        
        if response.status_code != 200:
            error_msg = f"Error checking test status: {response.text}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        run_status = response.json()
        status = run_status.get("status")
        
        if status == "completed":
            logger.info("Test completed successfully!")
            break
        elif status in ("failed", "error"):
            error_msg = f"Test {status}: {run_status.get('error', 'Unknown error')}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Test still running... (Status: {status})")
        time.sleep(poll_interval)
        attempts += 1
    
    if attempts >= max_attempts:
        raise TimeoutError(f"Test did not complete within {timeout} seconds")
    
    # Get test results
    logger.info("Retrieving test results...")
    response = client.get(f"/api/v1/tests/runs/{run_id}/results")
    
    if response.status_code != 200:
        error_msg = f"Error retrieving test results: {response.text}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    results = response.json()
    
    if "results" in results and len(results["results"]) > 0:
        result = results["results"][0]  # Get first result
        logger.info(f"Test result retrieved: status={result.get('status')}, score={result.get('score')}")
        return result
    else:
        logger.warning("No test results returned from API")
        return {"status": "unknown", "message": "No results returned"}


def main():
    """Run the example."""
    # Example configuration - replace with your actual values
    test_id = "custom_content_moderation_test"
    model_name = "gpt-3.5-turbo"
    api_url = "http://localhost:8000"  # Replace with your API URL
    api_key = None  # Replace with your API key if required
    
    # Custom parameters (optional)
    parameters = {
        "n_examples": 3,
        "content_types": ["profanity", "hate_speech"],
        "severity_levels": ["low", "medium", "high"]
    }
    
    # Additional model configuration (optional)
    model_config = {
        "temperature": 0.2,
        "max_tokens": 200
    }
    
    # Run the test
    try:
        result = run_test(
            test_id=test_id,
            model_name=model_name,
            parameters=parameters,
            model_config=model_config,
            api_url=api_url,
            api_key=api_key
        )
        
        # Print the result summary
        print("\nTest Result:")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Issues Found: {result.get('issues_found', 'N/A')}")
        
        # Save to file if desired
        # with open("test_result.json", "w") as f:
        #     json.dump(result, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error running test: {e}")


if __name__ == "__main__":
    main() 