"""Standardized test implementation that follows the standard test flow."""
from abc import abstractmethod
from typing import Dict, Any, List, Optional
import logging

from custom_test_framework.base.test_interface import CustomTest

logger = logging.getLogger(__name__)


class StandardizedCustomTest(CustomTest):
    """
    Base class for custom tests that follow the standard test execution pattern.
    
    This class ensures custom tests match the execution flow of built-in tests.
    """
    
    def __init__(self, test_id=None, name=None, description=None):
        """Initialize the standardized custom test."""
        super().__init__(test_id=test_id, name=name, description=description)
        
    async def run(self, model_adapter: Any, test_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the test following the standard execution pattern.
        
        This method follows the same structure as built-in tests:
        1. Parameter validation
        2. Test preparation
        3. Model invocation
        4. Result analysis
        5. Standardized result formatting
        
        Args:
            model_adapter: The model adapter to test
            test_parameters: Test configuration parameters
            
        Returns:
            Test results with standardized format
        """
        try:
            # Step 1: Parameter validation
            validated_params = self.validate_parameters(test_parameters)
            
            # Step 2: Test preparation
            test_inputs = await self.prepare_test_inputs(validated_params)
            
            # Step 3: Model invocation - same pattern as built-in tests
            results = []
            for test_input in test_inputs:
                try:
                    # Generate output using model adapter
                    response = await model_adapter.generate(
                        test_input["prompt"],
                        **validated_params.get("model_parameters", {})
                    )
                    
                    # Evaluate the response
                    evaluation = await self.evaluate_response(
                        test_input,
                        response,
                        validated_params
                    )
                    
                    # Add to results
                    results.append({
                        "input": test_input["prompt"],
                        "output": response,
                        "expected": test_input.get("expected", ""),
                        "evaluation": evaluation
                    })
                    
                except Exception as e:
                    logger.error(f"Error evaluating test input: {str(e)}")
                    # Handle individual test case errors
                    results.append({
                        "input": test_input["prompt"],
                        "output": f"Error: {str(e)}",
                        "expected": test_input.get("expected", ""),
                        "evaluation": {"error": str(e)}
                    })
            
            # Step 4: Result analysis - standardized approach
            analysis_result = await self.analyze_results(results, validated_params)
            
            # Step 5: Return formatted result - matches built-in test format
            return self.format_result(
                status="success",
                score=analysis_result["score"],
                metrics=analysis_result["metrics"],
                issues_found=analysis_result["issues_found"],
                analysis=analysis_result["analysis"]
            )
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            # Error handling follows the same pattern as built-in tests
            return self.format_result(
                status="error",
                score=0,
                metrics={},
                issues_found=1,
                analysis={"error": str(e)}
            )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against the test's schema.
        
        Args:
            parameters: Test parameters to validate
            
        Returns:
            Validated parameters with defaults applied
        """
        # In a full implementation, this would use schema validation
        # For now, we'll implement a simple version
        schema = self.get_parameter_schema()
        defaults = self.get_default_config()
        
        # Apply defaults for missing parameters
        validated = defaults.copy()
        validated.update(parameters)
        
        return validated
    
    @abstractmethod
    async def prepare_test_inputs(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare test inputs based on parameters.
        
        Args:
            parameters: Validated test parameters
            
        Returns:
            List of test input dictionaries
        """
        pass
    
    @abstractmethod
    async def evaluate_response(self, 
                              test_input: Dict[str, Any], 
                              response: str,
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a model response for a single test input.
        
        Args:
            test_input: Test input dictionary
            response: Model's response
            parameters: Test parameters
            
        Returns:
            Evaluation results for this input
        """
        pass
    
    @abstractmethod
    async def analyze_results(self, 
                            results: List[Dict[str, Any]],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze all results to produce overall metrics and score.
        
        Args:
            results: List of individual test results
            parameters: Test parameters
            
        Returns:
            Analysis dictionary with score, metrics, issues_found, and analysis
        """
        pass 