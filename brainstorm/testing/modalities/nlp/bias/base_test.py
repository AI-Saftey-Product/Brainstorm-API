"""Base class for bias testing."""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

from brainstorm.testing.base_test import BaseTest
from brainstorm.core.adapters.base_adapter import ModelAdapter
from brainstorm.core.websocket_manager import WebsocketManager

logger = logging.getLogger(__name__)

class BaseBiasTest(BaseTest):
    """Base class for all bias tests."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.test_type = "bias"  # Override base type
        self.test_category = "bias"  # Set test category
        
    async def generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Generate text with retry logic."""
        # Log the prompt with color coding
        logger.info(f"\033[94m[INPUT PROMPT]\033[0m: {prompt}")
        
        retry_count = 0
        max_total_retries = 5  # Maximum total retries to prevent infinite loops
        total_retries = 0
        
        while retry_count < max_retries and total_retries < max_total_retries:
            try:
                # Pass the prompt directly as a string, not wrapped in an object
                # This fixes the "invalid_type" error where the API expects a string for messages[0].content
                response = await self.model_adapter.generate(prompt)
                
                # Check if response is empty and provide a fallback
                if not response or response.strip() == "":
                    logger.warning(f"Empty response received for prompt: {prompt[:50]}...")
                    retry_count += 1
                    total_retries += 1
                    
                    # If we've tried too many times, provide a generic response
                    if retry_count >= max_retries or total_retries >= max_total_retries:
                        logger.warning("Providing fallback response after multiple empty responses")
                        response = f"I apologize, but I don't have a specific response to '{prompt[:30]}...'"
                else:
                    # Log the non-empty response with color coding
                    logger.info(f"\033[92m[MODEL RESPONSE]\033[0m: {response}")
                    return response
                    
            except Exception as e:
                retry_count += 1
                total_retries += 1
                logger.error(f"Error generating response (attempt {retry_count}): {str(e)}")
                if retry_count >= max_retries or total_retries >= max_total_retries:
                    logger.error(f"Failed to generate response after {retry_count} attempts")
                    return f"Error generating response: {str(e)}"
                    
            # Wait a bit before retrying
            await asyncio.sleep(0.5)
        
        # If we get here, we've hit our retry limits
        logger.warning("Hit maximum retry attempts, returning fallback response")
        return f"Unable to process prompt: '{prompt[:30]}...'. Please try again later."

    async def _send_progress_update(self, current: int, total: int, status: str, details: Dict[str, Any] = None):
        """Send progress update through websocket."""
        if self.websocket:
            message = {
                "type": "test_progress",
                "test_id": self.test_id,
                "test_type": self.test_type,
                "timestamp": datetime.utcnow().isoformat(),
                "progress": {
                    "current": current,
                    "total": total,
                    "percentage": (current / total * 100) if total > 0 else 0
                },
                "status": status
            }
            if details:
                message.update(details)
            
            await WebsocketManager.broadcast_json(message)

    async def _send_result(self, result: Dict[str, Any]):
        """Send test result through websocket."""
        if self.websocket:
            message = {
                "type": "test_result",
                "test_id": self.test_id,
                "test_type": self.test_type,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }
            await WebsocketManager.broadcast_json(message)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats
    
    async def run_test(self, model_adapter: ModelAdapter, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the test with the given model adapter and parameters."""
        self.start_time = datetime.utcnow()
        self.model = model_adapter
        
        try:
            # Send start notification
            await self._send_progress_update(0, 100, "started", {
                "message": f"Starting {self.test_type} test"
            })
            
            # Run the actual test implementation
            test_results = await self._run_test_implementation(model_parameters)
            
            # Update optimization stats
            end_time = datetime.utcnow()
            self.optimization_stats["total_time"] = (end_time - self.start_time).total_seconds()
            
            # Update metrics in the existing results
            if "metrics" not in test_results:
                test_results["metrics"] = {}
            test_results["metrics"].update({
                "total_time": self.optimization_stats["total_time"],
                "operation_count": self.optimization_stats["operation_count"]
            })
            
            # Ensure required fields exist
            test_results.setdefault("status", "success")
            test_results.setdefault("score", 0)
            test_results.setdefault("issues_found", 0)
            test_results.setdefault("results", [])
            test_results.setdefault("analysis", {})
            test_results.setdefault("n_examples", len(test_results.get("results", [])))
            test_results.setdefault("test_run_id", self.test_id)
            test_results.setdefault("test_id", self.test_type)
            test_results.setdefault("test_category", "bias")
            test_results.setdefault("test_name", "Unknown Test")
            test_results.setdefault("created_at", datetime.utcnow())
            
            # Send completion notification
            await self._send_progress_update(100, 100, "completed", {
                "message": f"Completed {self.test_type} test",
                "summary": {
                    "total_cases": test_results["n_examples"],
                    "issues_found": test_results["issues_found"],
                    "average_score": test_results["score"]
                }
            })
            
            # Send final results
            await self._send_result(test_results)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            error_result = self.create_result("Error", "error")
            error_result.update({
                "status": "error",
                "score": 0,
                "issues_found": 1,
                "results": [],
                "analysis": {"error": str(e)},
                "n_examples": 0,
                "metrics": {
                    "total_time": 0,
                    "operation_count": 0
                }
            })
            
            # Send error notification
            await self._send_progress_update(0, 100, "error", {
                "message": f"Test failed: {str(e)}"
            })
            await self._send_result(error_result)
            
            return error_result
    
    @abstractmethod
    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the actual test logic in subclasses."""
        raise NotImplementedError("Subclasses must implement _run_test_implementation()")
    
    def create_result(self, test_name: str, category: str = "bias") -> Dict[str, Any]:
        """Create a base result object."""
        import uuid
        return {
            "id": str(uuid.uuid4()),
            "test_run_id": self.test_id,
            "test_id": self.test_type,
            "test_name": test_name,
            "test_category": category,
            "status": "running",
            "score": 0.0,
            "issues_found": 0,
            "analysis": {},
            "metrics": {},
            "created_at": datetime.utcnow()
        } 