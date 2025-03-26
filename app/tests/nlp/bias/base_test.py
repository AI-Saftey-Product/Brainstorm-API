"""Base class for bias tests."""
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
import logging
import asyncio
from abc import ABC, abstractmethod

from app.core.model_adapter import ModelAdapter
from app.tests.base_test import BaseTest

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
        for attempt in range(max_retries):
            try:
                response = await self.model.generate(prompt, **kwargs)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)  # Wait before retry
        return ""

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
        return {
            "id": str(uuid4()),
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