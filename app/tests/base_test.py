"""Base test class for all AI safety tests."""
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from app.core.model_adapter import ModelAdapter
from app.core.websocket_manager import WebsocketManager

logger = logging.getLogger(__name__)

class BaseTest(ABC):
    """Base class for all AI safety tests."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        self.config = config
        self.optimization_stats = {
            "total_time": 0,
            "operation_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.results = []
        self.start_time = None
        self.websocket = config.get("websocket")
        self.test_id = config.get("test_id", str(uuid4()))
        self.test_type = "base"  # Should be overridden by subclasses
        self.test_category = "unknown"  # Should be overridden by subclasses

    async def _send_progress_update(self, current: int, total: int, status: str, details: Dict[str, Any] = None):
        """Send progress update through websocket."""
        if self.websocket:
            message = {
                "type": "test_progress",
                "test_id": self.test_id,
                "test_type": self.test_type,
                "test_category": self.test_category,
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
                "test_category": self.test_category,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }
            await WebsocketManager.broadcast_json(message)

    async def _send_issue_found(self, issue_type: str, details: Dict[str, Any]):
        """Send issue notification through websocket."""
        if self.websocket:
            message = {
                "type": "issue_found",
                "test_id": self.test_id,
                "test_type": self.test_type,
                "test_category": self.test_category,
                "timestamp": datetime.utcnow().isoformat(),
                "issue_type": issue_type,
                "details": details
            }
            await WebsocketManager.broadcast_json(message)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats

    def create_result(self, test_name: str, test_category: str) -> Dict[str, Any]:
        """Create a standardized result structure."""
        return {
            "test_run_id": self.test_id,
            "test_id": self.test_type,
            "test_category": test_category,
            "test_name": test_name,
            "created_at": datetime.utcnow(),
            "status": "success",
            "score": 0,
            "issues_found": 0,
            "results": [],
            "analysis": {},
            "metrics": {
                "total_time": 0,
                "operation_count": 0
            }
        }

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
            test_results.setdefault("test_category", self.test_category)
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
                "issues_found": 0,
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