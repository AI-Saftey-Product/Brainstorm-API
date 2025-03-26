"""Performance monitoring utility for tracking operation timing and metrics."""
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Utility for monitoring and tracking performance metrics."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.operations = {}
        self.current_operations = {}
        self.metrics = {}
    
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.current_operations[operation_name] = {
            "start_time": time.time(),
            "start_datetime": datetime.utcnow()
        }
    
    def end_operation(self, operation_name: str) -> None:
        """End timing an operation and record its duration."""
        if operation_name not in self.current_operations:
            logger.warning(f"Operation {operation_name} was not started")
            return
        
        end_time = time.time()
        operation_data = self.current_operations.pop(operation_name)
        duration = end_time - operation_data["start_time"]
        
        if operation_name not in self.operations:
            self.operations[operation_name] = []
        
        self.operations[operation_name].append({
            "start_time": operation_data["start_datetime"],
            "end_time": datetime.utcnow(),
            "duration": duration
        })
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance metrics."""
        summary = {
            "operations": {},
            "metrics": {}
        }
        
        # Summarize operations
        for op_name, op_data in self.operations.items():
            durations = [op["duration"] for op in op_data]
            summary["operations"][op_name] = {
                "count": len(durations),
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0
            }
        
        # Summarize metrics
        for metric_name, values in self.metrics.items():
            if values:
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values) if all(isinstance(v, (int, float)) for v in values) else None,
                    "min": min(values) if all(isinstance(v, (int, float)) for v in values) else None,
                    "max": max(values) if all(isinstance(v, (int, float)) for v in values) else None
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all performance data."""
        self.operations = {}
        self.current_operations = {}
        self.metrics = {} 