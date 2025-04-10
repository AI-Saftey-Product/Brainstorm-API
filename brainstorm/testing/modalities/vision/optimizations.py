"""Optimization utilities for vision tests."""
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing model instances."""
    
    _instance = None
    _models = {}
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ModelRegistry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_model(self, model_id: str, model: Any):
        """Register a model instance."""
        self._models[model_id] = model
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a registered model instance."""
        return self._models.get(model_id)
    
    def clear(self):
        """Clear all registered models."""
        self._models.clear()

class OutputCache:
    """Cache for model outputs to avoid redundant computations."""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._max_age = 3600  # 1 hour in seconds
    
    def get(self, key: str) -> Optional[str]:
        """Get a cached output if it exists and is not expired."""
        if key in self._cache:
            timestamp = self._timestamps.get(key)
            if timestamp and (datetime.utcnow() - timestamp).total_seconds() < self._max_age:
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache an output with current timestamp."""
        self._cache[key] = value
        self._timestamps[key] = datetime.utcnow()
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._timestamps.clear()

class ResourceManager:
    """Manager for controlling concurrent resource usage."""
    
    def __init__(self, max_concurrent: int = 3, rate_limit: Optional[float] = None):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = defaultdict(float)
    
    async def acquire(self):
        """Acquire a resource slot."""
        await self._semaphore.acquire()
    
    def release(self):
        """Release a resource slot."""
        self._semaphore.release()
    
    async def wait_for_rate_limit(self, resource_id: str):
        """Wait if necessary to respect rate limits."""
        if self.rate_limit is None:
            return
            
        current_time = datetime.utcnow().timestamp()
        last_request = self._last_request_time[resource_id]
        time_since_last = current_time - last_request
        
        if time_since_last < (1.0 / self.rate_limit):
            await asyncio.sleep((1.0 / self.rate_limit) - time_since_last)
        
        self._last_request_time[resource_id] = datetime.utcnow().timestamp()

class PerformanceMonitor:
    """Monitor for tracking test performance metrics."""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._start_times = {}
    
    def start_operation(self, operation_id: str):
        """Start timing an operation."""
        self._start_times[operation_id] = datetime.utcnow()
    
    def end_operation(self, operation_id: str):
        """End timing an operation and record its duration."""
        if operation_id in self._start_times:
            duration = (datetime.utcnow() - self._start_times[operation_id]).total_seconds()
            self._metrics[operation_id].append(duration)
            del self._start_times[operation_id]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            operation: {
                "count": len(times),
                "total_time": sum(times),
                "average_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0
            }
            for operation, times in self._metrics.items()
        }
    
    def clear(self):
        """Clear all metrics."""
        self._metrics.clear()
        self._start_times.clear() 