"""Resource management for concurrent operations."""

import logging
import asyncio
from typing import Callable, Any, Dict, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages concurrent operations and resource utilization.
    Implements rate limiting and concurrency control.
    """
    
    def __init__(self, max_concurrent: int = 3, rate_limit: Optional[int] = None):
        """
        Initialize the resource manager.
        
        Args:
            max_concurrent: Maximum number of concurrent operations
            rate_limit: Maximum operations per second (optional)
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = 0
        self.last_operation_time = 0
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire(self):
        """
        Context manager for acquiring resources.
        
        Usage:
            async with resource_manager.acquire():
                # do work
        """
        try:
            async with self._lock:
                self.active_operations += 1
                logger.debug(f"Active operations: {self.active_operations}")
            
            async with self.semaphore:
                if self.rate_limit:
                    await self._apply_rate_limit()
                yield
                
        finally:
            async with self._lock:
                self.active_operations -= 1
    
    async def _apply_rate_limit(self):
        """Apply rate limiting if configured."""
        if self.rate_limit:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_operation_time
            
            if time_since_last < (1.0 / self.rate_limit):
                delay = (1.0 / self.rate_limit) - time_since_last
                await asyncio.sleep(delay)
            
            self.last_operation_time = asyncio.get_event_loop().time()
    
    async def run_with_resources(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function with resource management.
        
        Args:
            func: Async function to run
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of the function call
        """
        async with self.acquire():
            return await func(*args, **kwargs)
    
    async def run_batch(self, funcs: list[Callable], batch_size: Optional[int] = None) -> list:
        """
        Run a batch of functions with resource management.
        
        Args:
            funcs: List of async functions to run
            batch_size: Optional batch size (defaults to max_concurrent)
            
        Returns:
            List of results from all functions
        """
        batch_size = batch_size or self.max_concurrent
        results = []
        
        for i in range(0, len(funcs), batch_size):
            batch = funcs[i:i + batch_size]
            batch_results = await asyncio.gather(
                *(self.run_with_resources(func) for func in batch)
            )
            results.extend(batch_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current resource utilization statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_operations": self.active_operations,
            "rate_limit": self.rate_limit
        } 