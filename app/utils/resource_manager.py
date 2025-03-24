"""Resource management utility for controlling concurrent access to resources."""
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResourceManager:
    """Utility for managing concurrent access to resources."""
    
    def __init__(self, max_concurrent: int = 3):
        """
        Initialize the resource manager.
        
        Args:
            max_concurrent: Maximum number of concurrent operations allowed
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = 0
        self.total_operations = 0
        self.max_concurrent = max_concurrent
    
    async def acquire(self):
        """Acquire access to resources."""
        await self.semaphore.acquire()
        self.active_operations += 1
        self.total_operations += 1
        logger.debug(f"Resource acquired. Active: {self.active_operations}, Total: {self.total_operations}")
        return self
    
    def release(self):
        """Release access to resources."""
        self.semaphore.release()
        self.active_operations -= 1
        logger.debug(f"Resource released. Active: {self.active_operations}, Total: {self.total_operations}")
    
    async def __aenter__(self):
        """Enter the async context manager."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        self.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return {
            "active_operations": self.active_operations,
            "total_operations": self.total_operations,
            "max_concurrent": self.max_concurrent
        }
    
    def reset(self) -> None:
        """Reset resource usage statistics."""
        self.active_operations = 0
        self.total_operations = 0 