"""Caching system for model outputs."""

import logging
from typing import Dict, Any, Optional
from collections import OrderedDict
import time
import hashlib
import json

logger = logging.getLogger(__name__)

class OutputCache:
    """
    Cache for storing and retrieving model outputs.
    Implements LRU (Least Recently Used) caching strategy.
    """
    
    def __init__(self, cache_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            cache_size: Maximum number of items to store
            ttl: Time to live for cache entries in seconds
        """
        self.cache_size = cache_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def _generate_key(self, input_data: Any) -> str:
        """Generate a cache key from input data."""
        if isinstance(input_data, str):
            data_str = input_data
        else:
            try:
                data_str = json.dumps(input_data, sort_keys=True)
            except (TypeError, ValueError):
                data_str = str(input_data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, input_data: Any) -> Optional[Any]:
        """
        Retrieve item from cache if it exists and is not expired.
        
        Args:
            input_data: Input data to look up
            
        Returns:
            Cached output if found and valid, None otherwise
        """
        key = self._generate_key(input_data)
        
        if key in self.cache:
            # Check if entry has expired
            if time.time() - self.timestamps[key] > self.ttl:
                self._remove_item(key)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        return None
    
    def set(self, input_data: Any, output: Any) -> None:
        """
        Store item in cache.
        
        Args:
            input_data: Input data to use as key
            output: Output data to cache
        """
        key = self._generate_key(input_data)
        
        # If key exists, update it
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = output
            self.timestamps[key] = time.time()
            return
        
        # If cache is full, remove oldest item
        if len(self.cache) >= self.cache_size:
            self._remove_oldest()
        
        # Add new item
        self.cache[key] = output
        self.timestamps[key] = time.time()
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache and timestamps."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _remove_oldest(self) -> None:
        """Remove the least recently used item."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove_item(oldest_key)
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "ttl": self.ttl
        } 