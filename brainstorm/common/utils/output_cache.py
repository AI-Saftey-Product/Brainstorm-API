"""Output caching utility for storing and retrieving test outputs."""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class OutputCache:
    """Utility for caching test outputs."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the output cache.
        
        Args:
            max_size: Maximum number of items to store in cache
            ttl_seconds: Time-to-live in seconds for cache entries
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _make_cache_key(self, key_dict: Dict[str, Any]) -> str:
        """Create a string key from a dictionary."""
        try:
            # Sort dictionary to ensure consistent key generation
            sorted_items = sorted(key_dict.items())
            return json.dumps(sorted_items)
        except (TypeError, ValueError) as e:
            logger.warning(f"Error creating cache key: {e}")
            return str(key_dict)
    
    def set(self, key: Dict[str, Any], value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Dictionary to use as cache key
            value: Value to store
        """
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        cache_key = self._make_cache_key(key)
        self.cache[cache_key] = {
            "value": value,
            "timestamp": datetime.utcnow()
        }
    
    def get(self, key: Dict[str, Any]) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Dictionary to use as cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        cache_key = self._make_cache_key(key)
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        if datetime.utcnow() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
            # Entry has expired
            del self.cache[cache_key]
            return None
        
        return entry["value"]
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = datetime.utcnow()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "active_entries": sum(
                1 for entry in self.cache.values()
                if current_time - entry["timestamp"] <= timedelta(seconds=self.ttl_seconds)
            )
        } 