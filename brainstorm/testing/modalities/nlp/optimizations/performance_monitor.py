"""Performance monitoring for test operations."""

import logging
import time
from typing import Dict, Any, List, Callable
from collections import defaultdict
import statistics
import asyncio

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for test operations.
    Provides insights into bottlenecks and performance patterns.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = defaultdict(list)
        self.start_times = {}
        self._lock = asyncio.Lock()
    
    async def measure_operation(self, operation: str, func: Callable) -> Any:
        """
        Measure the execution time of an operation.
        
        Args:
            operation: Name of the operation being measured
            func: Async function to execute and measure
            
        Returns:
            Result of the function call
        """
        start_time = time.time()
        try:
            result = await func()
            return result
        finally:
            duration = time.time() - start_time
            await self._record_timing(operation, duration)
    
    async def _record_timing(self, operation: str, duration: float) -> None:
        """Record timing for an operation."""
        async with self._lock:
            self.metrics[operation].append(duration)
    
    def start_operation(self, operation: str) -> None:
        """Mark the start of an operation."""
        self.start_times[operation] = time.time()
    
    def end_operation(self, operation: str) -> None:
        """
        Mark the end of an operation and record its duration.
        
        Args:
            operation: Name of the operation to end
        """
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)
            del self.start_times[operation]
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical analysis of recorded metrics.
        
        Returns:
            Dictionary containing statistics for each operation
        """
        stats = {}
        for operation, timings in self.metrics.items():
            if timings:
                stats[operation] = {
                    'count': len(timings),
                    'avg': statistics.mean(timings),
                    'min': min(timings),
                    'max': max(timings),
                    'median': statistics.median(timings),
                    'total': sum(timings)
                }
                
                # Add standard deviation if we have enough samples
                if len(timings) > 1:
                    stats[operation]['std_dev'] = statistics.stdev(timings)
                    
        return stats
    
    def get_bottlenecks(self, threshold: float = 1.0) -> List[str]:
        """
        Identify operations that might be bottlenecks.
        
        Args:
            threshold: Time threshold in seconds to consider as bottleneck
            
        Returns:
            List of operation names that exceed the threshold
        """
        stats = self.get_statistics()
        return [
            operation 
            for operation, metrics in stats.items() 
            if metrics['avg'] > threshold
        ]
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        self.start_times.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        stats = self.get_statistics()
        
        total_time = sum(
            metrics['total'] 
            for metrics in stats.values()
        )
        
        return {
            'total_time': total_time,
            'operation_count': len(stats),
            'stats': stats,
            'bottlenecks': self.get_bottlenecks()
        } 