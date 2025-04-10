"""Base class for data providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

class DataProvider(ABC):
    """Base class for dataset providers that supply test inputs."""
    
    @abstractmethod
    def get_examples(self, task_type: str, n_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples for a specific task type."""
        pass 