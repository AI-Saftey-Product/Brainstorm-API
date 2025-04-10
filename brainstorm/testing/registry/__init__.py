"""Test registry module for AI safety tests."""

from brainstorm.testing.registry.registry import test_registry
from functools import wraps
from typing import Callable, Any, TypeVar, cast

T = TypeVar('T')

def register_test(func: Callable[..., Any] = None) -> Callable[..., Any]:
    """Decorator for registering tests.
    
    This is a compatibility decorator that doesn't do anything but ensures imports don't break.
    In the future, this may be used for automatic registration of tests.
    """
    if func is None:
        return register_test
        
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    
    return cast(T, wrapper)