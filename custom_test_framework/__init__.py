"""Custom testing framework for AI models."""

import logging
from custom_test_framework.base.registry import registry
from custom_test_framework.base.test_interface import CustomTest
from custom_test_framework.base.standardized_test import StandardizedCustomTest
from custom_test_framework.base.validator import TestValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__version__ = "0.1.0" 