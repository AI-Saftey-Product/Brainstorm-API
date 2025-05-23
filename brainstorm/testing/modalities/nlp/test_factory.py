"""Factory for creating test instances."""
from typing import Dict, Any, Optional

from brainstorm.testing.base_test import BaseTest
from brainstorm.testing.modalities.nlp.bias.honest_test import HONESTTest
from brainstorm.testing.modalities.nlp.bias.cda_test import CDATest
from brainstorm.testing.modalities.nlp.bias.intersectional_test import IntersectionalBiasTest
from brainstorm.testing.modalities.nlp.bias.qa_test import QABiasTest
from brainstorm.testing.modalities.nlp.bias.occupational_test import OccupationalBiasTest
from brainstorm.testing.modalities.nlp.security.prompt_injection_test import PromptInjectionTest
from brainstorm.testing.modalities.nlp.security.jailbreak_test import JailbreakTest
from brainstorm.testing.modalities.nlp.security.data_extraction_test import DataExtractionTest
from brainstorm.testing.modalities.nlp.security.strong_reject_test import StrongRejectTest


class TestFactory:
    """Factory for creating test instances."""
    
    @staticmethod
    def create_test(test_id: str, config: Optional[Dict[str, Any]] = None) -> BaseTest:
        """Create a test instance based on test ID."""
        config = config or {}
        
        # Bias tests
        if test_id == "honest":
            return HONESTTest(config)
        elif test_id == "counterfactual":
            return CDATest(config)
        elif test_id == "intersectional":
            return IntersectionalBiasTest(config)
        elif test_id == "qa":
            return QABiasTest(config)
        elif test_id == "occupation":
            return OccupationalBiasTest(config)
        
        # Security tests
        elif test_id == "prompt_injection_test":
            return PromptInjectionTest(config)
        elif test_id == "jailbreak_test":
            return JailbreakTest(config)
        elif test_id == "data_extraction_test":
            return DataExtractionTest(config)
        elif test_id == "strong_reject_test":
            return StrongRejectTest(config)
        
        raise ValueError(f"Unknown test ID: {test_id}")
    
    @staticmethod
    def create_tests(test_ids: list, config: Optional[Dict[str, Any]] = None) -> list:
        """Create multiple test instances."""
        return [TestFactory.create_test(test_id, config) for test_id in test_ids] 