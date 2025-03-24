"""Bias test implementations."""
from typing import Dict, Any, Optional

from .base_test import BaseBiasTest
from .data_provider import BiasTestDataProvider
from .honest_test import HONESTTest
from .cda_test import CDATest
from .intersectional_test import IntersectionalBiasTest
from .qa_test import QABiasTest
from .occupation_test import OccupationalBiasTest
from .multilingual_test import MultilingualBiasTest
from .intersect_bench_test import IntersectBenchTest
from .crows_pairs_test import CrowSPairsTest

__all__ = [
    'BaseBiasTest',
    'BiasTestDataProvider',
    'HONESTTest',
    'CDATest',
    'IntersectionalBiasTest',
    'QABiasTest',
    'OccupationalBiasTest',
    'MultilingualBiasTest',
    'IntersectBenchTest',
    'CrowSPairsTest',
    'run_bias_test'
]

async def run_bias_test(test_id: str, model_parameters: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a specific bias test based on the test ID.
    
    Args:
        test_id: The identifier for the bias test to run
        model_parameters: Parameters for the model being tested
        config: Optional configuration for the test
        
    Returns:
        Dict containing the test results
    """
    config = config or {}
    
    test_classes = {
        "nlp_honest_test": HONESTTest,
        "nlp_cda_test": CDATest,
        "nlp_intersectional_test": IntersectionalBiasTest,
        "nlp_qa_test": QABiasTest,
        "nlp_occupation_test": OccupationalBiasTest,
        "nlp_multilingual_test": MultilingualBiasTest,
        "nlp_intersect_bench_test": IntersectBenchTest,
        "nlp_crows_pairs_test": CrowSPairsTest
    }
    
    test_class = test_classes.get(test_id)
    if not test_class:
        raise ValueError(f"Unknown test ID: {test_id}")
    
    test_instance = test_class(config)
    return await test_instance._run_test_implementation(model_parameters) 