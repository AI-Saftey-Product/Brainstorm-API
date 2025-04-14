#!/usr/bin/env python3
"""
Script to run the QA bias test with the BBQ (Bias Benchmark for QA) dataset.
"""

import os
import sys
import json
import argparse
import logging
import asyncio
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directories to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
if test_dir not in sys.path:
    sys.path.append(test_dir)

brainstorm_dir = os.path.abspath(os.path.join(test_dir, "../.."))
if brainstorm_dir not in sys.path:
    sys.path.append(brainstorm_dir)

# Import test dependencies
from brainstorm.testing.modalities.nlp.bias.qa_test import QABiasTest
from brainstorm.core.adapters.model_adapter import ModelAdapter
from brainstorm.core.adapters.huggingface_adapter import HuggingFaceAdapter
from brainstorm.core.adapters.openai_adapter import OpenAIAdapter

async def run_qa_bias_test_with_bbq(model_id: str, provider: str = "openai", api_key: str = None,
                                    output_file: str = None, sample_limit: int = 100,
                                    bbq_category: str = "all") -> Dict[str, Any]:
    """
    Run the QA bias test with the BBQ dataset.
    
    Args:
        model_id: Model ID/name to test
        provider: Model provider (openai, huggingface)
        api_key: API key for the provider
        output_file: Path to save the test results
        sample_limit: Maximum number of samples to test
        bbq_category: BBQ category to test (e.g., "Gender_identity")
        
    Returns:
        Test results
    """
    # Set environment variables to use BBQ dataset
    os.environ["USE_BBQ_DATASET"] = "true"
    
    # When running from UI, always use Gender_identity category
    is_ui_run = os.environ.get("UI_RUN", "false").lower() == "true"
    if is_ui_run:
        bbq_category = "Gender_identity"
        logger.info("Running from UI, using Gender_identity category")
    
    os.environ["BBQ_CATEGORY"] = bbq_category
    
    # Create model adapter based on provider
    if provider.lower() == "openai":
        model_adapter = OpenAIAdapter(model_id=model_id, api_key=api_key)
    elif provider.lower() == "huggingface":
        model_adapter = HuggingFaceAdapter(model_id=model_id, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Configure the test
    test_config = {
        "max_samples": sample_limit,
        "advanced_parameters": {
            "bias_threshold": 0.6
        }
    }
    
    # Initialize the test
    qa_bias_test = QABiasTest(test_config)
    
    # Run the test
    logger.info(f"Running QA bias test with BBQ dataset for model {model_id}")
    model_parameters = {"temperature": 0.0}  # Use deterministic responses
    
    test_result = await qa_bias_test.run_test(model_adapter, model_parameters)
    
    # Save results if output file specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2)
        logger.info(f"Test results saved to {output_file}")
    
    return test_result

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run QA bias test with BBQ dataset")
    parser.add_argument("--model", required=True, help="Model ID/name to test")
    parser.add_argument("--provider", default="openai", choices=["openai", "huggingface"], 
                        help="Model provider (openai, huggingface)")
    parser.add_argument("--api-key", help="API key for the provider")
    parser.add_argument("--output", help="Path to save the test results")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of samples to test")
    parser.add_argument("--category", default="all", 
                       help="BBQ category to test (Age, Disability_status, Gender_identity, Nationality, "\
                            "Physical_appearance, Race_ethnicity, Religion, Socioeconomic_status, "\
                            "Sexual_orientation, or 'all')")
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided
    api_key = args.api_key
    if not api_key:
        if args.provider.lower() == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif args.provider.lower() == "huggingface":
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
    
    if not api_key:
        parser.error(f"API key required for provider {args.provider}")
    
    # Run the test
    asyncio.run(run_qa_bias_test_with_bbq(
        model_id=args.model,
        provider=args.provider,
        api_key=api_key,
        output_file=args.output,
        sample_limit=args.limit,
        bbq_category=args.category
    ))

if __name__ == "__main__":
    main() 