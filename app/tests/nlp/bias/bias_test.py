"""Main bias test runner module."""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid

from app.core.model_adapter import ModelAdapter
from app.tests.nlp.bias.evaluators import (
    create_evaluator,
    HONESTEvaluator,
    CDAEvaluator,
    IntersectionalEvaluator,
    QAEvaluator,
    OccupationalEvaluator,
    MultilingualEvaluator
)
from app.tests.nlp.bias.data_provider import BiasTestDataProvider

logger = logging.getLogger(__name__)

async def run_bias_test(
    model_name: str,
    test_types: List[str] = None,
    max_samples: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """Run comprehensive bias testing on a language model.
    
    Args:
        model_name: Name or path of model to test
        test_types: List of test types to run (if None, runs all tests)
        max_samples: Maximum number of samples per test
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
    
    Returns:
        Dictionary containing test results and metrics
    """
    logger.info(f"Running bias test suite for model: {model_name}")
    logger.info(f"Parameters: max_samples={max_samples}, temperature={temperature}, top_p={top_p}")
    
    # Initialize model adapter
    try:
        model = ModelAdapter(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return {
            "status": "error",
            "error": str(e),
            "created_at": datetime.utcnow().isoformat()
        }
    
    # Initialize data provider
    data_provider = BiasTestDataProvider()
    
    # Default to all test types if none specified
    if test_types is None:
        test_types = ["honest", "counterfactual", "intersectional", "qa", "occupation", "multilingual"]
    
    # Initialize evaluators
    evaluators = {
        test_type: create_evaluator(test_type)
        for test_type in test_types
    }
    
    # Run tests concurrently
    async def run_test(test_type: str) -> Dict[str, Any]:
        try:
            evaluator = evaluators[test_type]
            test_data = data_provider.get_test_data(test_type, max_samples)
            
            if test_type == "honest":
                responses = []
                for prompt in test_data["prompts"]:
                    response = model.generate(prompt)
                    result = evaluator.evaluate(prompt, response)
                    responses.append(result)
                    
            elif test_type == "counterfactual":
                responses = []
                for prompt_pair in test_data["prompt_pairs"]:
                    resp1 = model.generate(prompt_pair[0])
                    resp2 = model.generate(prompt_pair[1])
                    result = evaluator.evaluate(resp1, resp2)
                    responses.append(result)
                    
            elif test_type == "intersectional":
                responses = []
                for prompt_set in test_data["prompt_sets"]:
                    resp_dict = {
                        group: model.generate(prompt)
                        for group, prompt in prompt_set.items()
                    }
                    result = evaluator.evaluate(resp_dict)
                    responses.append(result)
                    
            elif test_type == "qa":
                responses = []
                for qa_item in test_data["qa_items"]:
                    response = model.generate(qa_item["question"])
                    result = evaluator.evaluate(
                        response,
                        qa_item["stereotypical_terms"],
                        qa_item["counter_stereotypical_terms"]
                    )
                    responses.append(result)
                    
            elif test_type == "occupation":
                responses = []
                for prompt in test_data["prompts"]:
                    response = model.generate(prompt)
                    result = evaluator.evaluate(response)
                    responses.append(result)
                    
            elif test_type == "multilingual":
                responses = []
                for prompt_set in test_data["prompt_sets"]:
                    resp_dict = {
                        lang: model.generate(prompt)
                        for lang, prompt in prompt_set.items()
                    }
                    result = evaluator.evaluate(resp_dict)
                    responses.append(result)
            
            # Calculate aggregate metrics
            total_samples = len(responses)
            biased_samples = sum(1 for r in responses if r["is_biased"])
            avg_bias_score = sum(r["bias_score"] for r in responses) / total_samples if total_samples > 0 else 0
            
            return {
                "test_type": test_type,
                "total_samples": total_samples,
                "biased_samples": biased_samples,
                "bias_rate": biased_samples / total_samples if total_samples > 0 else 0,
                "average_bias_score": avg_bias_score,
                "detailed_results": responses
            }
            
        except Exception as e:
            logger.error(f"Error running {test_type} test: {e}")
            return {
                "test_type": test_type,
                "status": "error",
                "error": str(e)
            }
    
    # Run all tests concurrently
    test_results = await asyncio.gather(*[
        run_test(test_type) for test_type in test_types
    ])
    
    # Calculate overall metrics
    total_samples = sum(r["total_samples"] for r in test_results if "total_samples" in r)
    total_biased = sum(r["biased_samples"] for r in test_results if "biased_samples" in r)
    avg_bias_score = sum(r["average_bias_score"] * r["total_samples"] 
                        for r in test_results if "average_bias_score" in r) / total_samples if total_samples > 0 else 0
    
    return {
        "id": str(uuid.uuid4()),
        "model_name": model_name,
        "status": "success",
        "overall_metrics": {
            "total_samples": total_samples,
            "total_biased_samples": total_biased,
            "overall_bias_rate": total_biased / total_samples if total_samples > 0 else 0,
            "average_bias_score": avg_bias_score
        },
        "test_results": {
            result["test_type"]: result
            for result in test_results
        },
        "parameters": {
            "max_samples": max_samples,
            "temperature": temperature,
            "top_p": top_p,
            "test_types": test_types
        },
        "model_info": model.get_model_info(),
        "created_at": datetime.utcnow().isoformat()
    } 