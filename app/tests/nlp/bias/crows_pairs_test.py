from typing import Dict, Any, List, Optional
import logging
import json
from .base_test import BaseBiasTest
from .evaluators import create_evaluator

logger = logging.getLogger(__name__)

class CrowSPairsTest(BaseBiasTest):
    """
    Implementation of CrowS-Pairs bias test that evaluates model bias through
    stereotype and anti-stereotype sentence pairs.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CrowS-Pairs test with configuration.
        
        Args:
            config: Dictionary containing test configuration including:
                - max_samples: Maximum number of test cases to evaluate
                - bias_threshold: Threshold for determining significant bias
        """
        super().__init__(config)
        self.max_samples = config.get("max_samples", 100)
        self.bias_threshold = config.get("bias_threshold", 0.1)
        self.evaluator = create_evaluator("crows_pairs")

    async def _run_test_implementation(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the CrowS-Pairs test by evaluating model outputs on stereotype and
        anti-stereotype pairs.

        Args:
            test_data: Dictionary containing test cases with stereotype/anti-stereotype pairs

        Returns:
            Dictionary containing test results including:
                - overall_bias_score
                - category_scores
                - detailed_results
                - recommendations
        """
        logger.info("Starting CrowS-Pairs bias test")
        
        test_cases = test_data.get("test_cases", [])
        if not test_cases:
            raise ValueError("No test cases found in test data")

        # Limit number of test cases if max_samples is set
        test_cases = test_cases[:self.max_samples]
        
        results = {
            "test_id": "crows_pairs_test",
            "test_name": "CrowS-Pairs Bias Test",
            "test_category": "bias",
            "created_at": self._get_current_time(),
            "total_examples": len(test_cases),
            "bias_scores": {},
            "category_scores": {},
            "detailed_results": [],
            "issues_found": 0
        }

        # Track scores by category
        category_scores = {}
        
        for test_case in test_cases:
            case_id = test_case["id"]
            category = test_case["category"]
            stereotype = test_case["stereotype"]
            anti_stereotype = test_case["anti_stereotype"]
            
            logger.info(f"Processing test case {case_id} for category {category}")
            
            # Get model outputs for both versions
            stereotype_output = await self._generate_text(stereotype)
            anti_stereotype_output = await self._generate_text(anti_stereotype)
            
            # Evaluate bias in the outputs
            bias_score = await self.evaluator.evaluate(
                stereotype_output,
                anti_stereotype_output,
                stereotype,
                anti_stereotype
            )
            
            # Track scores by category
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(bias_score)
            
            # Record detailed results
            result = {
                "id": case_id,
                "category": category,
                "bias_type": test_case["bias_type"],
                "target": test_case["target"],
                "stereotype": stereotype,
                "anti_stereotype": anti_stereotype,
                "stereotype_output": stereotype_output,
                "anti_stereotype_output": anti_stereotype_output,
                "bias_score": bias_score,
                "has_bias": abs(bias_score) > self.bias_threshold
            }
            
            results["detailed_results"].append(result)
            if result["has_bias"]:
                results["issues_found"] += 1

        # Calculate average scores per category
        for category, scores in category_scores.items():
            results["category_scores"][category] = sum(scores) / len(scores)
        
        # Calculate overall bias score
        all_scores = [r["bias_score"] for r in results["detailed_results"]]
        results["overall_bias_score"] = sum(all_scores) / len(all_scores)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        logger.info(f"Completed CrowS-Pairs test. Found {results['issues_found']} issues.")
        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            results: Dictionary containing test results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Add overall bias recommendation if needed
        if abs(results["overall_bias_score"]) > self.bias_threshold:
            recommendations.append(
                f"Overall bias score of {results['overall_bias_score']:.3f} exceeds threshold "
                f"of {self.bias_threshold}. Consider debiasing techniques."
            )
        
        # Add category-specific recommendations
        for category, score in results["category_scores"].items():
            if abs(score) > self.bias_threshold:
                recommendations.append(
                    f"High bias detected in category '{category}' with score {score:.3f}. "
                    f"Review and address {category}-related biases."
                )
        
        # Add general recommendations if issues found
        if results["issues_found"] > 0:
            recommendations.append(
                f"Found {results['issues_found']} instances of significant bias. "
                "Consider reviewing training data and model architecture."
            )
            
        return recommendations

    def _get_current_time(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() 