"""Intersectional bias test implementation."""
from typing import Dict, Any, List
import logging
import numpy as np

from app.tests.nlp.bias.base_test import BaseBiasTest
from app.tests.nlp.bias.data_provider import BiasTestDataProvider
from app.tests.nlp.bias.evaluators import create_evaluator

logger = logging.getLogger(__name__)

class IntersectionalBiasTest(BaseBiasTest):
    """
    Implementation of the Intersectional Bias Test.
    This test evaluates model bias across multiple intersecting demographic attributes
    (e.g., gender + race, age + occupation, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_provider = BiasTestDataProvider()
        self.evaluator = create_evaluator("intersectional")
        self.test_type = "intersectional"
        self.max_samples = config.get("max_samples", 100)
        self.bias_threshold = config.get("advanced_parameters", {}).get("bias_threshold", 0.7)

    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the intersectional bias test implementation."""
        try:
            self.logger.info("Starting intersectional bias test implementation")
            
            # Get test data
            test_data = self.data_provider.get_test_data(self.test_type, self.max_samples)
            self.logger.info(f"Retrieved {len(test_data)} test cases")
            
            await self._send_progress_update(0, len(test_data), "processing", {
                "message": f"Retrieved {len(test_data)} intersectional test cases to process"
            })
            
            results = []
            issues_found = 0
            total_score = 0
            fairness_scores = []
            
            # Process each test case
            for i, test_case in enumerate(test_data):
                current_progress = i + 1
                await self._send_progress_update(
                    current_progress,
                    len(test_data),
                    "processing",
                    {"message": f"Processing test case {current_progress}/{len(test_data)}"}
                )
                
                prompt = test_case["prompt"]
                intersectional_attributes = test_case.get("intersectional_attributes", {})
                expected_fairness = test_case.get("expected_fairness", {})
                
                # Generate response
                response = await self.generate_with_retry(
                    prompt,
                    **model_parameters
                )
                
                if not response:
                    self.logger.warning(f"No response generated for prompt: {prompt}")
                    continue
                
                # Analyze response for intersectional bias
                analysis_result = await self._analyze_response(
                    response,
                    intersectional_attributes,
                    expected_fairness
                )
                
                # Update metrics
                if analysis_result["is_biased"]:
                    issues_found += 1
                    await self._send_progress_update(
                        current_progress,
                        len(test_data),
                        "issue_found",
                        {
                            "message": f"Found intersectional bias in test case {current_progress}",
                            "details": {
                                "bias_score": analysis_result["bias_score"],
                                "fairness_score": analysis_result["fairness_score"],
                                "intersectional_attributes": intersectional_attributes,
                                "bias_indicators": analysis_result.get("bias_indicators", [])
                            }
                        }
                    )
                
                total_score += analysis_result["fairness_score"]
                fairness_scores.append(analysis_result["fairness_score"])
                
                results.append({
                    "id": str(current_progress),
                    "prompt": prompt,
                    "response": response,
                    "intersectional_attributes": intersectional_attributes,
                    "expected_fairness": expected_fairness,
                    "analysis": analysis_result
                })
            
            # Calculate final metrics
            avg_score = total_score / len(results) if results else 0
            avg_fairness = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0
            
            # Create final result
            final_result = self.create_result("Intersectional Bias Test", "bias")
            final_result.update({
                "status": "success",
                "score": avg_score,
                "issues_found": issues_found,
                "results": results,
                "analysis": {
                    "intersectional_patterns": self._summarize_intersectional_patterns(results),
                    "recommendations": self._generate_recommendations(results),
                    "fairness_analysis": {
                        "average_fairness": avg_fairness,
                        "fairness_variance": np.var(fairness_scores) if fairness_scores else 0,
                        "fairness_range": {
                            "min": min(fairness_scores) if fairness_scores else 0,
                            "max": max(fairness_scores) if fairness_scores else 0
                        }
                    }
                },
                "metrics": {
                    "total_examples": len(results),
                    "bias_rate": issues_found / len(results) if results else 0,
                    "average_fairness_score": avg_fairness
                }
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in intersectional bias test: {str(e)}")
            self.logger.exception(e)
            
            error_result = self.create_result("Intersectional Bias Test", "bias")
            error_result.update({
                "status": "error",
                "score": 0,
                "issues_found": 0,
                "results": [],
                "analysis": {"error": str(e)},
                "metrics": {}
            })
            return error_result

    async def _analyze_response(
        self,
        response: str,
        intersectional_attributes: Dict[str, str],
        expected_fairness: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze a response for intersectional bias."""
        try:
            # Evaluate response fairness
            fairness_result = await self.evaluator.evaluate_fairness(
                response,
                intersectional_attributes
            )
            
            # Calculate overall fairness score
            fairness_score = fairness_result["fairness_score"]
            
            # Identify bias indicators
            bias_indicators = await self.evaluator.identify_bias_indicators(
                response,
                intersectional_attributes
            )
            
            # Calculate bias score
            bias_score = 1 - fairness_score
            
            return {
                "is_biased": fairness_score < self.bias_threshold,
                "fairness_score": fairness_score,
                "bias_score": bias_score,
                "bias_indicators": bias_indicators,
                "attribute_scores": fairness_result.get("attribute_scores", {}),
                "intersectional_attributes": intersectional_attributes
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing response: {str(e)}")
            return {
                "is_biased": True,
                "fairness_score": 0,
                "bias_score": 1,
                "bias_indicators": ["Error analyzing response"],
                "attribute_scores": {},
                "intersectional_attributes": intersectional_attributes,
                "error": str(e)
            }

    def _summarize_intersectional_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Summarize bias patterns across intersectional attributes."""
        patterns = {}
        for result in results:
            attrs = result.get("intersectional_attributes", {})
            attr_key = "+".join(sorted(attrs.values()))
            if attr_key not in patterns:
                patterns[attr_key] = {"total": 0, "biased": 0}
            patterns[attr_key]["total"] += 1
            if result["analysis"]["is_biased"]:
                patterns[attr_key]["biased"] += 1
        return patterns

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        patterns = self._summarize_intersectional_patterns(results)
        
        if not patterns:
            recommendations.append("No significant intersectional bias patterns detected")
            return recommendations
        
        # Add specific recommendations for each intersectional pattern
        for pattern, counts in patterns.items():
            if counts["biased"] > 0:
                bias_rate = counts["biased"] / counts["total"]
                recommendations.append(
                    f"Address intersectional bias for {pattern} "
                    f"(found in {counts['biased']}/{counts['total']} cases, "
                    f"{bias_rate:.1%} bias rate)"
                )
        
        # Add general recommendations
        if len(patterns) > 1:
            recommendations.append(
                "Implement comprehensive bias detection across multiple intersecting attributes"
            )
        
        return recommendations 