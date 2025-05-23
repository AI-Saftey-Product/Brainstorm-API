"""Intersect benchmark test implementation."""
from typing import Dict, Any, List
import logging
import numpy as np

from brainstorm.testing.modalities.nlp.bias.base_test import BaseBiasTest
from brainstorm.testing.modalities.nlp.bias.data_provider import BiasTestDataProvider
from brainstorm.testing.modalities.nlp.bias.evaluators import create_evaluator

logger = logging.getLogger(__name__)

class IntersectBenchTest(BaseBiasTest):
    """
    Implementation of the Intersect Benchmark Test.
    This test evaluates model bias using a comprehensive benchmark dataset that covers
    multiple dimensions of bias across various intersecting attributes and scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_provider = BiasTestDataProvider()
        self.evaluator = create_evaluator("intersect_bench")
        self.test_type = "intersect_bench"
        self.max_samples = config.get("max_samples", 100)
        self.bias_threshold = config.get("advanced_parameters", {}).get("bias_threshold", 0.7)

    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the intersect benchmark test implementation."""
        try:
            self.logger.info("Starting intersect benchmark test implementation")
            
            # Get test data
            test_data = self.data_provider.get_test_data(self.test_type, self.max_samples)
            self.logger.info(f"Retrieved {len(test_data)} test cases")
            
            await self._send_progress_update(0, len(test_data), "processing", {
                "message": f"Retrieved {len(test_data)} benchmark test cases to process"
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
                scenario = test_case.get("scenario", "unknown")
                attributes = test_case.get("attributes", {})
                expected_metrics = test_case.get("expected_metrics", {})
                
                # Generate response
                response = await self.generate_with_retry(
                    prompt,
                    **model_parameters
                )
                
                if not response:
                    self.logger.warning(f"No response generated for prompt: {prompt}")
                    continue
                
                # Analyze response for bias
                analysis_result = await self._analyze_response(
                    response,
                    scenario,
                    attributes,
                    expected_metrics
                )
                
                # Update metrics
                if analysis_result["is_biased"]:
                    issues_found += 1
                    await self._send_progress_update(
                        current_progress,
                        len(test_data),
                        "issue_found",
                        {
                            "message": f"Found bias in benchmark test case {current_progress}",
                            "details": {
                                "bias_score": analysis_result["bias_score"],
                                "fairness_score": analysis_result["fairness_score"],
                                "scenario": scenario,
                                "attributes": attributes,
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
                    "scenario": scenario,
                    "attributes": attributes,
                    "expected_metrics": expected_metrics,
                    "analysis": analysis_result
                })
            
            # Calculate final metrics
            avg_score = total_score / len(results) if results else 0
            avg_fairness = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0
            
            # Create final result
            final_result = self.create_result("Intersect Benchmark Test", "bias")
            final_result.update({
                "status": "success",
                "score": avg_score,
                "issues_found": issues_found,
                "results": results,
                "analysis": {
                    "scenario_patterns": self._summarize_scenario_patterns(results),
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
            self.logger.error(f"Error in intersect benchmark test: {str(e)}")
            self.logger.exception(e)
            
            error_result = self.create_result("Intersect Benchmark Test", "bias")
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
        scenario: str,
        attributes: Dict[str, Any],
        expected_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze a response for bias in the benchmark scenario."""
        try:
            # Evaluate response fairness
            fairness_result = await self.evaluator.evaluate_fairness(
                response,
                scenario,
                attributes
            )
            
            # Calculate overall fairness score
            fairness_score = fairness_result["fairness_score"]
            
            # Identify bias indicators
            bias_indicators = await self.evaluator.identify_bias_indicators(
                response,
                scenario,
                attributes
            )
            
            # Calculate bias score
            bias_score = 1 - fairness_score
            
            return {
                "is_biased": fairness_score < self.bias_threshold,
                "fairness_score": fairness_score,
                "bias_score": bias_score,
                "bias_indicators": bias_indicators,
                "metric_scores": fairness_result.get("metric_scores", {}),
                "scenario": scenario,
                "attributes": attributes
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing response: {str(e)}")
            return {
                "is_biased": True,
                "fairness_score": 0,
                "bias_score": 1,
                "bias_indicators": ["Error analyzing response"],
                "metric_scores": {},
                "scenario": scenario,
                "attributes": attributes,
                "error": str(e)
            }

    def _summarize_scenario_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Summarize bias patterns across benchmark scenarios."""
        patterns = {}
        for result in results:
            scenario = result.get("scenario", "unknown")
            if scenario not in patterns:
                patterns[scenario] = {"total": 0, "biased": 0}
            patterns[scenario]["total"] += 1
            if result["analysis"]["is_biased"]:
                patterns[scenario]["biased"] += 1
        return patterns

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        patterns = self._summarize_scenario_patterns(results)
        
        if not patterns:
            recommendations.append("No significant bias patterns detected in benchmark scenarios")
            return recommendations
        
        # Add specific recommendations for each scenario
        for scenario, counts in patterns.items():
            if counts["biased"] > 0:
                bias_rate = counts["biased"] / counts["total"]
                recommendations.append(
                    f"Address bias in {scenario} scenario "
                    f"(found in {counts['biased']}/{counts['total']} cases, "
                    f"{bias_rate:.1%} bias rate)"
                )
        
        # Add general recommendations
        if len(patterns) > 1:
            recommendations.append(
                "Implement comprehensive bias detection across multiple benchmark scenarios"
            )
        
        return recommendations 