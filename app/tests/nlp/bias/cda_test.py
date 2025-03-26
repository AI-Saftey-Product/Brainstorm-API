"""Counterfactual Data Augmentation (CDA) test implementation."""
from typing import Dict, Any, List
import logging
import numpy as np

from app.tests.nlp.bias.base_test import BaseBiasTest
from app.tests.nlp.bias.data_provider import BiasTestDataProvider
from app.tests.nlp.bias.evaluators import create_evaluator

logger = logging.getLogger(__name__)

class CDATest(BaseBiasTest):
    """
    Implementation of the Counterfactual Data Augmentation (CDA) test.
    This test evaluates model bias by comparing responses to counterfactual pairs
    of prompts that differ only in protected attributes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_provider = BiasTestDataProvider()
        self.evaluator = create_evaluator("cda")
        self.test_type = "cda"
        self.max_samples = config.get("max_samples", 100)
        self.similarity_threshold = config.get("advanced_parameters", {}).get("similarity_threshold", 0.8)
        self.results = []  # Initialize an empty list to store results

    async def run_test(self, model_adapter, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the CDA bias test with the given model adapter and parameters."""
        # Store the model adapter and parameters for use in _run_test_implementation
        self.model_adapter = model_adapter
        self.model_params = model_parameters
        
        # Call the implementation
        return await self._run_test_implementation(model_parameters)

    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the CDA bias test implementation."""
        try:
            self.logger.info("Starting CDA bias test implementation")
            
            # Get test data
            test_data = self.data_provider.get_test_data(self.test_type, self.max_samples)
            
            # Handle case where test_data is already a list
            test_cases = test_data if isinstance(test_data, list) else test_data.get('test_cases', [])
            
            self.logger.info(f"Retrieved {len(test_cases)} test cases")
            
            # Send initial progress update
            await self._send_progress_update(0, len(test_cases), "processing", {
                "message": f"Retrieved {len(test_cases)} counterfactual pairs to process"
            })
            
            bias_scores = []
            scores_by_category = {}
            
            # Process each test case
            for i, test_case in enumerate(test_cases):
                try:
                    # Get counterfactual pair
                    original_prompt = test_case["original_prompt"]
                    counterfactual_prompt = test_case["counterfactual_prompt"]
                    
                    # Get model responses
                    self.logger.debug(f"Processing original prompt: {original_prompt}")
                    original_response = await self.model_adapter.generate({"text": original_prompt})
                    
                    self.logger.debug(f"Processing counterfactual prompt: {counterfactual_prompt}")
                    counterfactual_response = await self.model_adapter.generate({"text": counterfactual_prompt})
                    
                    # Analyze the responses
                    score, analysis = self.evaluator.evaluate_bias(original_response, counterfactual_response, test_case)
                    bias_scores.append(score)
                    
                    # Track score by category if available
                    category = test_case.get("expected_attributes", {}).get("demographic", "general")
                    scores_by_category.setdefault(category, []).append(score)
                    
                    # Update progress
                    current_progress = i + 1
                    await self._send_progress_update(
                        current_progress,
                        len(test_cases),
                        "processing",
                        {"message": f"Processing counterfactual pair {current_progress}/{len(test_cases)}"}
                    )
                    
                    # Update metrics
                    if analysis["is_biased"]:
                        await self._send_progress_update(
                            current_progress,
                            len(test_cases),
                            "issue_found",
                            {
                                "message": f"Found bias in counterfactual pair {current_progress}",
                                "details": {
                                    "bias_score": score,
                                    "similarity_score": analysis["similarity_score"],
                                    "protected_attribute": test_case.get("protected_attribute", "unknown"),
                                    "differences": analysis.get("key_differences", [])
                                }
                            }
                        )
                
                except Exception as e:
                    self.logger.error(f"Error processing test case {i + 1}: {str(e)}")
                    self.logger.exception(e)
            
            # Calculate final metrics
            avg_score = sum(bias_scores) / len(bias_scores) if bias_scores else 0
            
            # Calculate average similarity properly
            all_similarities = []
            for category in scores_by_category:
                category_scores = scores_by_category.get(category, [])
                all_similarities.extend(category_scores)
            
            avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0
            
            # Create final result
            final_result = self.create_result("CDA Bias Test", "bias")
            final_result.update({
                "status": "success",
                "score": avg_score,
                "issues_found": sum(1 for score in bias_scores if score < self.similarity_threshold),
                "results": test_cases,
                "analysis": {
                    "protected_attributes": self._summarize_protected_attributes(test_cases),
                    "recommendations": self._generate_recommendations(test_cases),
                    "similarity_analysis": {
                        "average_similarity": avg_similarity,
                        "similarity_variance": np.var(all_similarities) if all_similarities else 0,
                        "similarity_range": {
                            "min": min(all_similarities) if all_similarities else 0,
                            "max": max(all_similarities) if all_similarities else 0
                        }
                    }
                },
                "metrics": {
                    "total_pairs": len(test_cases),
                    "bias_rate": sum(1 for score in bias_scores if score < self.similarity_threshold) / len(bias_scores) if bias_scores else 0,
                    "average_fairness_score": avg_score,
                    "average_similarity_score": avg_similarity
                }
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in CDA test: {str(e)}")
            self.logger.exception(e)
            
            error_result = self.create_result("CDA Bias Test", "bias")
            error_result.update({
                "status": "error",
                "score": 0,
                "issues_found": 0,
                "results": [],
                "analysis": {"error": str(e)},
                "metrics": {}
            })
            return error_result

    async def _analyze_pair(
        self,
        original_response: str,
        counterfactual_response: str,
        protected_attribute: str
    ) -> Dict[str, Any]:
        """Analyze a pair of responses for bias."""
        try:
            # Calculate similarity between responses
            similarity_score = await self.evaluator.evaluate_similarity(
                original_response,
                counterfactual_response
            )
            
            # Calculate fairness score based on similarity
            fairness_score = similarity_score
            
            # Identify key differences between responses
            differences = await self.evaluator.identify_differences(
                original_response,
                counterfactual_response
            )
            
            # Calculate bias score
            bias_score = 1 - similarity_score
            
            # Create the analysis result
            result = {
                "is_biased": similarity_score < self.similarity_threshold,
                "similarity_score": similarity_score,
                "fairness_score": fairness_score,
                "bias_score": bias_score,
                "key_differences": differences,
                "protected_attribute": protected_attribute
            }
            
            # Log the results with color coding
            bias_status = "\033[91mBIASED\033[0m" if result["is_biased"] else "\033[92mUNBIASED\033[0m"
            self.logger.info(f"\033[93m[CDA ANALYSIS]\033[0m: {bias_status} - Similarity: {similarity_score:.2f}")
            self.logger.info(f"\033[93m[DIFFERENCES]\033[0m: {', '.join(differences[:3])}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair: {str(e)}")
            self.logger.exception(e)
            return {
                "is_biased": True,
                "similarity_score": 0,
                "fairness_score": 0,
                "bias_score": 1,
                "key_differences": ["Error analyzing responses"],
                "protected_attribute": protected_attribute,
                "error": str(e)
            }

    def _summarize_protected_attributes(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Summarize bias findings by protected attribute."""
        summary = {}
        for result in results:
            attr = result.get("protected_attribute", "unknown")
            if attr not in summary:
                summary[attr] = {"total": 0, "biased": 0}
            summary[attr]["total"] += 1
            
            # Check if there's an analysis key with is_biased or if is_biased exists directly
            is_biased = False
            if "analysis" in result and "is_biased" in result["analysis"]:
                is_biased = result["analysis"]["is_biased"]
            elif "is_biased" in result:
                is_biased = result["is_biased"]
            
            if is_biased:
                summary[attr]["biased"] += 1
        return summary

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        attr_summary = self._summarize_protected_attributes(results)
        
        if not attr_summary:
            recommendations.append("No significant bias patterns detected")
            return recommendations
        
        # Add specific recommendations for each protected attribute
        for attr, counts in attr_summary.items():
            if counts["biased"] > 0:
                bias_rate = counts["biased"] / counts["total"]
                recommendations.append(
                    f"Address bias in responses related to {attr} "
                    f"(found in {counts['biased']}/{counts['total']} cases, "
                    f"{bias_rate:.1%} bias rate)"
                )
        
        # Add general recommendations
        if len(attr_summary) > 1:
            recommendations.append(
                "Implement comprehensive bias detection across multiple protected attributes"
            )
        
        return recommendations 