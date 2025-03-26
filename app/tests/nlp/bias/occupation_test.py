"""Occupational Bias Test implementation."""
from typing import Dict, Any, List
import logging

from app.tests.nlp.bias.base_test import BaseBiasTest
from app.tests.nlp.bias.data_provider import BiasTestDataProvider
from app.tests.nlp.bias.evaluators import create_evaluator

logger = logging.getLogger(__name__)

class OccupationalBiasTest(BaseBiasTest):
    """
    Implementation of the Occupational Bias Test.
    This test evaluates model bias in responses related to different occupations,
    checking for gender, racial, and other demographic stereotypes in professional contexts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_provider = BiasTestDataProvider()
        self.evaluator = create_evaluator("occupation")
        self.test_type = "occupation"
        self.max_samples = config.get("max_samples", 100)

    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the occupational bias test implementation.
        """
        try:
            self.logger.info("Starting occupational bias test implementation")
            
            # Get test data with occupation-related prompts
            test_data = self.data_provider.get_test_data(self.test_type, self.max_samples)
            self.logger.info(f"Retrieved {len(test_data)} test cases")
            
            results = []
            issues_found = 0
            total_score = 0
            
            # Process each test case
            for i, test_case in enumerate(test_data):
                self.logger.info(f"\nProcessing test case {i+1}/{len(test_data)}")
                
                prompt = test_case.get("prompt")
                occupation = test_case.get("occupation")
                demographic_terms = test_case.get("demographic_terms", [])
                bias_indicators = test_case.get("bias_indicators", [])
                
                if not prompt or not occupation:
                    self.logger.warning("Missing prompt or occupation in test case, skipping")
                    continue
                
                self.logger.info(f"Prompt: {prompt}")
                self.logger.info(f"Testing occupation: {occupation}")
                
                # Generate response
                response = await self.generate_with_retry(
                    prompt,
                    **model_parameters
                )
                
                if not response:
                    self.logger.warning("Failed to generate response")
                    continue
                
                self.logger.info(f"Response: {response}")
                
                # Analyze response for occupational bias
                analysis_result = self.evaluator.evaluate({
                    "response": response,
                    "occupation": occupation,
                    "demographic_terms": demographic_terms,
                    "bias_indicators": bias_indicators
                })
                
                # Add context to the analysis result
                analysis_result.update({
                    "test_case": {
                        "prompt": prompt,
                        "occupation": occupation,
                        "demographic_terms": demographic_terms,
                        "bias_indicators": bias_indicators
                    }
                })
                
                self.logger.info("Analysis Results:")
                self.logger.info(f"- Bias Score: {analysis_result['bias_score']:.2f}")
                self.logger.info(f"- Demographic Associations: {analysis_result.get('demographic_associations', [])}")
                self.logger.info(f"- Bias Indicators Found: {analysis_result.get('bias_indicators_found', [])}")
                self.logger.info(f"- Is Biased: {analysis_result['is_biased']}")
                
                # Update metrics
                if analysis_result["is_biased"]:
                    issues_found += 1
                    self.logger.info(f"Occupational bias detected! Issues found: {issues_found}")
                
                total_score += (1.0 - analysis_result["bias_score"])
                results.append({
                    "prompt": prompt,
                    "occupation": occupation,
                    "response": response,
                    "analysis": analysis_result
                })
            
            # Calculate final metrics
            avg_score = total_score / len(results) if results else 0
            
            final_result = {
                "status": "success",
                "score": avg_score,
                "issues_found": issues_found,
                "results": results,
                "analysis": {
                    "bias_patterns": self._analyze_bias_patterns(results),
                    "recommendations": self._generate_recommendations(results)
                },
                "metrics": {
                    "total_examples": len(results),
                    "bias_rate": issues_found / len(results) if results else 0,
                    "average_fairness_score": avg_score,
                    "occupation_specific_metrics": self._calculate_occupation_metrics(results)
                }
            }
            
            self.logger.info("\nTest Summary:")
            self.logger.info(f"Total examples processed: {len(results)}")
            self.logger.info(f"Issues found: {issues_found}")
            self.logger.info(f"Average fairness score: {avg_score:.2f}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in occupational bias test: {str(e)}")
            self.logger.exception(e)
            return {
                "status": "error",
                "score": 0,
                "issues_found": 0,
                "results": [],
                "analysis": {"error": str(e)},
                "metrics": {}
            }

    def _analyze_bias_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in bias across different occupations."""
        bias_patterns = {
            "gender_bias": 0,
            "racial_bias": 0,
            "age_bias": 0,
            "other_bias": 0,
            "neutral": 0
        }
        
        occupation_bias = {}  # Track bias by occupation
        
        for result in results:
            analysis = result.get("analysis", {})
            occupation = result.get("occupation", "unknown")
            
            if analysis.get("is_biased"):
                # Categorize the type of bias
                associations = analysis.get("demographic_associations", [])
                if any("gender" in assoc.lower() for assoc in associations):
                    bias_patterns["gender_bias"] += 1
                elif any("race" in assoc.lower() or "ethnic" in assoc.lower() for assoc in associations):
                    bias_patterns["racial_bias"] += 1
                elif any("age" in assoc.lower() for assoc in associations):
                    bias_patterns["age_bias"] += 1
                else:
                    bias_patterns["other_bias"] += 1
                
                # Track by occupation
                if occupation not in occupation_bias:
                    occupation_bias[occupation] = {"biased": 0, "total": 0}
                occupation_bias[occupation]["biased"] += 1
            else:
                bias_patterns["neutral"] += 1
            
            # Update occupation totals
            if occupation not in occupation_bias:
                occupation_bias[occupation] = {"biased": 0, "total": 0}
            occupation_bias[occupation]["total"] += 1
        
        return {
            "bias_distribution": bias_patterns,
            "occupation_bias_rates": {
                occupation: {
                    "bias_rate": stats["biased"] / stats["total"] if stats["total"] > 0 else 0,
                    "total_cases": stats["total"]
                }
                for occupation, stats in occupation_bias.items()
            }
        }

    def _calculate_occupation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed metrics for each occupation."""
        occupation_metrics = {}
        
        for result in results:
            occupation = result.get("occupation")
            if not occupation:
                continue
                
            if occupation not in occupation_metrics:
                occupation_metrics[occupation] = {
                    "total_cases": 0,
                    "biased_cases": 0,
                    "avg_bias_score": 0,
                    "demographic_associations": {}
                }
            
            metrics = occupation_metrics[occupation]
            analysis = result.get("analysis", {})
            
            metrics["total_cases"] += 1
            if analysis.get("is_biased"):
                metrics["biased_cases"] += 1
            
            metrics["avg_bias_score"] += analysis.get("bias_score", 0)
            
            # Track demographic associations
            for assoc in analysis.get("demographic_associations", []):
                if assoc not in metrics["demographic_associations"]:
                    metrics["demographic_associations"][assoc] = 0
                metrics["demographic_associations"][assoc] += 1
        
        # Calculate averages
        for metrics in occupation_metrics.values():
            if metrics["total_cases"] > 0:
                metrics["avg_bias_score"] /= metrics["total_cases"]
                metrics["bias_rate"] = metrics["biased_cases"] / metrics["total_cases"]
        
        return occupation_metrics

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on the bias analysis results."""
        bias_patterns = self._analyze_bias_patterns(results)
        recommendations = []
        
        # Check for specific types of bias
        if bias_patterns["bias_distribution"]["gender_bias"] > 0:
            recommendations.append(
                "Address gender stereotypes in occupational descriptions and requirements"
            )
        
        if bias_patterns["bias_distribution"]["racial_bias"] > 0:
            recommendations.append(
                "Review and eliminate racial bias in professional context representations"
            )
        
        if bias_patterns["bias_distribution"]["age_bias"] > 0:
            recommendations.append(
                "Ensure age-neutral language in occupation-related responses"
            )
        
        # Check occupation-specific patterns
        occupation_rates = bias_patterns.get("occupation_bias_rates", {})
        high_bias_occupations = [
            occupation for occupation, stats in occupation_rates.items()
            if stats["bias_rate"] > 0.3  # Threshold for high bias rate
        ]
        
        if high_bias_occupations:
            recommendations.append(
                f"Focus bias mitigation efforts on high-bias occupations: {', '.join(high_bias_occupations)}"
            )
        
        # General recommendations
        if any(bias > 0 for bias in bias_patterns["bias_distribution"].values()):
            recommendations.append(
                "Implement comprehensive bias detection and mitigation for occupational contexts"
            )
            recommendations.append(
                "Develop training data that represents diverse demographics across all professions"
            )
        
        return recommendations
 