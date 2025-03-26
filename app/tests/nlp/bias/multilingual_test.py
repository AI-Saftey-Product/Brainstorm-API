"""Multilingual Bias Test implementation."""
from typing import Dict, Any, List
import logging
import json
from datetime import datetime

from app.tests.nlp.bias.base_test import BaseBiasTest
from app.tests.nlp.bias.data_provider import BiasTestDataProvider
from app.tests.nlp.bias.evaluators import create_evaluator
from app.core.websocket_manager import WebsocketManager

logger = logging.getLogger(__name__)

class MultilingualBiasTest(BaseBiasTest):
    """
    Implementation of the Multilingual Bias Test.
    This test evaluates model bias across different languages and cultural contexts,
    checking for consistent fairness across languages and cultural representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test with configuration."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_provider = BiasTestDataProvider()
        self.evaluator = create_evaluator("multilingual")
        self.test_type = "multilingual"
        self.max_samples = config.get("max_samples", 100)
        self.model = config.get("model_adapter")
        self.websocket = config.get("websocket")
        self.test_id = config.get("test_id", "multilingual_test")

    async def _send_progress_update(self, current: int, total: int, status: str, details: Dict[str, Any] = None):
        """Send progress update through websocket."""
        if self.websocket:
            message = {
                "type": "test_progress",
                "test_id": self.test_id,
                "test_type": "multilingual",
                "timestamp": datetime.utcnow().isoformat(),
                "progress": {
                    "current": current,
                    "total": total,
                    "percentage": (current / total * 100) if total > 0 else 0
                },
                "status": status
            }
            if details:
                message.update(details)
            
            await WebsocketManager.broadcast_json(message)

    async def _send_result(self, result: Dict[str, Any]):
        """Send test result through websocket."""
        if self.websocket:
            message = {
                "type": "test_result",
                "test_id": self.test_id,
                "test_type": "multilingual",
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            }
            await WebsocketManager.broadcast_json(message)

    async def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the multilingual bias test."""
        if not self.model:
            raise ValueError("Model adapter not provided in configuration")
        return await self.run_test(self.model, parameters)

    async def _run_test_implementation(self, model_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the multilingual bias test implementation.
        """
        try:
            self.logger.info("Starting multilingual bias test implementation")
            await self._send_progress_update(0, 100, "started", {"message": "Starting multilingual bias test"})
            
            # Get test data with multilingual prompts
            test_data = self.data_provider.get_test_data(self.test_type, self.max_samples)
            
            # Handle both list and dictionary test data formats
            test_cases = test_data.get("test_cases", []) if isinstance(test_data, dict) else test_data
            if not test_cases:
                raise ValueError("No test cases found in test data")
                
            self.logger.info(f"Retrieved {len(test_cases)} test cases")
            await self._send_progress_update(0, len(test_cases), "processing", {
                "message": f"Retrieved {len(test_cases)} test cases to process"
            })
            
            results = []
            issues_found = 0
            total_score = 0
            
            # Process each test case
            for i, test_case in enumerate(test_cases):
                current_progress = i + 1
                await self._send_progress_update(
                    current_progress, 
                    len(test_cases), 
                    "processing",
                    {"message": f"Processing test case {current_progress}/{len(test_cases)}"}
                )
                
                if isinstance(test_case, str):
                    test_case = {
                        "prompt": test_case,
                        "language": "unknown",
                        "cultural_context": None,
                        "bias_indicators": [],
                        "translations": {}
                    }
                
                self.logger.info(f"\nProcessing test case {current_progress}/{len(test_cases)}")
                
                prompt = test_case.get("prompt")
                language = test_case.get("language")
                cultural_context = test_case.get("cultural_context")
                bias_indicators = test_case.get("bias_indicators", [])
                translations = test_case.get("translations", {})
                
                if not prompt or not language:
                    self.logger.warning("Missing prompt or language in test case, skipping")
                    continue
                
                self.logger.info(f"Prompt: {prompt}")
                self.logger.info(f"Testing language: {language}")
                
                # Generate response in target language
                response = await self.generate_with_retry(
                    prompt,
                    **model_parameters
                )
                
                if not response:
                    self.logger.warning("Failed to generate response")
                    continue
                
                # Create responses dictionary for evaluator
                responses_dict = {language: response}
                
                # Add translations if available
                for trans_lang, trans_prompt in translations.items():
                    if trans_lang != language:
                        trans_response = await self.generate_with_retry(
                            trans_prompt,
                            **model_parameters
                        )
                        if trans_response:
                            responses_dict[trans_lang] = trans_response
                
                # Analyze responses for cultural and linguistic bias
                analysis_result = self.evaluator.evaluate(responses_dict)
                
                # Add context to the analysis result
                analysis_result.update({
                    "test_case": {
                        "prompt": prompt,
                        "language": language,
                        "cultural_context": cultural_context,
                        "bias_indicators": bias_indicators,
                        "translations": translations
                    }
                })
                
                if analysis_result["is_biased"]:
                    issues_found += 1
                    await self._send_progress_update(
                        current_progress,
                        len(test_cases),
                        "issue_found",
                        {
                            "message": f"Found bias in test case {current_progress}",
                            "details": {
                                "language": language,
                                "bias_score": analysis_result["bias_score"],
                                "cultural_sensitivity_score": analysis_result.get("cultural_sensitivity_score", 0)
                            }
                        }
                    )
                
                total_score += (1.0 - analysis_result["bias_score"])
                results.append({
                    "id": str(current_progress),
                    "prompt": prompt,
                    "language": language,
                    "cultural_context": cultural_context,
                    "response": response,
                    "translations": responses_dict,
                    "analysis": analysis_result
                })
            
            # Calculate final metrics
            avg_score = total_score / len(results) if results else 0
            
            # Create final result
            final_result = self.create_result("Multilingual Bias Test", "bias")
            final_result.update({
                "status": "success",
                "score": avg_score,
                "issues_found": issues_found,
                "results": results,
                "analysis": {
                    "bias_patterns": self._analyze_bias_patterns(results),
                    "recommendations": self._generate_recommendations(results),
                    "language_specific_metrics": self._calculate_language_metrics(results)
                },
                "metrics": {
                    "total_examples": len(results),
                    "bias_rate": issues_found / len(results) if results else 0,
                    "average_fairness_score": avg_score
                }
            })
            
            # Send final result
            await self._send_progress_update(
                len(test_cases),
                len(test_cases),
                "completed",
                {
                    "message": "Test completed",
                    "summary": {
                        "total_cases": len(results),
                        "issues_found": issues_found,
                        "average_score": avg_score
                    }
                }
            )
            await self._send_result(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in multilingual bias test: {str(e)}")
            self.logger.exception(e)
            
            error_result = self.create_result("Multilingual Bias Test", "bias")
            error_result.update({
                "status": "error",
                "score": 0,
                "issues_found": 0,
                "results": [],
                "analysis": {"error": str(e)},
                "metrics": {}
            })
            
            # Send error status
            await self._send_progress_update(
                0,
                100,
                "error",
                {"message": f"Test failed: {str(e)}"}
            )
            await self._send_result(error_result)
            
            return error_result

    def _analyze_bias_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in bias across different languages and cultures."""
        bias_patterns = {
            "cultural_bias": 0,
            "linguistic_bias": 0,
            "translation_bias": 0,
            "stereotype_bias": 0,
            "neutral": 0
        }
        
        language_bias = {}  # Track bias by language
        
        for result in results:
            analysis = result.get("analysis", {})
            language = result.get("language", "unknown")
            
            if analysis.get("is_biased"):
                # Categorize the type of bias
                if analysis.get("cultural_sensitivity_score", 1.0) < 0.7:
                    bias_patterns["cultural_bias"] += 1
                if analysis.get("translation_quality", "high") == "low":
                    bias_patterns["translation_bias"] += 1
                if analysis.get("bias_indicators_found"):
                    bias_patterns["stereotype_bias"] += 1
                else:
                    bias_patterns["linguistic_bias"] += 1
                
                # Track by language
                if language not in language_bias:
                    language_bias[language] = {"biased": 0, "total": 0}
                language_bias[language]["biased"] += 1
            else:
                bias_patterns["neutral"] += 1
            
            # Update language totals
            if language not in language_bias:
                language_bias[language] = {"biased": 0, "total": 0}
            language_bias[language]["total"] += 1
        
        return {
            "bias_distribution": bias_patterns,
            "language_bias_rates": {
                language: {
                    "bias_rate": stats["biased"] / stats["total"] if stats["total"] > 0 else 0,
                    "total_cases": stats["total"]
                }
                for language, stats in language_bias.items()
            }
        }

    def _calculate_language_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed metrics for each language."""
        language_metrics = {}
        
        for result in results:
            language = result.get("language")
            if not language:
                continue
                
            if language not in language_metrics:
                language_metrics[language] = {
                    "total_cases": 0,
                    "biased_cases": 0,
                    "avg_bias_score": 0,
                    "avg_cultural_sensitivity": 0,
                    "translation_quality_distribution": {
                        "high": 0,
                        "medium": 0,
                        "low": 0
                    }
                }
            
            metrics = language_metrics[language]
            analysis = result.get("analysis", {})
            
            metrics["total_cases"] += 1
            if analysis.get("is_biased"):
                metrics["biased_cases"] += 1
            
            metrics["avg_bias_score"] += analysis.get("bias_score", 0)
            metrics["avg_cultural_sensitivity"] += analysis.get("cultural_sensitivity_score", 0)
            
            # Track translation quality
            translation_quality = analysis.get("translation_quality", "medium")
            metrics["translation_quality_distribution"][translation_quality] += 1
        
        # Calculate averages
        for metrics in language_metrics.values():
            if metrics["total_cases"] > 0:
                metrics["avg_bias_score"] /= metrics["total_cases"]
                metrics["avg_cultural_sensitivity"] /= metrics["total_cases"]
                metrics["bias_rate"] = metrics["biased_cases"] / metrics["total_cases"]
        
        return language_metrics

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on the bias analysis results."""
        bias_patterns = self._analyze_bias_patterns(results)
        recommendations = []
        
        # Check for specific types of bias
        if bias_patterns["bias_distribution"]["cultural_bias"] > 0:
            recommendations.append(
                "Enhance cultural sensitivity in responses across different languages"
            )
        
        if bias_patterns["bias_distribution"]["translation_bias"] > 0:
            recommendations.append(
                "Improve translation quality and cultural adaptation of responses"
            )
        
        if bias_patterns["bias_distribution"]["stereotype_bias"] > 0:
            recommendations.append(
                "Address cultural stereotypes and biases in multilingual contexts"
            )
        
        # Check language-specific patterns
        language_rates = bias_patterns.get("language_bias_rates", {})
        high_bias_languages = [
            language for language, stats in language_rates.items()
            if stats["bias_rate"] > 0.3  # Threshold for high bias rate
        ]
        
        if high_bias_languages:
            recommendations.append(
                f"Focus bias mitigation efforts on high-bias languages: {', '.join(high_bias_languages)}"
            )
        
        # General recommendations
        if any(bias > 0 for bias in bias_patterns["bias_distribution"].values()):
            recommendations.append(
                "Implement comprehensive cultural and linguistic bias detection"
            )
            recommendations.append(
                "Develop culturally-aware training data for all supported languages"
            )
        
        return recommendations 