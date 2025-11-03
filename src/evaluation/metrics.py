"""
Evaluation metrics for multi-agent system performance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import numpy as np
from collections import defaultdict
import json


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Task performance metrics
    accuracy: float
    exact_match: float
    f1_score: Optional[float] = None
    pass_at_k: Optional[Dict[str, float]] = None
    
    # Process metrics
    deliberation_cost: Dict[str, Any] = None  # tokens, wall-clock time
    consensus_difficulty: Dict[str, Any] = None  # rounds, disagreement rate
    judge_robustness: Dict[str, Any] = None  # aggregation method performance
    
    # Diversity/variance metrics
    output_disagreement: Dict[str, float] = None
    distributional_distance: Dict[str, float] = None
    source_heterogeneity: float = None
    
    # Metadata
    task_id: str = None
    team_id: str = None
    condition_id: str = None
    timestamp: float = None
    duration: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "pass_at_k": self.pass_at_k,
            "deliberation_cost": self.deliberation_cost,
            "consensus_difficulty": self.consensus_difficulty,
            "judge_robustness": self.judge_robustness,
            "output_disagreement": self.output_disagreement,
            "distributional_distance": self.distributional_distance,
            "source_heterogeneity": self.source_heterogeneity,
            "task_id": self.task_id,
            "team_id": self.team_id,
            "condition_id": self.condition_id,
            "timestamp": self.timestamp,
            "duration": self.duration
        }


class MetricsCalculator(ABC):
    """Abstract base class for metrics calculation."""
    
    @abstractmethod
    def calculate_metrics(
        self, 
        team_response: Dict[str, Any], 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Calculate evaluation metrics."""
        pass


class TaskPerformanceMetrics(MetricsCalculator):
    """Calculate task-specific performance metrics."""
    
    def calculate_metrics(
        self, 
        team_response: Dict[str, Any], 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Calculate task performance metrics."""
        # Extract final answer
        final_answer = self._extract_final_answer(team_response)
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy(final_answer, ground_truth, task_metadata)
        exact_match = self._calculate_exact_match(final_answer, ground_truth)
        f1_score = self._calculate_f1_score(final_answer, ground_truth)
        pass_at_k = self._calculate_pass_at_k(final_answer, ground_truth, task_metadata)
        
        return EvaluationResult(
            accuracy=accuracy,
            exact_match=exact_match,
            f1_score=f1_score,
            pass_at_k=pass_at_k,
            task_id=task_metadata.get("task_id") if task_metadata else None
        )
    
    def _extract_final_answer(self, team_response: Dict[str, Any]) -> str:
        """Extract final answer from team response."""
        # TODO: Implement answer extraction logic
        return team_response.get("final_answer", "")
    
    def _calculate_accuracy(
        self, 
        answer: str, 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate accuracy based on task type."""
        task_type = task_metadata.get("task_type") if task_metadata else None
        
        if task_type == "math_reasoning":
            return self._math_accuracy(answer, ground_truth)
        elif task_type == "multi_hop_qa":
            return self._qa_accuracy(answer, ground_truth)
        elif task_type == "code_generation":
            return self._code_accuracy(answer, ground_truth)
        else:
            return self._general_accuracy(answer, ground_truth)
    
    def _math_accuracy(self, answer: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate accuracy for math reasoning tasks."""
        try:
            answer_num = float(answer)
            if isinstance(ground_truth, list):
                return 1.0 if any(abs(answer_num - float(gt)) < 1e-6 for gt in ground_truth) else 0.0
            else:
                return 1.0 if abs(answer_num - float(ground_truth)) < 1e-6 else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _qa_accuracy(self, answer: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate accuracy for QA tasks."""
        answer_clean = answer.strip().lower()
        if isinstance(ground_truth, list):
            return 1.0 if any(answer_clean == gt.strip().lower() for gt in ground_truth) else 0.0
        else:
            return 1.0 if answer_clean == ground_truth.strip().lower() else 0.0
    
    def _code_accuracy(self, answer: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate accuracy for code generation tasks."""
        # TODO: Implement code evaluation logic
        return 1.0 if answer == ground_truth else 0.0
    
    def _general_accuracy(self, answer: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate general accuracy."""
        return 1.0 if answer == ground_truth else 0.0
    
    def _calculate_exact_match(self, answer: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate exact match score."""
        if isinstance(ground_truth, list):
            return 1.0 if answer in ground_truth else 0.0
        return 1.0 if answer == ground_truth else 0.0
    
    def _calculate_f1_score(self, answer: str, ground_truth: Union[str, List[str]]) -> Optional[float]:
        """Calculate F1 score for text-based tasks."""
        # TODO: Implement F1 calculation
        return None
    
    def _calculate_pass_at_k(
        self, 
        answer: str, 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, float]]:
        """Calculate pass@k for code generation tasks."""
        if task_metadata and task_metadata.get("task_type") == "code_generation":
            # TODO: Implement pass@k calculation
            return {"pass_at_1": 0.0, "pass_at_10": 0.0, "pass_at_100": 0.0}
        return None


class ProcessMetrics(MetricsCalculator):
    """Calculate process-related metrics."""
    
    def calculate_metrics(
        self, 
        team_response: Dict[str, Any], 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Calculate process metrics."""
        deliberation_cost = self._calculate_deliberation_cost(team_response)
        consensus_difficulty = self._calculate_consensus_difficulty(team_response)
        judge_robustness = self._calculate_judge_robustness(team_response)
        
        return EvaluationResult(
            accuracy=0.0,  # Not calculated by this metric
            exact_match=0.0,
            deliberation_cost=deliberation_cost,
            consensus_difficulty=consensus_difficulty,
            judge_robustness=judge_robustness
        )
    
    def _calculate_deliberation_cost(self, team_response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deliberation cost metrics."""
        # Extract cost information from team response
        total_tokens = team_response.get("total_tokens", 0)
        wall_clock_time = team_response.get("duration", 0.0)
        num_rounds = team_response.get("num_rounds", 1)
        
        return {
            "total_tokens": total_tokens,
            "wall_clock_time": wall_clock_time,
            "tokens_per_round": total_tokens / max(num_rounds, 1),
            "time_per_round": wall_clock_time / max(num_rounds, 1),
            "cost_efficiency": total_tokens / max(wall_clock_time, 0.001)
        }
    
    def _calculate_consensus_difficulty(self, team_response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus difficulty metrics."""
        # Extract consensus information
        num_rounds = team_response.get("num_rounds", 1)
        disagreement_rate = team_response.get("disagreement_rate", 0.0)
        consensus_achieved = team_response.get("consensus_achieved", False)
        
        return {
            "num_rounds": num_rounds,
            "disagreement_rate": disagreement_rate,
            "consensus_achieved": consensus_achieved,
            "convergence_speed": 1.0 / max(num_rounds, 1),
            "stability": 1.0 - disagreement_rate
        }
    
    def _calculate_judge_robustness(self, team_response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate judge robustness metrics."""
        # Extract judge performance information
        aggregation_method = team_response.get("aggregation_method", "unknown")
        confidence_scores = team_response.get("confidence_scores", [])
        final_confidence = team_response.get("final_confidence", 0.0)
        
        return {
            "aggregation_method": aggregation_method,
            "confidence_variance": np.var(confidence_scores) if confidence_scores else 0.0,
            "final_confidence": final_confidence,
            "robustness_score": min(final_confidence, 1.0 - np.var(confidence_scores)) if confidence_scores else final_confidence
        }


class DiversityMetrics(MetricsCalculator):
    """Calculate diversity and variance metrics."""
    
    def calculate_metrics(
        self, 
        team_response: Dict[str, Any], 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Calculate diversity metrics."""
        output_disagreement = self._calculate_output_disagreement(team_response)
        distributional_distance = self._calculate_distributional_distance(team_response)
        source_heterogeneity = self._calculate_source_heterogeneity(team_response)
        
        return EvaluationResult(
            accuracy=0.0,  # Not calculated by this metric
            exact_match=0.0,
            output_disagreement=output_disagreement,
            distributional_distance=distributional_distance,
            source_heterogeneity=source_heterogeneity
        )
    
    def _calculate_output_disagreement(self, team_response: Dict[str, Any]) -> Dict[str, float]:
        """Calculate output disagreement metrics."""
        individual_responses = team_response.get("individual_responses", {})
        
        if len(individual_responses) < 2:
            return {"pairwise_disagreement": 0.0, "consensus_score": 1.0}
        
        # Calculate pairwise disagreement
        responses = list(individual_responses.values())
        disagreements = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                disagreement = self._calculate_response_disagreement(responses[i], responses[j])
                disagreements.append(disagreement)
        
        pairwise_disagreement = np.mean(disagreements) if disagreements else 0.0
        consensus_score = 1.0 - pairwise_disagreement
        
        return {
            "pairwise_disagreement": pairwise_disagreement,
            "consensus_score": consensus_score,
            "max_disagreement": max(disagreements) if disagreements else 0.0,
            "min_disagreement": min(disagreements) if disagreements else 0.0
        }
    
    def _calculate_response_disagreement(self, response1: Dict[str, Any], response2: Dict[str, Any]) -> float:
        """Calculate disagreement between two responses."""
        # Extract text responses
        text1 = response1.get("text", "")
        text2 = response2.get("text", "")
        
        # Simple word-level disagreement
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    def _calculate_distributional_distance(self, team_response: Dict[str, Any]) -> Dict[str, float]:
        """Calculate distributional distance metrics."""
        # TODO: Implement KL divergence and embedding dispersion
        individual_responses = team_response.get("individual_responses", {})
        
        if len(individual_responses) < 2:
            return {"kl_divergence": 0.0, "embedding_dispersion": 0.0}
        
        # Placeholder implementation
        return {
            "kl_divergence": 0.5,  # Placeholder
            "embedding_dispersion": 0.3,  # Placeholder
            "response_variance": 0.4  # Placeholder
        }
    
    def _calculate_source_heterogeneity(self, team_response: Dict[str, Any]) -> float:
        """Calculate source heterogeneity index."""
        individual_responses = team_response.get("individual_responses", {})
        
        if not individual_responses:
            return 0.0
        
        # Extract model information
        models = [resp.get("model", "unknown") for resp in individual_responses.values()]
        unique_models = set(models)
        
        # Calculate heterogeneity based on model diversity
        heterogeneity = len(unique_models) / len(models) if models else 0.0
        
        return heterogeneity


class ComprehensiveMetricsCalculator:
    """Calculate all metrics comprehensively."""
    
    def __init__(self):
        self.task_metrics = TaskPerformanceMetrics()
        self.process_metrics = ProcessMetrics()
        self.diversity_metrics = DiversityMetrics()
    
    def calculate_all_metrics(
        self, 
        team_response: Dict[str, Any], 
        ground_truth: Union[str, List[str]], 
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Calculate all evaluation metrics."""
        # Calculate each type of metrics
        task_result = self.task_metrics.calculate_metrics(team_response, ground_truth, task_metadata)
        process_result = self.process_metrics.calculate_metrics(team_response, ground_truth, task_metadata)
        diversity_result = self.diversity_metrics.calculate_metrics(team_response, ground_truth, task_metadata)
        
        # Combine results
        combined_result = EvaluationResult(
            accuracy=task_result.accuracy,
            exact_match=task_result.exact_match,
            f1_score=task_result.f1_score,
            pass_at_k=task_result.pass_at_k,
            deliberation_cost=process_result.deliberation_cost,
            consensus_difficulty=process_result.consensus_difficulty,
            judge_robustness=process_result.judge_robustness,
            output_disagreement=diversity_result.output_disagreement,
            distributional_distance=diversity_result.distributional_distance,
            source_heterogeneity=diversity_result.source_heterogeneity,
            task_id=task_metadata.get("task_id") if task_metadata else None,
            team_id=team_response.get("team_id"),
            condition_id=team_response.get("condition_id"),
            timestamp=time.time(),
            duration=team_response.get("duration", 0.0)
        )
        
        return combined_result

