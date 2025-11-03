"""
Mixture of Agents (MoA) baseline implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import random
import numpy as np

from ..core.agent import Agent, AgentConfig, LLMAgent
from ..core.task import TaskItem
from .single_agent import BaselineConfig


@dataclass
class MoAConfig(BaselineConfig):
    """Configuration for Mixture of Agents baseline."""
    num_agents: int = 4
    model_families: List[str] = None
    aggregation_method: str = "weighted_average"  # weighted_average, majority, confidence_weighted
    temperature_diversity: float = 0.3  # Range of temperature variation
    
    def __post_init__(self):
        if self.model_families is None:
            self.model_families = ["gpt-3.5-turbo", "claude-3-sonnet", "llama-2-7b"]


class MixtureOfAgentsBaseline:
    """
    Mixture of Agents baseline implementation.
    
    Uses multiple diverse agents without explicit leadership structure.
    """
    
    def __init__(self, config: MoAConfig):
        self.config = config
        self.agents = self._create_agents()
        
        if config.seed is not None:
            random.seed(config.seed)
    
    def _create_agents(self) -> List[Agent]:
        """Create diverse agents for the mixture."""
        agents = []
        
        for i in range(self.config.num_agents):
            # Select model family
            model_family = random.choice(self.config.model_families)
            
            # Create agent configuration
            agent_config = AgentConfig(
                agent_id=f"moa_agent_{i}",
                agent_type="peer",
                model_name=model_family,
                temperature=self.config.temperature + random.uniform(
                    -self.config.temperature_diversity, 
                    self.config.temperature_diversity
                ),
                max_tokens=self.config.max_tokens,
                prompt_template=f"moa_template_{i}",
                seed=self.config.seed
            )
            
            # Ensure temperature is within valid range
            agent_config.temperature = max(0.1, min(1.0, agent_config.temperature))
            
            agents.append(LLMAgent(agent_config))
        
        return agents
    
    async def solve_task(self, task_item: TaskItem) -> Dict[str, Any]:
        """Solve task using mixture of agents."""
        start_time = time.time()
        
        # Generate responses from all agents
        agent_responses = []
        for i, agent in enumerate(self.agents):
            try:
                # Create task-specific prompt
                prompt = self._create_moa_prompt(task_item, i)
                
                # Generate response
                response = await agent.generate_response(prompt)
                agent_responses.append({
                    "agent_id": agent.config.agent_id,
                    "response": response,
                    "model": agent.config.model_name,
                    "temperature": agent.config.temperature
                })
            except Exception as e:
                print(f"Error with agent {i}: {e}")
                continue
        
        if not agent_responses:
            return {
                "baseline_type": "MixtureOfAgents",
                "task_id": task_item.item_id,
                "error": "No agents responded successfully"
            }
        
        # Aggregate responses
        final_answer = self._aggregate_responses(agent_responses)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(agent_responses)
        
        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(agent_responses)
        
        duration = time.time() - start_time
        
        return {
            "baseline_type": "MixtureOfAgents",
            "task_id": task_item.item_id,
            "question": task_item.question,
            "ground_truth": task_item.ground_truth,
            "final_answer": final_answer,
            "agent_responses": agent_responses,
            "diversity_metrics": diversity_metrics,
            "consensus_metrics": consensus_metrics,
            "num_agents": len(agent_responses),
            "aggregation_method": self.config.aggregation_method,
            "duration": duration,
            "timestamp": time.time()
        }
    
    def _create_moa_prompt(self, task_item: TaskItem, agent_index: int) -> str:
        """Create prompt for a specific agent in the mixture."""
        task_type = task_item.task_type.value
        
        # Add some variation to prompts
        prompt_variations = [
            f"Solve this {task_type} problem:",
            f"Please solve the following {task_type} question:",
            f"Answer this {task_type} problem step by step:",
            f"Work through this {task_type} task:"
        ]
        
        intro = prompt_variations[agent_index % len(prompt_variations)]
        
        if task_type == "math_reasoning":
            return f"""{intro}

Problem: {task_item.question}

Please show your work and provide a clear final answer.

Solution:
"""
        elif task_type == "multi_hop_qa":
            return f"""{intro}

Question: {task_item.question}

Please provide a clear and accurate answer.

Answer:
"""
        elif task_type == "code_generation":
            return f"""{intro}

Problem: {task_item.question}

Please provide working code with clear comments.

Code:
"""
        else:
            return f"""{intro}

Problem: {task_item.question}

Please provide a clear answer.

Answer:
"""
    
    def _aggregate_responses(self, agent_responses: List[Dict[str, Any]]) -> str:
        """Aggregate responses from multiple agents."""
        if not agent_responses:
            return ""
        
        if self.config.aggregation_method == "majority":
            return self._majority_voting(agent_responses)
        elif self.config.aggregation_method == "confidence_weighted":
            return self._confidence_weighted_aggregation(agent_responses)
        else:  # weighted_average
            return self._weighted_average_aggregation(agent_responses)
    
    def _majority_voting(self, agent_responses: List[Dict[str, Any]]) -> str:
        """Aggregate using majority voting."""
        # Extract answers
        answers = []
        for resp in agent_responses:
            answer = self._extract_final_answer(resp["response"].get("text", ""))
            answers.append(answer)
        
        # Count votes
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Return most common answer
        if answer_counts:
            return max(answer_counts, key=answer_counts.get)
        else:
            return answers[0] if answers else ""
    
    def _confidence_weighted_aggregation(self, agent_responses: List[Dict[str, Any]]) -> str:
        """Aggregate using confidence-weighted voting."""
        # Extract answers and confidences
        answer_confidence_pairs = []
        for resp in agent_responses:
            answer = self._extract_final_answer(resp["response"].get("text", ""))
            confidence = resp["response"].get("confidence", 0.5)
            answer_confidence_pairs.append((answer, confidence))
        
        # Weight by confidence
        answer_weights = {}
        for answer, confidence in answer_confidence_pairs:
            answer_weights[answer] = answer_weights.get(answer, 0) + confidence
        
        # Return highest weighted answer
        if answer_weights:
            return max(answer_weights, key=answer_weights.get)
        else:
            return answer_confidence_pairs[0][0] if answer_confidence_pairs else ""
    
    def _weighted_average_aggregation(self, agent_responses: List[Dict[str, Any]]) -> str:
        """Aggregate using weighted average (for numerical answers)."""
        # Try to extract numerical answers
        numerical_answers = []
        confidences = []
        
        for resp in agent_responses:
            answer = self._extract_final_answer(resp["response"].get("text", ""))
            confidence = resp["response"].get("confidence", 0.5)
            
            try:
                # Try to convert to number
                num_answer = float(answer)
                numerical_answers.append(num_answer)
                confidences.append(confidence)
            except ValueError:
                # If not numerical, fall back to majority voting
                continue
        
        if numerical_answers:
            # Weighted average
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # Normalize
            
            weighted_avg = np.sum(np.array(numerical_answers) * weights)
            return str(weighted_avg)
        else:
            # Fall back to majority voting
            return self._majority_voting(agent_responses)
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        lines = response.split('\n')
        
        # Look for final answer patterns
        for line in reversed(lines):
            line = line.strip()
            if any(phrase in line.lower() for phrase in ['answer:', 'result:', 'final:', 'solution:']):
                # Extract after the phrase
                for phrase in ['answer:', 'result:', 'final:', 'solution:']:
                    if phrase in line.lower():
                        answer = line.lower().split(phrase)[-1].strip()
                        if answer:
                            return answer
        
        # Return last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line
        
        return response.strip()
    
    def _calculate_diversity_metrics(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diversity metrics across agents."""
        if len(agent_responses) < 2:
            return {"diversity_score": 0.0, "model_diversity": 0.0}
        
        # Model diversity
        models = [resp["model"] for resp in agent_responses]
        unique_models = set(models)
        model_diversity = len(unique_models) / len(models)
        
        # Temperature diversity
        temperatures = [resp["response"].get("temperature", 0.7) for resp in agent_responses]
        temp_variance = np.var(temperatures)
        
        # Response diversity (simple text similarity)
        response_texts = [resp["response"].get("text", "") for resp in agent_responses]
        text_diversity = self._calculate_text_diversity(response_texts)
        
        return {
            "diversity_score": (model_diversity + temp_variance + text_diversity) / 3,
            "model_diversity": model_diversity,
            "temperature_variance": temp_variance,
            "text_diversity": text_diversity
        }
    
    def _calculate_text_diversity(self, texts: List[str]) -> float:
        """Calculate text diversity using simple word overlap."""
        if len(texts) < 2:
            return 0.0
        
        # Tokenize texts
        tokenized_texts = [set(text.lower().split()) for text in texts]
        
        # Calculate pairwise Jaccard distances
        distances = []
        for i in range(len(tokenized_texts)):
            for j in range(i + 1, len(tokenized_texts)):
                intersection = len(tokenized_texts[i] & tokenized_texts[j])
                union = len(tokenized_texts[i] | tokenized_texts[j])
                jaccard_distance = 1 - (intersection / union) if union > 0 else 0
                distances.append(jaccard_distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_consensus_metrics(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus metrics across agents."""
        if len(agent_responses) < 2:
            return {"consensus_score": 1.0, "agreement_rate": 1.0}
        
        # Extract answers
        answers = [self._extract_final_answer(resp["response"].get("text", "")) for resp in agent_responses]
        
        # Calculate agreement
        unique_answers = set(answers)
        agreement_rate = len(unique_answers) / len(answers)
        consensus_score = 1.0 - agreement_rate
        
        # Calculate confidence variance
        confidences = [resp["response"].get("confidence", 0.5) for resp in agent_responses]
        confidence_variance = np.var(confidences)
        
        return {
            "consensus_score": consensus_score,
            "agreement_rate": agreement_rate,
            "unique_answers": len(unique_answers),
            "confidence_variance": confidence_variance
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get baseline performance statistics."""
        agent_stats = [agent.get_stats() for agent in self.agents]
        
        return {
            "baseline_type": "MixtureOfAgents",
            "config": self.config.__dict__,
            "num_agents": len(self.agents),
            "agent_stats": agent_stats
        }

