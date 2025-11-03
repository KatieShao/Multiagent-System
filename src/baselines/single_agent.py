"""
Single-agent baseline implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import random

from ..core.agent import Agent, AgentConfig, LLMAgent
from ..core.task import Task, TaskItem


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    num_samples: int = 1
    seed: Optional[int] = None


class SingleAgentBaseline(ABC):
    """Abstract base class for single-agent baselines."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.agent = self._create_agent()
        
        if config.seed is not None:
            random.seed(config.seed)
    
    @abstractmethod
    def _create_agent(self) -> Agent:
        """Create the agent for this baseline."""
        pass
    
    @abstractmethod
    async def solve_task(self, task_item: TaskItem) -> Dict[str, Any]:
        """Solve a task using this baseline method."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get baseline performance statistics."""
        return {
            "baseline_type": self.__class__.__name__,
            "config": self.config.__dict__,
            "agent_stats": self.agent.get_stats()
        }


class CoTBaseline(SingleAgentBaseline):
    """
    Chain-of-Thought baseline implementation.
    
    Uses a single agent with chain-of-thought prompting.
    """
    
    def _create_agent(self) -> Agent:
        """Create a CoT agent."""
        agent_config = AgentConfig(
            agent_id="cot_agent",
            agent_type="peer",  # Using peer as generic type
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt_template="cot_template"
        )
        return LLMAgent(agent_config)
    
    async def solve_task(self, task_item: TaskItem) -> Dict[str, Any]:
        """Solve task using chain-of-thought reasoning."""
        start_time = time.time()
        
        # Create CoT prompt
        cot_prompt = self._create_cot_prompt(task_item)
        
        # Generate response
        response = await self.agent.generate_response(cot_prompt)
        
        # Extract final answer
        final_answer = self._extract_final_answer(response.get("text", ""))
        
        duration = time.time() - start_time
        
        return {
            "baseline_type": "CoT",
            "task_id": task_item.item_id,
            "question": task_item.question,
            "ground_truth": task_item.ground_truth,
            "final_answer": final_answer,
            "reasoning": response.get("text", ""),
            "confidence": response.get("confidence", 0.0),
            "tokens_used": response.get("tokens_used", 0),
            "duration": duration,
            "timestamp": time.time()
        }
    
    def _create_cot_prompt(self, task_item: TaskItem) -> str:
        """Create chain-of-thought prompt for the task."""
        task_type = task_item.task_type.value
        
        if task_type == "math_reasoning":
            return f"""Solve this math problem step by step.

Problem: {task_item.question}

Let's think through this step by step:

1. First, I need to understand what the problem is asking.
2. Then, I'll identify the key information and what I need to find.
3. Next, I'll work through the solution step by step.
4. Finally, I'll check my answer.

Solution:
"""
        elif task_type == "multi_hop_qa":
            return f"""Answer this question by reasoning through multiple steps.

Question: {task_item.question}

Let me break this down:

1. First, I need to identify what information I need to find.
2. Then, I'll search for the relevant facts.
3. Next, I'll combine the information to answer the question.
4. Finally, I'll verify my answer.

Answer:
"""
        elif task_type == "code_generation":
            return f"""Write code to solve this problem.

Problem: {task_item.question}

Let me think about this:

1. First, I'll understand what the function should do.
2. Then, I'll plan the implementation.
3. Next, I'll write the code.
4. Finally, I'll test it with examples.

Code:
"""
        else:
            return f"""Solve this problem step by step.

Problem: {task_item.question}

Let me think through this:

1. First, I'll understand the problem.
2. Then, I'll work through the solution.
3. Finally, I'll provide my answer.

Solution:
"""
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from CoT response."""
        # Look for the final answer in the response
        lines = response.split('\n')
        
        # Look for patterns like "The answer is X" or "Final answer: X"
        for line in reversed(lines):
            line = line.strip()
            if any(phrase in line.lower() for phrase in ['the answer is', 'final answer', 'answer:', 'result:']):
                # Extract the answer after the phrase
                for phrase in ['the answer is', 'final answer', 'answer:', 'result:']:
                    if phrase in line.lower():
                        answer = line.lower().split(phrase)[-1].strip()
                        if answer:
                            return answer
        
        # If no clear answer found, return the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                return line
        
        return response.strip()


class SelfConsistencyBaseline(SingleAgentBaseline):
    """
    Self-consistency baseline implementation.
    
    Uses multiple samples from a single agent and aggregates the results.
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        self.num_samples = config.num_samples
    
    def _create_agent(self) -> Agent:
        """Create a self-consistency agent."""
        agent_config = AgentConfig(
            agent_id="sc_agent",
            agent_type="peer",
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt_template="sc_template"
        )
        return LLMAgent(agent_config)
    
    async def solve_task(self, task_item: TaskItem) -> Dict[str, Any]:
        """Solve task using self-consistency."""
        start_time = time.time()
        
        # Generate multiple samples
        samples = []
        for i in range(self.num_samples):
            # Vary temperature slightly for diversity
            temp = self.config.temperature + random.uniform(-0.1, 0.1)
            temp = max(0.1, min(1.0, temp))
            
            # Create prompt
            prompt = self._create_sc_prompt(task_item)
            
            # Generate response
            response = await self.agent.generate_response(prompt)
            samples.append(response)
        
        # Aggregate responses
        final_answer = self._aggregate_responses(samples)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(samples)
        
        duration = time.time() - start_time
        
        return {
            "baseline_type": "SelfConsistency",
            "task_id": task_item.item_id,
            "question": task_item.question,
            "ground_truth": task_item.ground_truth,
            "final_answer": final_answer,
            "samples": [s.get("text", "") for s in samples],
            "consistency_metrics": consistency_metrics,
            "num_samples": self.num_samples,
            "duration": duration,
            "timestamp": time.time()
        }
    
    def _create_sc_prompt(self, task_item: TaskItem) -> str:
        """Create self-consistency prompt."""
        # Similar to CoT but with slight variations
        base_prompt = f"""Solve this problem step by step.

Problem: {task_item.question}

Let me work through this carefully:

"""
        return base_prompt
    
    def _aggregate_responses(self, samples: List[Dict[str, Any]]) -> str:
        """Aggregate multiple responses to get final answer."""
        # Extract answers from each sample
        answers = []
        for sample in samples:
            answer = self._extract_final_answer(sample.get("text", ""))
            answers.append(answer)
        
        # Simple majority voting
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Return most common answer
        if answer_counts:
            return max(answer_counts, key=answer_counts.get)
        else:
            return answers[0] if answers else ""
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response (same as CoT)."""
        # Reuse CoT logic
        lines = response.split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            if any(phrase in line.lower() for phrase in ['the answer is', 'final answer', 'answer:', 'result:']):
                for phrase in ['the answer is', 'final answer', 'answer:', 'result:']:
                    if phrase in line.lower():
                        answer = line.lower().split(phrase)[-1].strip()
                        if answer:
                            return answer
        
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                return line
        
        return response.strip()
    
    def _calculate_consistency_metrics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consistency metrics across samples."""
        if len(samples) < 2:
            return {"consistency_score": 1.0, "agreement_rate": 1.0}
        
        # Extract answers
        answers = [self._extract_final_answer(s.get("text", "")) for s in samples]
        
        # Calculate agreement rate
        unique_answers = set(answers)
        agreement_rate = len(unique_answers) / len(answers) if answers else 0.0
        
        # Calculate consistency score (inverse of disagreement)
        consistency_score = 1.0 - agreement_rate
        
        return {
            "consistency_score": consistency_score,
            "agreement_rate": agreement_rate,
            "unique_answers": len(unique_answers),
            "total_samples": len(samples)
        }


class SingleAgentBaseline(SingleAgentBaseline):
    """
    Simple single-agent baseline without special prompting.
    """
    
    def _create_agent(self) -> Agent:
        """Create a simple single agent."""
        agent_config = AgentConfig(
            agent_id="single_agent",
            agent_type="peer",
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt_template="simple_template"
        )
        return LLMAgent(agent_config)
    
    async def solve_task(self, task_item: TaskItem) -> Dict[str, Any]:
        """Solve task using simple single agent."""
        start_time = time.time()
        
        # Create simple prompt
        prompt = f"Question: {task_item.question}\nAnswer:"
        
        # Generate response
        response = await self.agent.generate_response(prompt)
        
        duration = time.time() - start_time
        
        return {
            "baseline_type": "SingleAgent",
            "task_id": task_item.item_id,
            "question": task_item.question,
            "ground_truth": task_item.ground_truth,
            "final_answer": response.get("text", ""),
            "confidence": response.get("confidence", 0.0),
            "tokens_used": response.get("tokens_used", 0),
            "duration": duration,
            "timestamp": time.time()
        }

