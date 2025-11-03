"""
Team and architecture implementations for multi-agent systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
import time
from collections import defaultdict

from .agent import Agent, AgentConfig, AgentType, VarianceLevel


class Architecture(Enum):
    """Multi-agent system architectures."""
    DEBATE_VOTE = "debate_vote"           # Parallel agents debate, judge aggregates
    ORCHESTRATOR_SUBAGENTS = "orchestrator_subagents"  # Planner/critic/executors
    ROLE_PLAY_TEAMWORK = "role_play_teamwork"  # Peer-to-peer (CAMEL-style)


class DominanceLevel(Enum):
    """Leader dominance levels in team structures."""
    NONE = "none"        # Equal vote (majority or mean of confidences)
    MODERATE = "moderate"  # Leader has tie-break + agenda setting
    STRONG = "strong"    # Leader has soft-veto or final say


@dataclass
class TeamConfig:
    """Configuration for a multi-agent team."""
    team_id: str
    architecture: Architecture
    dominance_level: DominanceLevel
    variance_level: VarianceLevel
    agent_configs: List[AgentConfig]
    max_rounds: int = 10
    consensus_threshold: float = 0.8
    timeout_seconds: float = 300.0
    
    # Architecture-specific parameters
    debate_rounds: int = 3
    voting_strategy: str = "majority"  # majority, weighted, confidence
    early_stopping: bool = True
    
    # Leader-specific parameters
    leader_agent_id: Optional[str] = None
    leader_override_threshold: float = 0.9
    agenda_control: bool = True


class Team(ABC):
    """
    Abstract base class for multi-agent teams.
    
    Implements different architectures and dominance patterns.
    """
    
    def __init__(self, config: TeamConfig):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.conversation_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents based on team configuration."""
        from .agent import LLMAgent
        
        for agent_config in self.config.agent_configs:
            agent = LLMAgent(agent_config)
            self.agents[agent_config.agent_id] = agent
    
    @abstractmethod
    async def solve_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a task using the team's architecture.
        
        Args:
            task: The task description
            context: Additional context
            
        Returns:
            Dictionary containing solution, reasoning, and metadata
        """
        pass
    
    @abstractmethod
    async def deliberate(
        self, 
        topic: str, 
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Conduct deliberation on a topic.
        
        Args:
            topic: The topic to deliberate on
            max_rounds: Maximum number of deliberation rounds
            
        Returns:
            Dictionary containing deliberation results and consensus
        """
        pass
    
    def get_team_stats(self) -> Dict[str, Any]:
        """Get team performance statistics."""
        agent_stats = {agent_id: agent.get_stats() for agent_id, agent in self.agents.items()}
        
        return {
            "team_id": self.config.team_id,
            "architecture": self.config.architecture.value,
            "dominance_level": self.config.dominance_level.value,
            "variance_level": self.config.variance_level.value,
            "agent_count": len(self.agents),
            "conversation_length": len(self.conversation_log),
            "agent_stats": agent_stats,
            "performance_metrics": self.performance_metrics
        }


class DebateVoteTeam(Team):
    """
    Debate and vote architecture implementation.
    
    Agents debate in parallel, then a judge aggregates the results.
    """
    
    async def solve_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Solve task using debate and vote architecture."""
        start_time = time.time()
        
        # Store task for judge reference
        self.current_task = task
        
        # Enhance context with task information
        enhanced_context = context.copy() if context else {}
        enhanced_context["task"] = task
        enhanced_context["is_multiple_choice"] = "0) " in task or "1) " in task or "Options:" in task
        
        # Phase 1: Parallel debate
        debate_results = await self._conduct_debate(task, enhanced_context)
        
        # Phase 2: Judge aggregation
        final_decision = await self._aggregate_votes(debate_results, task, enhanced_context)
        
        # Log the interaction
        self.conversation_log.append({
            "task": task,
            "debate_results": debate_results,
            "final_decision": final_decision,
            "timestamp": time.time(),
            "duration": time.time() - start_time
        })
        
        return final_decision
    
    async def deliberate(
        self, 
        topic: str, 
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Conduct deliberation using debate format."""
        rounds = max_rounds or self.config.debate_rounds
        deliberation_log = []
        
        for round_num in range(rounds):
            # Each agent provides their perspective
            round_responses = {}
            for agent_id, agent in self.agents.items():
                if agent.config.agent_type.value != "judge":
                    response = await agent.generate_response(
                        f"Round {round_num + 1}: {topic}",
                        context={"round": round_num, "previous_responses": deliberation_log}
                    )
                    round_responses[agent_id] = response
            
            deliberation_log.append({
                "round": round_num + 1,
                "responses": round_responses
            })
            
            # Check for early consensus
            if self.config.early_stopping:
                consensus = await self._check_consensus(round_responses)
                if consensus["achieved"]:
                    break
        
        # Final aggregation
        final_result = await self._aggregate_deliberation(deliberation_log)
        
        return {
            "topic": topic,
            "deliberation_log": deliberation_log,
            "final_result": final_result,
            "rounds_completed": len(deliberation_log)
        }
    
    async def _conduct_debate(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conduct parallel debate among agents."""
        debate_results = {}
        
        # Extract task metadata if available
        task_type = context.get("task_type", "") if context else ""
        is_multiple_choice = (
            "commonsense_qa" in task_type or 
            "hellaswag" in task_type.lower() or 
            (context.get("is_multiple_choice", False) if context else False) or
            "0) " in task or "1) " in task or "Options:" in task
        )
        
        # Create role-based prompts for different debaters
        roles_and_approaches = [
            {
                "role": "You are a logical reasoning expert",
                "approach": "Analyze the logical connections and causal relationships between the context and each option. Think step-by-step about what would naturally follow."
            },
            {
                "role": "You are a commonsense knowledge specialist",
                "approach": "Use your understanding of how the world works. Consider everyday scenarios, human behavior, and physical laws. Think about what makes the most sense in context."
            },
            {
                "role": "You are a language understanding expert",
                "approach": "Focus on linguistic coherence and semantic fit. Analyze how well each option connects to the context linguistically and meaningfully."
            }
        ]
        
        # All debaters respond in parallel
        debater_index = 0
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type != AgentType.JUDGE:
                # Get role and approach for this debater
                if debater_index < len(roles_and_approaches):
                    role_info = roles_and_approaches[debater_index]
                else:
                    role_info = roles_and_approaches[debater_index % len(roles_and_approaches)]
                
                # Create specialized prompt
                if is_multiple_choice:
                    debate_prompt = f"""{role_info['role']}. You are solving a commonsense reasoning task.

{task}

{role_info['approach']}

For each option, briefly explain:
- Why it does or doesn't make sense given the context
- How it fits (or doesn't fit) with commonsense understanding

After analyzing all options, provide your answer in this exact format:
**Answer: [number]**

Where [number] is 0, 1, 2, or 3 corresponding to the best option."""
                else:
                    debate_prompt = f"""{role_info['role']}. You are solving the following task:

{task}

{role_info['approach']}

Provide your best answer with clear reasoning."""
                
                response = await agent.generate_response(debate_prompt, context)
                debate_results[agent_id] = response
                debater_index += 1
        
        return debate_results
    
    async def _aggregate_votes(self, debate_results: Dict[str, Any], task: str = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate votes using configured strategy."""
        if not debate_results:
            return {
                "final_answer": "",
                "confidence": 0.0,
                "voting_method": self.config.voting_strategy,
                "individual_responses": {}
            }
        
        # Get task text
        task_text = task if task else (self.current_task if hasattr(self, 'current_task') else 'Task')
        
        # Get judge agent if available
        judge_agent = None
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type == AgentType.JUDGE:
                judge_agent = agent
                break
        
        if judge_agent:
            # Judge evaluates all responses
            # Format responses clearly
            formatted_responses = []
            for agent_id, resp in debate_results.items():
                response_text = resp.get('text', '').strip()
                formatted_responses.append(f"**Agent {agent_id}:**\n{response_text}\n")
            
            all_responses = "\n".join(formatted_responses)
            
            # Detect if this is a multiple choice task
            is_multiple_choice = (
                context.get("is_multiple_choice", False) if context 
                else ("0) " in task_text or "1) " in task_text or "Options:" in task_text)
            )
            
            if is_multiple_choice:
                judge_prompt = f"""You are an expert judge evaluating responses to a multiple-choice commonsense reasoning task.

**Original Task:**
{task_text}

**Agent Responses:**
{all_responses}

Your task:
1. Review each agent's reasoning and answer choice
2. Consider which agent provided the most sound logical analysis
3. Synthesize the best reasoning from all agents if needed
4. Make the final decision

**IMPORTANT:** You must output your answer in this exact format:
**Answer: [number]**

Where [number] is 0, 1, 2, or 3 corresponding to the best option. Only output the number, nothing else."""
            else:
                judge_prompt = f"""You are an expert judge evaluating multiple responses to the following task:

{task_text}

**Agent Responses:**
{all_responses}

Review all responses, evaluate their quality and reasoning, and provide the best final answer. Synthesize insights from multiple agents if appropriate."""
            
            judge_response = await judge_agent.generate_response(judge_prompt)
            final_answer = judge_response.get('text', '')
            confidence = judge_response.get('confidence', 0.8)
            
            # Debug: Check if this is a placeholder response
            if judge_response.get('is_placeholder', False):
                print(f"⚠️  WARNING: Judge response is placeholder! Model may not be loaded correctly.")
                print(f"   Judge response preview: {final_answer[:100]}")
            
            # Debug: Print the full judge response for debugging
            print(f"🔍 Judge raw response: {final_answer[:200]}")
        else:
            # No judge - use voting strategy
            if self.config.voting_strategy == "majority":
                # Extract answers (simple - first line or number)
                answers = {}
                for agent_id, resp in debate_results.items():
                    text = resp.get('text', '').strip()
                    # Try to extract first answer/number
                    lines = text.split('\n')
                    answer = lines[0] if lines else text[:50]
                    answers[answer] = answers.get(answer, []) + [agent_id]
                
                # Most common answer
                final_answer = max(answers.items(), key=lambda x: len(x[1]))[0] if answers else ""
                confidence = 0.7
                
            elif self.config.voting_strategy == "weighted" or self.config.voting_strategy == "confidence":
                # Weight by confidence
                best_response = max(
                    debate_results.items(), 
                    key=lambda x: x[1].get('confidence', 0.0)
                )
                final_answer = best_response[1].get('text', '').strip()
                confidence = best_response[1].get('confidence', 0.8)
            else:
                # Default - just use first response
                first_resp = next(iter(debate_results.values()))
                final_answer = first_resp.get('text', '').strip()
                confidence = first_resp.get('confidence', 0.8)
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "voting_method": self.config.voting_strategy,
            "individual_responses": debate_results,
            "num_rounds": 1,
            "total_tokens": sum(r.get('tokens_used', 0) for r in debate_results.values())
        }
    
    async def _check_consensus(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Check if consensus has been reached."""
        # TODO: Implement consensus checking logic
        return {"achieved": False, "consensus_score": 0.5}
    
    async def _aggregate_deliberation(self, deliberation_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate deliberation results."""
        # TODO: Implement deliberation aggregation
        return {"consensus": "Final deliberation result"}


class OrchestratorSubagentsTeam(Team):
    """
    Orchestrator with sub-agents architecture implementation.
    
    A central orchestrator coordinates planner, critic, and executor agents.
    """
    
    async def solve_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Solve task using orchestrator-subagents architecture."""
        start_time = time.time()
        
        # Phase 1: Planning
        plan = await self._create_plan(task, context)
        
        # Phase 2: Execution with criticism
        execution_results = await self._execute_with_criticism(plan, task, context)
        
        # Phase 3: Final orchestration
        final_result = await self._orchestrate_final_result(execution_results)
        
        # Log the interaction
        self.conversation_log.append({
            "task": task,
            "plan": plan,
            "execution_results": execution_results,
            "final_result": final_result,
            "timestamp": time.time(),
            "duration": time.time() - start_time
        })
        
        return final_result
    
    async def deliberate(
        self, 
        topic: str, 
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Conduct deliberation using orchestrator-subagents format."""
        # TODO: Implement orchestrator-based deliberation
        return {"deliberation_result": "Orchestrator deliberation"}
    
    async def _create_plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a plan using the planner agent."""
        # TODO: Implement planning logic
        return {"plan": "Generated plan for task"}
    
    async def _execute_with_criticism(
        self, 
        plan: Dict[str, Any], 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute plan with critic feedback."""
        # TODO: Implement execution with criticism
        return {"execution_results": "Executed with criticism"}
    
    async def _orchestrate_final_result(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate final result from execution."""
        # TODO: Implement final orchestration
        return {"final_result": "Orchestrated final result"}


class RolePlayTeamworkTeam(Team):
    """
    Role-play teamwork architecture implementation.
    
    Peer-to-peer collaboration similar to CAMEL framework.
    """
    
    async def solve_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Solve task using role-play teamwork architecture."""
        start_time = time.time()
        
        # Phase 1: Role assignment and initial responses
        role_responses = await self._assign_roles_and_respond(task, context)
        
        # Phase 2: Peer-to-peer collaboration
        collaboration_results = await self._collaborate_peers(role_responses, task, context)
        
        # Phase 3: Consensus building
        final_result = await self._build_consensus(collaboration_results)
        
        # Log the interaction
        self.conversation_log.append({
            "task": task,
            "role_responses": role_responses,
            "collaboration_results": collaboration_results,
            "final_result": final_result,
            "timestamp": time.time(),
            "duration": time.time() - start_time
        })
        
        return final_result
    
    async def deliberate(
        self, 
        topic: str, 
        max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Conduct deliberation using role-play format."""
        # TODO: Implement role-play deliberation
        return {"deliberation_result": "Role-play deliberation"}
    
    async def _assign_roles_and_respond(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign roles and get initial responses."""
        # TODO: Implement role assignment and response
        return {"role_responses": "Assigned roles and responses"}
    
    async def _collaborate_peers(
        self, 
        role_responses: Dict[str, Any], 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Conduct peer-to-peer collaboration."""
        # TODO: Implement peer collaboration
        return {"collaboration_results": "Peer collaboration results"}
    
    async def _build_consensus(self, collaboration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus from collaboration."""
        # TODO: Implement consensus building
        return {"consensus": "Built consensus"}


def create_team(config: TeamConfig) -> Team:
    """Factory function to create teams based on architecture."""
    if config.architecture == Architecture.DEBATE_VOTE:
        return DebateVoteTeam(config)
    elif config.architecture == Architecture.ORCHESTRATOR_SUBAGENTS:
        return OrchestratorSubagentsTeam(config)
    elif config.architecture == Architecture.ROLE_PLAY_TEAMWORK:
        return RolePlayTeamworkTeam(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

