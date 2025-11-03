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
from .prompts import get_benchmark_specific_prompts


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
        
        # Extract dataset from context
        dataset = context.get("dataset", "") if context else ""
        
        # Get benchmark-specific prompts
        prompt_config = get_benchmark_specific_prompts(dataset, task)
        roles_and_approaches = prompt_config["roles_and_approaches"]
        debater_base_prompt = prompt_config["debater_base_prompt"]
        
        # All debaters respond in parallel
        debater_index = 0
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type != AgentType.JUDGE:
                # Get role and approach for this debater
                if debater_index < len(roles_and_approaches):
                    role_info = roles_and_approaches[debater_index]
                else:
                    role_info = roles_and_approaches[debater_index % len(roles_and_approaches)]
                
                # Create specialized prompt using benchmark-specific template
                debate_prompt = debater_base_prompt.format(
                    role=role_info['role'],
                    task=task,
                    approach=role_info['approach']
                )
                
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
            
            # Get benchmark-specific judge prompt
            dataset = context.get("dataset", "") if context else ""
            prompt_config = get_benchmark_specific_prompts(dataset, task_text)
            judge_base_prompt = prompt_config["judge_base_prompt"]
            
            # Format the judge prompt with task and responses
            judge_prompt = judge_base_prompt.format(
                task=task_text,
                all_responses=all_responses
            )
            
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
        
        # Store task for reference
        self.current_task = task
        
        # Enhance context
        enhanced_context = context.copy() if context else {}
        enhanced_context["task"] = task
        
        # Phase 1: Planning
        plan = await self._create_plan(task, enhanced_context)
        
        # Phase 2: Execution with criticism
        execution_results = await self._execute_with_criticism(plan, task, enhanced_context)
        
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
        planner_agent = None
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type == AgentType.PLANNER:
                planner_agent = agent
                break
        
        if planner_agent:
            plan_prompt = f"""You are a planner tasked with creating a detailed plan to solve the following task:

{task}

Create a step-by-step plan. Be specific and clear about each step."""
            
            plan_response = await planner_agent.generate_response(plan_prompt, context)
            return {
                "plan": plan_response.get("text", ""),
                "planner_response": plan_response
            }
        else:
            return {"plan": "No planner agent available"}
    
    async def _execute_with_criticism(
        self, 
        plan: Dict[str, Any], 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute plan with critic feedback."""
        # Get executor and critic agents
        executor_agent = None
        critic_agent = None
        
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type == AgentType.EXECUTOR:
                executor_agent = agent
            elif agent.config.agent_type == AgentType.CRITIC:
                critic_agent = agent
        
        plan_text = plan.get("plan", "") if isinstance(plan, dict) else str(plan)
        
        # Executor executes the plan
        if executor_agent:
            execution_prompt = f"""You are an executor. Execute the following plan to solve the task:

Task: {task}

Plan:
{plan_text}

Execute the plan step by step and provide your solution."""
            
            execution_response = await executor_agent.generate_response(execution_prompt, context)
            
            # Critic reviews the execution
            if critic_agent:
                critic_prompt = f"""You are a critic. Review the execution of the following plan:

Task: {task}

Plan:
{plan_text}

Execution Result:
{execution_response.get("text", "")}

Provide critical feedback and suggest improvements."""
                
                critic_response = await critic_agent.generate_response(critic_prompt, context)
                
                return {
                    "execution": execution_response.get("text", ""),
                    "criticism": critic_response.get("text", ""),
                    "execution_response": execution_response,
                    "critic_response": critic_response
                }
            else:
                return {
                    "execution": execution_response.get("text", ""),
                    "execution_response": execution_response
                }
        else:
            return {"execution_results": "No executor agent available"}
    
    async def _orchestrate_final_result(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate final result from execution."""
        orchestrator_agent = None
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type == AgentType.ORCHESTRATOR:
                orchestrator_agent = agent
                break
        
        if orchestrator_agent:
            execution_text = execution_results.get("execution", "")
            criticism = execution_results.get("criticism", "")
            
            orchestration_prompt = f"""You are an orchestrator. Synthesize the final result from the execution and criticism:

Execution Result:
{execution_text}

Critic Feedback:
{criticism}

Provide the final, refined answer integrating both the execution and the critic's feedback."""
            
            orchestration_response = await orchestrator_agent.generate_response(orchestration_prompt)
            
            return {
                "final_answer": orchestration_response.get("text", ""),
                "confidence": orchestration_response.get("confidence", 0.8),
                "orchestration_response": orchestration_response,
                "execution_results": execution_results,
                "num_rounds": 1,
                "total_tokens": orchestration_response.get("tokens_used", 0)
            }
        else:
            # Fallback: use execution result directly
            return {
                "final_answer": execution_results.get("execution", ""),
                "confidence": 0.7,
                "execution_results": execution_results,
                "num_rounds": 1,
                "total_tokens": 0
            }


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
        
        # Store task for reference
        self.current_task = task
        
        # Enhance context
        enhanced_context = context.copy() if context else {}
        enhanced_context["task"] = task
        
        # Phase 1: Role assignment and initial responses
        role_responses = await self._assign_roles_and_respond(task, enhanced_context)
        
        # Phase 2: Peer-to-peer collaboration
        collaboration_results = await self._collaborate_peers(role_responses, task, enhanced_context)
        
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
        role_responses = {}
        
        # Each peer provides their initial response
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type == AgentType.PEER:
                peer_prompt = f"""You are collaborating with peers to solve the following task:

{task}

Provide your initial thoughts and approach to solving this task. Be collaborative and considerate of other perspectives."""
                
                response = await agent.generate_response(peer_prompt, context)
                role_responses[agent_id] = response
        
        return role_responses
    
    async def _collaborate_peers(
        self, 
        role_responses: Dict[str, Any], 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Conduct peer-to-peer collaboration."""
        # Format previous responses for peer review
        previous_responses = "\n\n".join([
            f"**Peer {agent_id}:**\n{resp.get('text', '')}" 
            for agent_id, resp in role_responses.items()
        ])
        
        collaboration_rounds = {}
        max_collaboration_rounds = 2  # Allow a few rounds of discussion
        
        for round_num in range(max_collaboration_rounds):
            round_responses = {}
            
            for agent_id, agent in self.agents.items():
                if agent.config.agent_type == AgentType.PEER:
                    if round_num == 0:
                        # First round: respond to initial thoughts
                        collab_prompt = f"""You are collaborating with peers to solve:

{task}

Initial thoughts from peers:
{previous_responses}

Provide your response, building on or responding to your peers' ideas."""
                    else:
                        # Subsequent rounds: respond to previous round
                        prev_round = "\n\n".join([
                            f"**Peer {a_id}:**\n{r.get('text', '')}" 
                            for a_id, r in collaboration_rounds[round_num - 1].items()
                        ])
                        collab_prompt = f"""Continuing collaboration on:

{task}

Previous round responses:
{prev_round}

Continue the discussion, refine ideas, or build consensus."""
                    
                    response = await agent.generate_response(collab_prompt, context)
                    round_responses[agent_id] = response
            
            collaboration_rounds[round_num] = round_responses
        
        return {
            "collaboration_rounds": collaboration_rounds,
            "all_responses": round_responses  # Final round responses
        }
    
    async def _build_consensus(self, collaboration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus from collaboration."""
        # Get final round responses
        all_responses = collaboration_results.get("all_responses", {})
        
        if not all_responses:
            # Fallback if no responses
            return {
                "final_answer": "",
                "confidence": 0.0,
                "consensus": "No responses available"
            }
        
        # Simple consensus: extract answers and find most common or best
        answers = []
        for agent_id, resp in all_responses.items():
            answer_text = resp.get("text", "").strip()
            if answer_text:
                answers.append(answer_text)
        
        if not answers:
            return {
                "final_answer": "",
                "confidence": 0.0,
                "consensus": "No valid answers"
            }
        
        # For now, use the first peer's final response as consensus
        # TODO: Implement more sophisticated consensus building
        first_peer = next(iter(all_responses.values()))
        final_answer = first_peer.get("text", "")
        
        # Calculate confidence based on agreement
        # Simple: check if responses are similar
        unique_answers = set(answers[:3])  # Check first 3 for similarity
        confidence = 0.7 if len(unique_answers) == 1 else 0.5
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "consensus": "Peer consensus",
            "individual_responses": all_responses,
            "num_rounds": len(collaboration_results.get("collaboration_rounds", {})),
            "total_tokens": sum(r.get("tokens_used", 0) for r in all_responses.values())
        }


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

