"""
Configuration system for multi-agent evaluation framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import random
import itertools

from .agent import AgentConfig, AgentType, VarianceLevel
from .team import Architecture, DominanceLevel, TeamConfig


class ModelFamily(Enum):
    """LLM model families for variance experiments."""
    GPT = "gpt"           # OpenAI GPT models
    CLAUDE = "claude"     # Anthropic Claude models
    OPENSOURCE = "opensource"  # Open source models (Llama, Mistral, etc.)
    MIXTURE = "mixture"   # Mixture of Agents


@dataclass
class VarianceConfig:
    """Configuration for agent variance levels."""
    variance_level: VarianceLevel
    model_families: List[ModelFamily]
    temperature_range: tuple[float, float]
    prompt_variations: List[str]
    seed_variations: List[int]
    
    @classmethod
    def create_low_variance(cls, base_model: str = "gpt-3.5-turbo") -> "VarianceConfig":
        """Create low variance configuration (same LLM, same prompt, low temperature)."""
        return cls(
            variance_level=VarianceLevel.LOW,
            model_families=[ModelFamily.GPT],
            temperature_range=(0.1, 0.3),
            prompt_variations=["standard"],
            seed_variations=[42, 123, 456]
        )
    
    @classmethod
    def create_medium_variance(cls) -> "VarianceConfig":
        """Create medium variance configuration (same family, different prompts/temps)."""
        return cls(
            variance_level=VarianceLevel.MEDIUM,
            model_families=[ModelFamily.GPT],
            temperature_range=(0.3, 0.8),
            prompt_variations=["standard", "detailed", "concise", "analytical"],
            seed_variations=[42, 123, 456, 789, 101112]
        )
    
    @classmethod
    def create_high_variance(cls) -> "VarianceConfig":
        """Create high variance configuration (different LLMs, different styles)."""
        return cls(
            variance_level=VarianceLevel.HIGH,
            model_families=[ModelFamily.GPT, ModelFamily.CLAUDE, ModelFamily.OPENSOURCE],
            temperature_range=(0.1, 0.9),
            prompt_variations=["standard", "detailed", "concise", "analytical", "creative", "systematic"],
            seed_variations=[42, 123, 456, 789, 101112, 131415]
        )


@dataclass
class DominanceConfig:
    """Configuration for leader dominance levels."""
    dominance_level: DominanceLevel
    leader_override_threshold: float
    agenda_control: bool
    tie_break_power: bool
    early_stopping_authority: bool
    final_say_authority: bool
    
    @classmethod
    def create_none_dominance(cls) -> "DominanceConfig":
        """Create no dominance configuration (equal voting)."""
        return cls(
            dominance_level=DominanceLevel.NONE,
            leader_override_threshold=1.0,  # Never override
            agenda_control=False,
            tie_break_power=False,
            early_stopping_authority=False,
            final_say_authority=False
        )
    
    @classmethod
    def create_moderate_dominance(cls) -> "DominanceConfig":
        """Create moderate dominance configuration (tie-break + agenda)."""
        return cls(
            dominance_level=DominanceLevel.MODERATE,
            leader_override_threshold=0.8,
            agenda_control=True,
            tie_break_power=True,
            early_stopping_authority=False,
            final_say_authority=False
        )
    
    @classmethod
    def create_strong_dominance(cls) -> "DominanceConfig":
        """Create strong dominance configuration (soft-veto + final say)."""
        return cls(
            dominance_level=DominanceLevel.STRONG,
            leader_override_threshold=0.6,
            agenda_control=True,
            tie_break_power=True,
            early_stopping_authority=True,
            final_say_authority=True
        )


@dataclass
class TeamComposition:
    """Configuration for team composition and roles."""
    team_size: int
    agent_types: List[AgentType]
    role_descriptions: Dict[str, str]
    interaction_patterns: List[str]
    
    @classmethod
    def create_debate_team(cls, size: int = 4) -> "TeamComposition":
        """Create team for debate architecture."""
        return cls(
            team_size=size,
            agent_types=[AgentType.DEBATER] * (size - 1) + [AgentType.JUDGE],
            role_descriptions={
                "debater": "Argue for or against the given position with evidence and reasoning",
                "judge": "Evaluate arguments and make final decisions based on quality and evidence"
            },
            interaction_patterns=["parallel_debate", "judge_aggregation"]
        )
    
    @classmethod
    def create_orchestrator_team(cls) -> "TeamComposition":
        """Create team for orchestrator architecture."""
        return cls(
            team_size=4,
            agent_types=[AgentType.ORCHESTRATOR, AgentType.PLANNER, AgentType.CRITIC, AgentType.EXECUTOR],
            role_descriptions={
                "orchestrator": "Coordinate team activities and make high-level decisions",
                "planner": "Create detailed plans and strategies",
                "critic": "Review and critique plans and executions",
                "executor": "Implement plans and carry out tasks"
            },
            interaction_patterns=["orchestrator_coordination", "plan_critique_execute"]
        )
    
    @classmethod
    def create_roleplay_team(cls, size: int = 3) -> "TeamComposition":
        """Create team for role-play architecture."""
        return cls(
            team_size=size,
            agent_types=[AgentType.PEER] * size,
            role_descriptions={
                "peer": "Collaborate as equals to solve problems through discussion and consensus"
            },
            interaction_patterns=["peer_collaboration", "consensus_building"]
        )


@dataclass
class ExperimentConfig:
    """Configuration for the full experiment."""
    # Independent variables
    architectures: List[Architecture]
    variance_levels: List[VarianceLevel]
    dominance_levels: List[DominanceLevel]
    
    # Task configuration
    tasks: List[str]
    datasets: List[str]
    num_samples_per_condition: int = 100
    
    # Replication settings
    num_replications: int = 5
    random_seed: int = 42
    
    # Evaluation settings
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "consensus_difficulty", "deliberation_cost", 
        "output_disagreement", "distributional_distance"
    ])
    
    # Output settings
    output_dir: str = "results"
    save_intermediate_results: bool = True
    log_level: str = "INFO"
    
    def generate_conditions(self) -> List[Dict[str, Any]]:
        """Generate all experimental conditions (3x3x3 factorial design)."""
        conditions = []
        
        for arch, var, dom in itertools.product(
            self.architectures, 
            self.variance_levels, 
            self.dominance_levels
        ):
            conditions.append({
                "architecture": arch,
                "variance_level": var,
                "dominance_level": dom,
                "condition_id": f"{arch.value}_{var.value}_{dom.value}"
            })
        
        return conditions
    
    def create_team_config(
        self, 
        condition: Dict[str, Any], 
        team_id: str
    ) -> TeamConfig:
        """Create team configuration for a specific condition."""
        # Get variance and dominance configs
        var_config = self._get_variance_config(condition["variance_level"])
        dom_config = self._get_dominance_config(condition["dominance_level"])
        
        # Create team composition
        composition = self._get_team_composition(condition["architecture"])
        
        # Generate agent configs
        agent_configs = self._generate_agent_configs(
            composition, var_config, condition["architecture"]
        )
        
        return TeamConfig(
            team_id=team_id,
            architecture=condition["architecture"],
            dominance_level=condition["dominance_level"],
            variance_level=condition["variance_level"],
            agent_configs=agent_configs,
            leader_agent_id=agent_configs[0].agent_id if dom_config.agenda_control else None,
            leader_override_threshold=dom_config.leader_override_threshold,
            agenda_control=dom_config.agenda_control
        )
    
    def _get_variance_config(self, variance_level: VarianceLevel) -> VarianceConfig:
        """Get variance configuration for given level."""
        if variance_level == VarianceLevel.LOW:
            return VarianceConfig.create_low_variance()
        elif variance_level == VarianceLevel.MEDIUM:
            return VarianceConfig.create_medium_variance()
        else:  # HIGH
            return VarianceConfig.create_high_variance()
    
    def _get_dominance_config(self, dominance_level: DominanceLevel) -> DominanceConfig:
        """Get dominance configuration for given level."""
        if dominance_level == DominanceLevel.NONE:
            return DominanceConfig.create_none_dominance()
        elif dominance_level == DominanceLevel.MODERATE:
            return DominanceConfig.create_moderate_dominance()
        else:  # STRONG
            return DominanceConfig.create_strong_dominance()
    
    def _get_team_composition(self, architecture: Architecture) -> TeamComposition:
        """Get team composition for given architecture."""
        if architecture == Architecture.DEBATE_VOTE:
            return TeamComposition.create_debate_team()
        elif architecture == Architecture.ORCHESTRATOR_SUBAGENTS:
            return TeamComposition.create_orchestrator_team()
        else:  # ROLE_PLAY_TEAMWORK
            return TeamComposition.create_roleplay_team()
    
    def _generate_agent_configs(
        self, 
        composition: TeamComposition, 
        var_config: VarianceConfig,
        architecture: Architecture
    ) -> List[AgentConfig]:
        """Generate agent configurations based on composition and variance."""
        agent_configs = []
        
        for i, agent_type in enumerate(composition.agent_types):
            # Select model family based on variance
            model_family = random.choice(var_config.model_families)
            model_name = self._get_model_name(model_family)
            
            # Select temperature based on variance
            temp_min, temp_max = var_config.temperature_range
            temperature = random.uniform(temp_min, temp_max)
            
            # Select prompt variation
            prompt_template = random.choice(var_config.prompt_variations)
            
            # Create agent config
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type=agent_type,
                model_name=model_name,
                temperature=temperature,
                prompt_template=prompt_template,
                role_description=composition.role_descriptions.get(agent_type.value, ""),
                variance_level=var_config.variance_level,
                seed=random.choice(var_config.seed_variations)
            )
            
            agent_configs.append(config)
        
        return agent_configs
    
    def _get_model_name(self, model_family: ModelFamily) -> str:
        """Get specific model name for given family."""
        model_mapping = {
            ModelFamily.GPT: "gpt-3.5-turbo",
            ModelFamily.CLAUDE: "claude-3-sonnet-20240229",
            ModelFamily.OPENSOURCE: "llama-2-7b-chat",
            ModelFamily.MIXTURE: "mixture-of-agents"
        }
        return model_mapping[model_family]


# Predefined experiment configurations
def create_standard_experiment() -> ExperimentConfig:
    """Create standard 3x3x3 factorial experiment configuration."""
    return ExperimentConfig(
        architectures=[Architecture.DEBATE_VOTE, Architecture.ORCHESTRATOR_SUBAGENTS, Architecture.ROLE_PLAY_TEAMWORK],
        variance_levels=[VarianceLevel.LOW, VarianceLevel.MEDIUM, VarianceLevel.HIGH],
        dominance_levels=[DominanceLevel.NONE, DominanceLevel.MODERATE, DominanceLevel.STRONG],
        tasks=["math_reasoning", "multi_hop_qa", "code_generation"],
        datasets=["gsm8k", "math", "hotpotqa", "humaneval"],
        num_samples_per_condition=100,
        num_replications=5
    )


def create_quick_experiment() -> ExperimentConfig:
    """Create quick experiment for testing (reduced conditions)."""
    return ExperimentConfig(
        architectures=[Architecture.DEBATE_VOTE],
        variance_levels=[VarianceLevel.LOW, VarianceLevel.HIGH],
        dominance_levels=[DominanceLevel.NONE, DominanceLevel.STRONG],
        tasks=["math_reasoning"],
        datasets=["gsm8k"],
        num_samples_per_condition=10,
        num_replications=2
    )

