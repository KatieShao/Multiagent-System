"""
Utility functions for creating teams with multiple models.
"""

from typing import List, Dict, Optional, Any
from .agent import AgentConfig, AgentType, VarianceLevel
from .team import TeamConfig, Architecture, DominanceLevel, create_team


def create_multi_model_team(
    team_id: str,
    architecture: Architecture,
    dominance_level: DominanceLevel,
    model_configs: List[Dict[str, Any]],
    base_model_path: Optional[str] = None
) -> TeamConfig:
    """
    Create a team configuration with multiple different models.
    
    Args:
        team_id: Unique identifier for the team
        architecture: Team architecture (debate_vote, orchestrator_subagents, etc.)
        dominance_level: Leader dominance level
        model_configs: List of model configurations, each with:
            - agent_id: str
            - agent_type: AgentType
            - model_name: str (e.g., "qwen2.5-3b-instruct", "qwen2.5-7b-instruct")
            - model_path: str (path to the model on HPC cluster)
            - temperature: float (optional, default 0.7)
            - max_tokens: int (optional, default 1000)
            - role_description: str (optional)
        base_model_path: Base path for models if model_path not specified per agent
    
    Returns:
        TeamConfig ready to create a team with create_team()
    
    Example:
        >>> model_configs = [
        ...     {"agent_id": "debater_0", "agent_type": AgentType.DEBATER,
        ...      "model_name": "qwen2.5-3b-instruct", "model_path": "/path/to/qwen2.5-3b-instruct"},
        ...     {"agent_id": "debater_1", "agent_type": AgentType.DEBATER,
        ...      "model_name": "qwen2.5-7b-instruct", "model_path": "/path/to/qwen2.5-7b-instruct"},
        ...     {"agent_id": "debater_2", "agent_type": AgentType.DEBATER,
        ...      "model_name": "qwen2.5-32b-instruct", "model_path": "/path/to/qwen2.5-32b-instruct"},
        ...     {"agent_id": "judge", "agent_type": AgentType.JUDGE,
        ...      "model_name": "qwen2.5-72b-instruct", "model_path": "/path/to/qwen2.5-72b-instruct"},
        ... ]
        >>> team_config = create_multi_model_team(
        ...     "multi_model_team", Architecture.DEBATE_VOTE, DominanceLevel.NONE, model_configs
        ... )
        >>> team = create_team(team_config)
    """
    agent_configs = []
    
    for model_config in model_configs:
        # Extract required fields
        agent_id = model_config["agent_id"]
        agent_type = model_config["agent_type"]
        model_name = model_config["model_name"]
        
        # Get model path (prioritize per-agent path, then base path)
        model_path = model_config.get("model_path")
        if not model_path and base_model_path:
            # Try to construct path from base + model_name
            model_path = f"{base_model_path}/{model_name}"
        
        # Get optional fields
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 1000)
        role_description = model_config.get("role_description", "")
        variance_level = model_config.get("variance_level", VarianceLevel.HIGH)  # Default HIGH for multi-model
        seed = model_config.get("seed", 42)
        
        # Create agent config
        agent_config = AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            model_name=model_name,
            model_path=model_path,  # Direct model path support
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_template=model_config.get("prompt_template"),
            role_description=role_description,
            variance_level=variance_level,
            seed=seed,
            custom_params={"model_path": model_path} if model_path else None  # Also in custom_params for backward compat
        )
        
        agent_configs.append(agent_config)
    
    # Determine variance level based on number of unique models
    unique_models = set(config["model_name"] for config in model_configs)
    if len(unique_models) > 1:
        variance_level = VarianceLevel.HIGH
    else:
        variance_level = VarianceLevel.LOW
    
    # Create team config
    team_config = TeamConfig(
        team_id=team_id,
        architecture=architecture,
        dominance_level=dominance_level,
        variance_level=variance_level,
        agent_configs=agent_configs,
        max_rounds=10,
        consensus_threshold=0.8,
        debate_rounds=3,
        voting_strategy="majority",
        early_stopping=True
    )
    
    return team_config


def create_qwen_multi_model_team(
    team_id: str,
    architecture: Architecture,
    dominance_level: DominanceLevel,
    model_sizes: List[str],
    base_model_path: str,
    team_composition: Optional[Dict[str, Any]] = None
) -> TeamConfig:
    """
    Convenience function to create a team with multiple Qwen models of different sizes.
    
    Args:
        team_id: Unique identifier for the team
        architecture: Team architecture
        dominance_level: Leader dominance level
        model_sizes: List of model sizes, e.g., ["3b", "7b", "32b", "72b"]
        base_model_path: Base path where Qwen models are stored (e.g., "/gpfs/models/qwen2.5")
        team_composition: Optional dict with:
            - debater_count: int (default: len(model_sizes) - 1)
            - judge_model: str (default: largest model)
            - temperatures: List[float] (optional, default varies by size)
    
    Returns:
        TeamConfig ready to create a team
    
    Example:
        >>> team_config = create_qwen_multi_model_team(
        ...     "qwen_team", Architecture.DEBATE_VOTE, DominanceLevel.NONE,
        ...     ["3b", "7b", "32b", "72b"],
        ...     "/gpfs/models/qwen2.5"
        ... )
    """
    if team_composition is None:
        team_composition = {}
    
    debater_count = team_composition.get("debater_count", len(model_sizes) - 1)
    judge_model = team_composition.get("judge_model", model_sizes[-1])  # Use largest by default
    temperatures = team_composition.get("temperatures", None)
    
    # Ensure we have enough models for debaters + judge
    if len(model_sizes) < debater_count + 1:
        raise ValueError(f"Need at least {debater_count + 1} models (for {debater_count} debaters + 1 judge), got {len(model_sizes)}")
    
    model_configs = []
    
    # Create debater configs
    debater_models = model_sizes[:debater_count]
    for i, size in enumerate(debater_models):
        model_name = f"qwen2.5-{size}-instruct"
        model_path = f"{base_model_path}/{model_name}"
        
        # Default temperatures based on model size (smaller = lower temp)
        if temperatures and i < len(temperatures):
            temp = temperatures[i]
        else:
            # Smaller models get lower temperature for more focused responses
            size_num = int(size.replace('b', ''))
            temp = max(0.3, min(0.7, 0.3 + (size_num / 72) * 0.4))
        
        model_configs.append({
            "agent_id": f"debater_{i}",
            "agent_type": AgentType.DEBATER,
            "model_name": model_name,
            "model_path": model_path,
            "temperature": temp,
            "max_tokens": 300,
            "role_description": f"Debater using {model_name} model"
        })
    
    # Create judge config
    judge_model_name = f"qwen2.5-{judge_model}-instruct"
    judge_path = f"{base_model_path}/{judge_model_name}"
    
    model_configs.append({
        "agent_id": "judge",
        "agent_type": AgentType.JUDGE,
        "model_name": judge_model_name,
        "model_path": judge_path,
        "temperature": 0.1,  # Low temperature for consistent judging
        "max_tokens": 150,
        "role_description": f"Judge using {judge_model_name} model"
    })
    
    return create_multi_model_team(
        team_id=team_id,
        architecture=architecture,
        dominance_level=dominance_level,
        model_configs=model_configs
    )
