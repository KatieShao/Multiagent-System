"""
Core framework components for multi-agent evaluation.
"""

from .agent import Agent, AgentConfig
from .team import Team, TeamConfig
from .task import Task, TaskConfig
from .config import ExperimentConfig
from .experiment import ExperimentResult, ExperimentRunner, run_experiment, run_quick_experiment
from .team_utils import create_multi_model_team, create_qwen_multi_model_team

__all__ = [
    "Agent", "AgentConfig",
    "Team", "TeamConfig", 
    "Task", "TaskConfig",
    "ExperimentConfig",
    "ExperimentResult", "ExperimentRunner", 
    "run_experiment", "run_quick_experiment",
    "create_multi_model_team", "create_qwen_multi_model_team"
]

