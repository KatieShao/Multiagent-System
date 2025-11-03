"""
Baseline implementations for multi-agent system evaluation.
"""

from .single_agent import SingleAgentBaseline, CoTBaseline, SelfConsistencyBaseline
from .mixture_of_agents import MixtureOfAgentsBaseline

__all__ = [
    "SingleAgentBaseline", "CoTBaseline", "SelfConsistencyBaseline",
    "MixtureOfAgentsBaseline"
]

