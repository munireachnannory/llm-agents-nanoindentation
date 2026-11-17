"""
Nanoindentation Multi-Agent System
===================================
LangGraph-based autonomous nanoindentation experiment control.

This package provides:
- Multi-agent workflow for autonomous nanoindentation experiments
- Integration with InstrumentClient for instrument control
- Hypothesis generation, testing, and validation
- Literature-informed experiment planning
"""

from .state import AgentState, create_initial_state
from .graph import build_nanoindent_graph, run_experiment

__all__ = [
    "AgentState",
    "create_initial_state",
    "build_nanoindent_graph",
    "run_experiment",
]

__version__ = "0.1.0"
