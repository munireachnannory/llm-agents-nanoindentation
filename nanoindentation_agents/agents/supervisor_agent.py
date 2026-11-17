"""
supervisor_agent.py
===================
Supervisor node using LangGraph Command for dynamic worker routing.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.types import Command

from ..config import get_config
from ..state import AgentState

WorkerName = Literal[
    "instrument",
    "analysis",
    "hypothesis",
    "literature",
    "policy",
    "conclude",
]


def _state_summary(state: AgentState) -> str:
    """Build a summary of current state for supervisor decision."""
    parts = [
        f"stage={state.get('current_stage')}",
        f"iteration={state.get('iteration')}/{state.get('max_iterations')}",
        f"ic_cycle={state.get('ic_cycle')}/{state.get('ic_max_cycles')}",
        f"hypothesis_approved={state.get('user_approved')}",
        f"measurement_decision={state.get('measurement_critique_decision')}",
        f"instrument_conclude={state.get('instrument_conclude')}",
        f"indent_arrays={len(state.get('indent_history', []))}",
        f"visited_targets={len(state.get('visited_target_keys', []))}",
    ]
    return "\n".join(parts)


def _evidence_summary(state: AgentState) -> dict[str, Any]:
    """Summarize collected evidence."""
    return {
        "indent_arrays": len(state.get("indent_history", [])),
        "images_captured": len(state.get("image_history", [])),
        "visited_locations": len(state.get("visited_target_keys", [])),
        "instrument_actions": len(state.get("decision_history", [])),
    }


def _can_confirm_hypothesis(state: AgentState, cfg=None) -> tuple[bool, str]:
    """Check if there's enough evidence to confirm hypothesis."""
    cfg = cfg or get_config()
    ev = _evidence_summary(state)
    
    min_arrays = cfg.experiment.min_indent_arrays_for_confirmation
    min_locations = cfg.experiment.min_visited_locations_for_confirmation
    min_actions = cfg.experiment.min_instrument_actions_before_confirmation
    
    if ev["indent_arrays"] < min_arrays:
        return False, f"Need {min_arrays} indent arrays, have {ev['indent_arrays']}"
    if ev["visited_locations"] < min_locations:
        return False, f"Need {min_locations} locations, visited {ev['visited_locations']}"
    if ev["instrument_actions"] < min_actions:
        return False, f"Need {min_actions} actions, have {ev['instrument_actions']}"
    
    return True, "Evidence requirements met"


def supervisor_node(state: AgentState) -> Command:
    """
    Supervisor node that routes to the next worker.
    
    Uses state-based logic to determine the optimal next step.
    Can be enhanced with LLM-based reasoning.
    """
    cfg = get_config()

    # Check for confirmed hypothesis
    if state.get("measurement_critique_decision") == "hypothesis_confirmed":
        ok, gate_reason = _can_confirm_hypothesis(state, cfg)
        if ok:
            return Command(
                goto="policy_learning",
                update={"supervisor_next": "policy"},
            )
        print(f"[supervisor] Blocked early conclusion: {gate_reason}")
        return Command(
            goto="instrument_subgraph",
            update={
                "supervisor_next": "instrument",
                "measurement_critique_decision": "more_data_needed",
                "supervisor_reasoning": gate_reason[:400],
                "ic_cycle": 0,
                "instrument_conclude": False,
            },
        )

    # If hypothesis not approved, go to hypothesis generation
    if not state.get("user_approved"):
        return Command(
            goto="hypothesis_gen",
            update={"supervisor_next": "hypothesis"},
        )

    # Check iteration limits
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", cfg.experiment.max_iterations)
    if iteration >= max_iterations:
        return Command(
            goto="policy_learning",
            update={"supervisor_next": "conclude"},
        )

    # Determine next worker based on state
    measurement_decision = state.get("measurement_critique_decision", "")
    recommended_worker = state.get("measurement_critique_recommended_worker", "")
    
    # Follow critic recommendation if available
    if recommended_worker in ("instrument", "analysis", "hypothesis", "literature"):
        worker = recommended_worker
        reasoning = f"Following measurement critic recommendation: {worker}"
    elif measurement_decision == "more_data_needed":
        worker = "instrument"
        reasoning = "More data needed - continuing instrument cycle"
    elif measurement_decision == "hypothesis_invalid":
        worker = "hypothesis"
        reasoning = "Hypothesis invalid - need refinement"
    elif measurement_decision == "inconclusive":
        worker = "instrument"
        reasoning = "Results inconclusive - collecting more data"
    elif state.get("literature_refresh_requested"):
        worker = "literature"
        reasoning = "Literature refresh requested"
    elif state.get("hypothesis_refinement_requested"):
        worker = "hypothesis"
        reasoning = "Hypothesis refinement requested"
    else:
        # Default: continue with instrument
        worker = "instrument"
        reasoning = "Default: continue instrument cycle"

    # Map worker to graph node
    worker_map = {
        "instrument": "instrument_subgraph",
        "analysis": "image_analysis_agent",
        "hypothesis": "hypothesis_gen",
        "literature": "literature",
        "policy": "policy_learning",
        "conclude": "policy_learning",
    }
    goto = worker_map.get(worker, "instrument_subgraph")

    # Reset instrument state if going to instrument
    updates: dict[str, Any] = {
        "supervisor_next": worker,
        "supervisor_reasoning": reasoning,
        "current_stage": "supervisor",
    }
    
    if goto == "instrument_subgraph":
        updates.update({
            "ic_cycle": 0,
            "instrument_conclude": False,
            "stall_count": 0,
            "last_decision_type": "",
            "measurement_critique_decision": "more_data_needed",
        })

    return Command(goto=goto, update=updates)
