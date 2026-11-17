"""
policy_learning_agent.py
========================
Policy Learning Agent for capturing experiment insights.

Records successful patterns and learned policies from
completed experiments.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def policy_learning_node(state: AgentState) -> dict[str, Any]:
    """
    Extract and record learned policies from the experiment.
    
    Analyzes the decision history to identify successful patterns
    that can inform future experiments.
    """
    decision_history = state.get("decision_history", [])
    indent_history = state.get("indent_history", [])
    
    # Analyze decision patterns
    decision_counts = {}
    for d in decision_history:
        dt = d.get("decision_type", "unknown")
        decision_counts[dt] = decision_counts.get(dt, 0) + 1
    
    # Identify successful patterns
    successful_patterns = []
    
    # Check if hypothesis was confirmed
    if state.get("measurement_critique_decision") == "hypothesis_confirmed":
        successful_patterns.append("hypothesis_validation_success")
    
    # Check data collection efficiency
    if indent_history:
        avg_valid = sum(i.get("valid_indent_count", 0) for i in indent_history) / len(indent_history)
        if avg_valid > 20:
            successful_patterns.append("high_valid_indent_rate")
    
    # Build policy
    policy = {
        "experiment_name": state.get("experiment_name", ""),
        "total_iterations": state.get("iteration", 0),
        "decision_counts": decision_counts,
        "successful_patterns": successful_patterns,
        "indent_arrays_collected": len(indent_history),
        "locations_visited": len(state.get("visited_target_keys", [])),
        "final_decision": state.get("measurement_critique_decision", ""),
        "recommendations": [],
    }
    
    # Generate recommendations
    if decision_counts.get("capture_image", 0) < 2:
        policy["recommendations"].append("Consider more frequent imaging for better targeting")
    
    if len(indent_history) < 3:
        policy["recommendations"].append("Collect more indent arrays for statistical significance")
    
    return {
        "policy": policy,
        "current_stage": "policy_learning",
        "scratchpad": [f"policy: learned from {len(decision_history)} decisions"],
    }
