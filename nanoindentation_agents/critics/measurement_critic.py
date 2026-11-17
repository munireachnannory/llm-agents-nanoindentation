"""
measurement_critic.py
=====================
Measurement Critic for evaluating indentation data.

Evaluates measurement results to determine:
- Data quality
- Hypothesis support/refutation
- Need for additional measurements
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage

from ..config import get_config
from ..state import AgentState


CritiqueDecision = Literal["more_data_needed", "hypothesis_invalid", "hypothesis_confirmed", "inconclusive"]


def _evidence_summary(state: AgentState) -> dict[str, Any]:
    """Summarize collected evidence."""
    indent_history = state.get("indent_history", [])
    
    return {
        "indent_arrays": len(indent_history),
        "total_indents": sum(i.get("valid_indent_count", 0) for i in indent_history),
        "images_captured": len(state.get("image_history", [])),
        "visited_locations": len(state.get("visited_target_keys", [])),
        "instrument_actions": len(state.get("decision_history", [])),
    }


def _can_confirm_hypothesis(state: AgentState, cfg=None) -> tuple[bool, str]:
    """Check if evidence is sufficient to confirm hypothesis."""
    cfg = cfg or get_config()
    ev = _evidence_summary(state)

    # objective-derived criteria (set by the orchestrator) override config defaults
    crit = state.get("objective_criteria") or {}
    min_arrays = int(crit.get("min_indent_arrays", cfg.experiment.min_indent_arrays_for_confirmation))
    min_locations = int(crit.get("min_visited_locations", cfg.experiment.min_visited_locations_for_confirmation))

    if ev["indent_arrays"] < min_arrays:
        return False, f"Need {min_arrays} indent arrays, have {ev['indent_arrays']}"
    if ev["visited_locations"] < min_locations:
        return False, f"Need {min_locations} locations, visited {ev['visited_locations']}"

    return True, "Evidence requirements met"


def _build_critique_prompt(state: AgentState) -> str:
    """Build the prompt for measurement critique."""
    parts = []

    # Hypothesis
    hypothesis = state.get("hypothesis", "")
    parts.append(f"## Hypothesis Under Test\n{hypothesis[:500]}")

    # Research objective (orchestrator mode)
    crit = state.get("objective_criteria") or {}
    if crit.get("objective_summary"):
        parts.append(f"## Research Objective\n{crit['objective_summary'][:400]}")
    
    # Success criteria
    hypothesis_data = state.get("hypothesis_data", {})
    success_criteria = hypothesis_data.get("success_criteria", "")
    if success_criteria:
        parts.append(f"## Success Criteria\n{success_criteria}")
    
    failure_criteria = hypothesis_data.get("failure_criteria", "")
    if failure_criteria:
        parts.append(f"## Failure Criteria\n{failure_criteria}")
    
    # Indentation results
    indent_history = state.get("indent_history", [])
    if indent_history:
        parts.append("## Indentation Results")
        for i, indent in enumerate(indent_history[-5:]):
            parts.append(
                f"- Array {i+1}: H={indent.get('hardness_mean', 0):.2f}±{indent.get('hardness_std', 0):.2f} GPa, "
                f"E={indent.get('modulus_mean', 0):.2f}±{indent.get('modulus_std', 0):.2f} GPa, "
                f"n={indent.get('valid_indent_count', 0)} valid"
            )
    
    # Evidence summary
    ev = _evidence_summary(state)
    parts.append(
        f"## Evidence Summary\n"
        f"- Indent arrays: {ev['indent_arrays']}\n"
        f"- Total indents: {ev['total_indents']}\n"
        f"- Visited locations: {ev['visited_locations']}\n"
        f"- Images: {ev['images_captured']}"
    )
    
    # Image analysis
    image_analysis = state.get("image_analysis", {})
    if image_analysis:
        interpretation = image_analysis.get("interpretation", "")
        if interpretation:
            parts.append(f"## Image Analysis\n{interpretation[:300]}")
    
    return "\n\n".join(parts)


def _analyze_data_quality(state: AgentState) -> dict[str, Any]:
    """Analyze the quality of collected indentation data."""
    indent_history = state.get("indent_history", [])
    
    if not indent_history:
        return {
            "quality": "insufficient",
            "issues": ["No indentation data collected"],
            "valid_fraction": 0.0,
        }
    
    total_indents = sum(i.get("n_indents", 25) for i in indent_history)
    valid_indents = sum(i.get("valid_indent_count", 0) for i in indent_history)
    valid_fraction = valid_indents / total_indents if total_indents > 0 else 0.0
    
    issues = []
    if valid_fraction < 0.8:
        issues.append(f"Low valid indent fraction: {valid_fraction:.1%}")
    
    # Check for reasonable property ranges
    for indent in indent_history:
        h = indent.get("hardness_mean", 0)
        e = indent.get("modulus_mean", 0)
        if h <= 0 or h > 100:
            issues.append(f"Unusual hardness value: {h} GPa")
        if e <= 0 or e > 1000:
            issues.append(f"Unusual modulus value: {e} GPa")
    
    quality = "good" if not issues else ("acceptable" if len(issues) < 2 else "poor")
    
    return {
        "quality": quality,
        "issues": issues,
        "valid_fraction": valid_fraction,
        "total_indents": total_indents,
        "valid_indents": valid_indents,
    }


def _objective_array_scores(
    indent_history: list[dict[str, Any]],
    crit: dict[str, Any],
) -> list[float]:
    """
    Per-array signed improvement over the sample mean, weighted across the
    objective's properties and respecting each property's direction.
    Positive = better under the objective. Pure-python arithmetic.
    """
    weights = dict(crit.get("property_weights") or {})
    directions = dict(crit.get("directions") or {})
    total_w = sum(weights.values()) or 1.0
    prop_keys = {"hardness": "hardness_mean", "modulus": "modulus_mean"}

    means: dict[str, float] = {}
    values: dict[str, list[float]] = {}
    for name, key in prop_keys.items():
        if name not in weights:
            continue
        vals = [float(d.get(key) or 0.0) for d in indent_history]
        values[name] = vals
        means[name] = (sum(vals) / len(vals)) if vals else 0.0

    scores: list[float] = []
    for i in range(len(indent_history)):
        score = 0.0
        for name in values:
            mean = means[name]
            if mean == 0.0:
                continue
            rel = (values[name][i] - mean) / abs(mean)
            if directions.get(name) == "minimize":
                rel = -rel
            score += (weights[name] / total_w) * rel
        scores.append(score)
    return scores


def _evaluate_objective_criteria(
    state: AgentState,
    crit: dict[str, Any],
) -> tuple[str, str]:
    """Objective-driven evaluation (orchestrator mode): best array vs baseline."""
    indent_history = state.get("indent_history", [])
    threshold = float(crit.get("min_improvement_pct", 10.0)) / 100.0
    min_arrays = int(crit.get("min_indent_arrays", 2))

    scores = _objective_array_scores(indent_history, crit)
    if not scores:
        return "more_data_needed", "No usable array statistics for objective evaluation."

    best = max(scores)
    best_idx = scores.index(best)
    best_label = indent_history[best_idx].get("area_name", f"array {best_idx + 1}")

    if best >= threshold:
        return "hypothesis_confirmed", (
            f"[OBJECTIVE CRITERIA] Best array '{best_label}' beats the sample baseline "
            f"by {best:.1%} >= required {threshold:.0%} on the weighted objective."
        )
    # well past the evidence minimum and still flat -> objective landscape looks uniform
    if len(indent_history) >= min_arrays + 2 and best < threshold / 2.0:
        return "hypothesis_invalid", (
            f"[OBJECTIVE CRITERIA] After {len(indent_history)} arrays the best improvement "
            f"is only {best:.1%} (< half the required {threshold:.0%}). The material appears "
            f"uniform relative to the objective."
        )
    return "more_data_needed", (
        f"[OBJECTIVE CRITERIA] Best improvement so far {best:.1%} < required {threshold:.0%} "
        f"— objective margin not yet reached."
    )


def _evaluate_hypothesis(state: AgentState) -> tuple[str, str]:
    """
    Evaluate hypothesis against collected data.

    Returns:
        (decision, rationale)
    """
    cfg = get_config()
    indent_history = state.get("indent_history", [])

    if not indent_history:
        return "more_data_needed", "No indentation data collected yet."

    # Check evidence requirements
    can_confirm, gate_reason = _can_confirm_hypothesis(state, cfg)
    if not can_confirm:
        return "more_data_needed", gate_reason

    # Objective-driven evaluation when the orchestrator supplied criteria
    crit = state.get("objective_criteria") or {}
    if crit.get("property_weights"):
        return _evaluate_objective_criteria(state, crit)

    # Analyze spatial variation
    if len(indent_history) >= 2:
        hardness_values = [i.get("hardness_mean", 0) for i in indent_history if i.get("hardness_mean")]
        modulus_values = [i.get("modulus_mean", 0) for i in indent_history if i.get("modulus_mean")]
        
        if hardness_values and len(hardness_values) >= 2:
            h_range = max(hardness_values) - min(hardness_values)
            h_mean = sum(hardness_values) / len(hardness_values)
            h_variation = h_range / h_mean if h_mean > 0 else 0
            
            if h_variation > 0.1:  # >10% variation
                return "hypothesis_confirmed", (
                    f"Significant hardness variation detected: {h_variation:.1%}. "
                    f"Range: {min(hardness_values):.2f} - {max(hardness_values):.2f} GPa."
                )
            else:
                return "hypothesis_invalid", (
                    f"Hardness variation is only {h_variation:.1%}, below expected threshold. "
                    f"Material appears relatively uniform."
                )
    
    return "inconclusive", (
        "Insufficient data to draw conclusions. "
        "Need more measurements at different locations."
    )


def measurement_critic_node(state: AgentState) -> dict[str, Any]:
    """
    Evaluate indentation data against the hypothesis.
    
    Determines whether the hypothesis is supported, refuted,
    or if more data is needed.
    """
    cfg = get_config()
    
    # Check for any data
    indent_history = state.get("indent_history", [])
    image_history = state.get("image_history", [])
    
    if not indent_history and not image_history:
        return {
            "measurement_critique": "No measurement data available to evaluate.",
            "measurement_critique_decision": "more_data_needed",
            "measurement_critique_recommended_worker": "instrument",
            "current_stage": "measurement_critic",
            "should_continue": True,
        }
    
    # Analyze data quality
    quality = _analyze_data_quality(state)
    
    # Evaluate hypothesis
    decision, rationale = _evaluate_hypothesis(state)
    
    # Check evidence gate for confirmation
    if decision == "hypothesis_confirmed":
        can_confirm, gate_reason = _can_confirm_hypothesis(state, cfg)
        if not can_confirm:
            decision = "more_data_needed"
            rationale = f"{rationale}\n\n[EVIDENCE GATE] {gate_reason}"
    
    # Build critique text
    critique_text = f"""
Decision: {decision.upper().replace('_', ' ')}

Rationale:
{rationale}

Data Quality:
- Quality: {quality['quality']}
- Valid indents: {quality.get('valid_indents', 0)}/{quality.get('total_indents', 0)}
"""
    
    if quality.get("issues"):
        critique_text += f"- Issues: {'; '.join(quality['issues'])}\n"
    
    # Determine recommended worker
    if decision == "more_data_needed":
        recommended_worker = "instrument"
        recommendations = "Continue collecting indentation data at additional locations."
    elif decision == "hypothesis_invalid":
        recommended_worker = "hypothesis_gen"
        recommendations = "Revise hypothesis based on uniform property distribution."
    elif decision == "hypothesis_confirmed":
        recommended_worker = "policy"
        recommendations = "Hypothesis validated. Proceed to conclusion."
    else:
        recommended_worker = "instrument"
        recommendations = "Collect additional data for more conclusive evaluation."
    
    critique_text += f"\nRecommendations:\n{recommendations}"
    
    # Determine if we should continue
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    should_continue = decision == "more_data_needed" and iteration < max_iterations - 1
    
    # Update messages
    prompt = _build_critique_prompt(state)
    messages = list(state.get("messages", []))
    messages.extend([
        HumanMessage(content=prompt),
        AIMessage(content=critique_text),
    ])
    
    return {
        "measurement_critique": critique_text.strip(),
        "measurement_critique_decision": decision,
        "measurement_critique_recommended_worker": recommended_worker,
        "current_stage": "measurement_critic",
        "should_continue": should_continue,
        "iteration": iteration + 1,
        "messages": messages,
    }


def get_routing_decision(state: AgentState) -> str:
    """
    Get the routing decision based on measurement critique.
    
    Used for conditional edges in the LangGraph.
    """
    decision = state.get("measurement_critique_decision", "")
    should_continue = state.get("should_continue", False)

    if decision == "hypothesis_confirmed":
        cfg = get_config()
        can_confirm, _ = _can_confirm_hypothesis(state, cfg)
        if can_confirm:
            return "conclude"
        return "more_data_needed"
    elif decision == "hypothesis_invalid":
        return "hypothesis_invalid"
    elif should_continue:
        return "more_data_needed"
    else:
        return "conclude"


def format_critique_for_display(state: AgentState) -> str:
    """Format the measurement critique for user display."""
    critique = state.get("measurement_critique", "")
    decision = state.get("measurement_critique_decision", "")
    
    decision_icons = {
        "hypothesis_confirmed": "CONFIRMED",
        "hypothesis_invalid": "INVALID",
        "more_data_needed": "MORE DATA NEEDED",
        "inconclusive": "INCONCLUSIVE",
    }
    
    lines = [
        "=" * 60,
        "MEASUREMENT CRITIQUE",
        "=" * 60,
        "",
        f"Decision: {decision_icons.get(decision, decision.upper())}",
        "",
        critique if critique else "No critique available.",
        "",
        "-" * 60,
    ]
    
    return "\n".join(lines)
