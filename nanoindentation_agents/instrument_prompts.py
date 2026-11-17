"""
instrument_prompts.py
=======================
Build context blocks for instrument planner prompts from AgentState.
"""

from __future__ import annotations

import json
from typing import Any

from .state import AgentState


def _json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def experiment_context_block(state: AgentState) -> str:
    return (
        state.get("experiment_context")
        or state.get("research_question")
        or "(none)"
    )


def hypothesis_block(state: AgentState) -> str:
    return state.get("hypothesis", "") or "(none)"


def success_criteria_block(state: AgentState) -> str:
    data = state.get("hypothesis_data") or {}
    return data.get("success_criteria") or state.get("_hypothesis_success_criteria", "") or "(none)"


def failure_criteria_block(state: AgentState) -> str:
    data = state.get("hypothesis_data") or {}
    return data.get("failure_criteria") or state.get("_hypothesis_failure_criteria", "") or "(none)"


def test_progress_text(state: AgentState) -> str:
    tp = state.get("test_progress") or {}
    if not tp:
        return "(no active test)"
    return _json_block(tp)


def position_text(state: AgentState) -> str:
    pos = state.get("current_position", (0.0, 0.0))
    return f"X={pos[0]:.2f}, Y={pos[1]:.2f}"


def targets_text(state: AgentState) -> str:
    ia = state.get("image_analysis") or {}
    targets = ia.get("target_locations") or []
    if not targets:
        return "(none from image analysis)"
    lines = []
    for t in targets[:8]:
        lines.append(
            f"- {t.get('label', '?')}: ({t.get('x', 0):.1f}, {t.get('y', 0):.1f}) "
            f"reason={t.get('reason', '')} priority={t.get('priority', '')}"
        )
    return "\n".join(lines)


def visited_targets_text(state: AgentState) -> str:
    visited = state.get("visited_target_keys") or []
    return ", ".join(visited) if visited else "(none)"


def indent_history_text(state: AgentState) -> str:
    history = state.get("indent_history") or []
    if not history:
        return "(none yet)"
    lines = []
    for i, indent in enumerate(history[-5:], start=1):
        lines.append(
            f"- {indent.get('test_name', f'array_{i}')}: "
            f"H={indent.get('hardness_mean', 'N/A')} GPa, "
            f"E={indent.get('modulus_mean', 'N/A')} GPa, "
            f"pos={indent.get('position', 'N/A')}"
        )
    return "\n".join(lines)


def decision_history_text(state: AgentState) -> str:
    history = state.get("decision_history") or []
    if not history:
        return "(none)"
    lines = []
    for h in history[-8:]:
        lines.append(
            f"- cycle {h.get('cycle')}: {h.get('tool_name')} "
            f"({h.get('decision_type', '')}) — {str(h.get('result_summary', ''))[:80]}"
        )
    return "\n".join(lines)


def operator_feedback_text(state: AgentState) -> str:
    fb = state.get("operator_feedback") or []
    return "\n".join(f"- {n}" for n in fb[-5:]) if fb else "(none)"


def denied_action_text(state: AgentState) -> str:
    denied = state.get("last_denied_action") or {}
    if not denied.get("tool_name"):
        return "(none)"
    return _json_block(denied)


def build_instrument_user_prompt(
    state: AgentState,
    *,
    tool_catalog: str,
    workflows_text: str,
) -> str:
    from instrument_control.prompt_tools import format_prompt

    pos = state.get("current_position", (0.0, 0.0))
    alignment = state.get("alignment_data") or {}
    return format_prompt(
        "instrument_user",
        experiment_context=experiment_context_block(state),
        hypothesis=hypothesis_block(state),
        success_criteria=success_criteria_block(state),
        failure_criteria=failure_criteria_block(state),
        collection_phase=state.get("collection_phase", "exploring"),
        test_progress_text=test_progress_text(state),
        position_text=position_text(state),
        extension=state.get("current_extension", 0.0),
        alignment_active=bool(alignment.get("area_name")),
        plane_params=state.get("plane_params") or [],
        targets_text=targets_text(state),
        visited_targets=visited_targets_text(state),
        indent_history_text=indent_history_text(state),
        decision_history_text=decision_history_text(state),
        operator_feedback=operator_feedback_text(state),
        denied_action_text=denied_action_text(state),
        tool_catalog=tool_catalog,
        workflows_text=workflows_text,
    )
