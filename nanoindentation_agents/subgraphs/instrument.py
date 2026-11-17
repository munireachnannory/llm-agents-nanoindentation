"""
subgraphs/instrument.py
=======================
Instrument control subgraph: plan -> approve -> execute -> interpret -> spatial update.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from instrument_control.prompt_tools import format_prompt
from instrument_control.schemas import ProposedAction, ToolOutcome, safety

from ..agents.plan_validator import validate_instrument_plan
from ..config import get_config
from ..constants import DECISION_TYPES
from ..instrument_prompts import build_instrument_user_prompt
from ..registry.models import get_llm_for_role
from ..registry.tool_catalog import (
    format_tool_catalog_text,
    format_workflows_text,
    merge_toolkit_catalog,
    parse_llm_json,
)
from ..state import AgentState
from ..tools.indent_tools import get_toolkit


def _state_summary(state: AgentState, max_items: int = 8) -> str:
    """Generate a text summary of decision history."""
    history = state.get("decision_history", [])
    if not history:
        return "(none)"
    lines = []
    for h in history[-max_items:]:
        lines.append(
            f"- {h.get('decision_type', '?')}/{h.get('measurement_mode', '?')}: "
            f"{h.get('tool_name', 'N/A')} — {h.get('reasoning', '')[:120]}"
        )
    return "\n".join(lines)


def _plan_update_from_action(
    state: AgentState,
    *,
    proposed: ProposedAction,
    decision_type: str,
    measurement_mode: str = "indentation",
    risk_level: str = "low",
    source: str = "rule",
) -> dict[str, Any]:
    ic_cycle = state.get("ic_cycle", 0)
    return {
        "instrument_approved": False,
        "last_denied_action": {},
        "pending_proposed_action": {
            "tool_name": proposed.tool_name,
            "arguments": proposed.arguments,
            "justification": proposed.justification,
            "risk_level": risk_level,
        },
        "orchestrator_plan": {
            "tool_name": proposed.tool_name,
            "parameters": proposed.arguments,
            "justification": proposed.justification,
            "risk_level": risk_level,
            "action_type": "tool",
            "decision_type": decision_type,
            "measurement_mode": measurement_mode,
        },
        "current_stage": "instrument_plan",
        "ic_cycle": ic_cycle + 1,
        "scratchpad": [f"planned ({source}): {proposed.tool_name} — {proposed.justification[:120]}"],
    }


def _conclude_update(reasoning: str) -> dict[str, Any]:
    return {
        "instrument_conclude": True,
        "current_stage": "instrument_plan",
        "scratchpad": [f"instrument_plan: conclude — {reasoning}"],
    }


def _rule_based_plan(state: AgentState, cfg) -> dict[str, Any] | None:
    """Deterministic fallback planner when LLM is unavailable or invalid."""
    indent_history = state.get("indent_history", [])
    image_analysis = state.get("image_analysis", {})
    targets = image_analysis.get("target_locations", [])
    visited = state.get("visited_target_keys", [])
    test_progress = state.get("test_progress") or {}

    if test_progress.get("status") == "configured":
        proposed = ProposedAction(
            tool_name="start_test_normal",
            arguments={"t": 2},
            justification="Test configured — start indentation array",
            safety=safety("medium"),
        )
        return _plan_update_from_action(
            state,
            proposed=proposed,
            decision_type="run_test",
            measurement_mode="indentation",
            risk_level="medium",
            source="rule",
        )

    if test_progress.get("status") == "running":
        proposed = ProposedAction(
            tool_name="get_current_sample_and_test_ocr_passive",
            arguments={"debug": False},
            justification="Poll indent progress until array completes",
            safety=safety("read_only"),
        )
        return _plan_update_from_action(
            state,
            proposed=proposed,
            decision_type="monitor",
            measurement_mode="indentation",
            risk_level="read_only",
            source="rule",
        )

    if not state.get("current_image"):
        proposed = ProposedAction(
            tool_name="capture_video_panel_remote",
            arguments={},
            justification="Capture microscope image for targeting and verification",
            safety=safety("low"),
        )
        return _plan_update_from_action(
            state,
            proposed=proposed,
            decision_type="capture_image",
            measurement_mode="imaging",
            source="rule",
        )

    if targets and len(visited) < len(targets):
        unvisited = [t for t in targets if t.get("label") not in visited]
        if unvisited:
            target = unvisited[0]
            if state.get("alignment_data"):
                tool_name = "move_to_aligned_point"
                arguments = {
                    "area_name": state.get("alignment_data", {}).get("area_name", "Area1"),
                    "x": target.get("x", 0),
                    "y": target.get("y", 0),
                    "plane_params": state.get("plane_params", []),
                }
            else:
                tool_name = "move"
                arguments = {
                    "amount": target.get("distance", 50),
                    "direction": target.get("direction", "right"),
                    "t": 2,
                }
            proposed = ProposedAction(
                tool_name=tool_name,
                arguments=arguments,
                justification=f"Move to target {target.get('label', 'unknown')}",
                safety=safety("low"),
            )
            return _plan_update_from_action(
                state,
                proposed=proposed,
                decision_type="move_to_location",
                measurement_mode="indentation",
                source="rule",
            )

    if len(indent_history) < cfg.experiment.min_indent_arrays_for_confirmation:
        defaults = cfg.indent_defaults
        sample_name = f"test_{len(indent_history) + 1}_{int(time.time())}"
        proposed = ProposedAction(
            tool_name="starting_tests",
            arguments={
                "sample_name": sample_name,
                "setup": dict(defaults.setup),
                "indents": {
                    "X Indents": defaults.grid_nx,
                    "Y Indents": defaults.grid_ny,
                    "X Spacing": defaults.spacing_um,
                    "Y Spacing": defaults.spacing_um,
                    "Rotation": defaults.rotation,
                },
                "remove_n": 1,
                "method_type": defaults.method_type,
                "use_import": 0,
                "t": 2,
            },
            justification="Configure indentation array at current location",
            safety=safety("medium"),
        )
        return _plan_update_from_action(
            state,
            proposed=proposed,
            decision_type="setup_test",
            measurement_mode="indentation",
            risk_level="medium",
            source="rule",
        )

    return None


def _llm_plan(state: AgentState, cfg) -> dict[str, Any] | None:
    """LLM planner using tool catalog and workflow descriptions."""
    try:
        toolkit = get_toolkit()
        live_tools = merge_toolkit_catalog(toolkit.list_available_tools())
    except Exception:
        live_tools = []

    catalog_text = format_tool_catalog_text()
    workflows_text = format_workflows_text()
    system = format_prompt("instrument_system")
    user = build_instrument_user_prompt(
        state,
        tool_catalog=catalog_text,
        workflows_text=workflows_text,
    )

    try:
        llm = get_llm_for_role("instrument", cfg)
        response = llm.chat(user_message=user, system_message=system)
        decision = parse_llm_json(response)
    except Exception as exc:
        print(f"[instrument_plan] LLM unavailable ({exc}) — using rule fallback")
        return None

    validation = validate_instrument_plan(decision, state=state)
    if validation.warnings:
        for w in validation.warnings:
            print(f"[instrument_plan] warning: {w}")

    if not validation.ok:
        print(f"[instrument_plan] blocked: {validation.blocked_by}")
        return None

    if validation.decision_type == "conclude":
        reasoning = str(decision.get("reasoning") or decision.get("justification") or "LLM concluded cycle")
        return _conclude_update(reasoning)

    if not validation.action:
        return None

    risk = validation.action.safety.risk_level
    update = _plan_update_from_action(
        state,
        proposed=validation.action,
        decision_type=validation.decision_type,
        measurement_mode=validation.measurement_mode,
        risk_level=risk,
        source="llm",
    )
    update["instrument_tool_catalog"] = live_tools
    if validation.action.tool_name == "starting_tests":
        update["test_progress"] = {
            "test_name": validation.action.arguments.get("sample_name"),
            "status": "pending_start",
        }
    return update


def plan_next_action_node(state: AgentState) -> dict[str, Any]:
    """Plan the next instrument action using LLM + catalog, with rule fallback."""
    cfg = get_config()

    if state.get("instrument_conclude"):
        return {
            "current_stage": "instrument_plan",
            "orchestrator_plan": {"action_type": "conclude"},
        }

    ic_cycle = state.get("ic_cycle", 0)
    if ic_cycle >= state.get("ic_max_cycles", cfg.experiment.ic_max_cycles):
        return _conclude_update("Maximum instrument cycles reached")

    plan = _llm_plan(state, cfg)
    if plan is None:
        plan = _rule_based_plan(state, cfg)
    if plan is None:
        return _conclude_update("Sufficient data collected")

    if plan.get("instrument_conclude"):
        return plan

    return plan


def instrument_approve_node(state: AgentState) -> dict[str, Any]:
    """Human approval gate for instrument actions."""
    cfg = get_config()
    pending = state.get("pending_proposed_action") or {}

    if state.get("instrument_approved"):
        return {
            "current_stage": "instrument_approve",
            "instrument_approved": True,
            "last_denied_action": {},
        }

    if (state.get("last_denied_action") or {}).get("tool_name"):
        return {
            "instrument_approved": False,
            "instrument_conclude": False,
            "current_stage": "instrument_approve",
        }

    if not pending.get("tool_name"):
        return {"current_stage": "instrument_approve"}

    if cfg.graph.auto_approve_instruments:
        return {
            "current_stage": "instrument_approve",
            "instrument_approved": True,
            "last_denied_action": {},
        }

    from instrument_control.approval import request_approval

    proposed = ProposedAction(
        tool_name=pending["tool_name"],
        arguments=pending.get("arguments", {}),
        justification=pending.get("justification", ""),
        safety=safety(pending.get("risk_level", "low")),
    )
    decision = request_approval(proposed, run_mode="human_suggests_alternatives")

    if decision.decision != "approved":
        notes = (decision.human_notes or decision.rationale or "").strip()
        feedback = list(state.get("operator_feedback", []))
        if notes:
            feedback.append(f"{pending.get('tool_name')}: {notes}")
        if any(w in notes.lower() for w in ("stop", "conclude", "abort", "end experiment", "quit")):
            return {
                "instrument_approved": False,
                "instrument_conclude": True,
                "operator_feedback": feedback,
                "current_stage": "instrument_approve",
                "scratchpad": [f"instrument STOP: {notes[:200]}"],
            }
        return {
            "instrument_approved": False,
            "instrument_conclude": False,
            "operator_feedback": feedback,
            "last_denied_action": {
                "tool_name": pending.get("tool_name"),
                "arguments": pending.get("arguments", {}),
                "human_notes": notes,
                "rationale": decision.rationale,
            },
            "pending_proposed_action": {},
            "current_stage": "instrument_approve",
            "scratchpad": [f"instrument DENIED — replanning: {notes[:200]}"],
        }

    return {
        "instrument_approved": True,
        "last_denied_action": {},
        "current_stage": "instrument_approve",
    }


def execute_tool_node(state: AgentState) -> dict[str, Any]:
    """Execute the approved instrument tool."""
    cfg = get_config()
    pending = state.get("pending_proposed_action") or {}
    
    if not pending.get("tool_name"):
        return {
            "error_message": "No pending tool to execute",
            "current_stage": "instrument_execute",
        }

    if not state.get("instrument_approved") and not cfg.graph.auto_approve_instruments:
        return {
            "error_message": "Instrument action not approved",
            "current_stage": "instrument_execute",
            "last_tool_result": {"ok": False, "error": "not approved"},
        }

    toolkit = get_toolkit()
    tool_name = pending["tool_name"]
    arguments = pending.get("arguments", {})
    
    outcome = toolkit.call_tool(tool_name, arguments)
    
    result: dict[str, Any] = {"ok": outcome.ok}
    if outcome.ok:
        result = outcome.result if isinstance(outcome.result, dict) else {"result": outcome.result}
        result["ok"] = True
    else:
        result = {"ok": False, "error": outcome.error}

    # Build decision record
    plan = state.get("orchestrator_plan") or {}
    decision_record = {
        "cycle": state.get("ic_cycle", len(state.get("decision_history", [])) + 1),
        "tool_name": tool_name,
        "arguments": arguments,
        "approved": True,
        "reasoning": pending.get("justification", ""),
        "decision_type": plan.get("decision_type", ""),
        "result_summary": str(result.get("message", "ok" if outcome.ok else outcome.error))[:120],
    }

    updates: dict[str, Any] = {
        "last_tool_result": result,
        "decision_history": [decision_record],
        "current_stage": "instrument_execute",
        "error_message": None if outcome.ok else outcome.error,
    }

    # Handle specific tool results
    if outcome.ok:
        if tool_name == "capture_video_panel_remote":
            updates["current_image"] = {
                "image_path": result.get("path", ""),
                "timestamp": str(time.time()),
                "position": state.get("current_position", (0.0, 0.0)),
            }
            updates["image_history"] = [updates["current_image"]]

        elif tool_name == "get_xyz_positions":
            updates["current_position"] = (
                result.get("x", 0.0),
                result.get("y", 0.0),
            )
            updates["current_extension"] = result.get("extension", 0.0)

        elif tool_name in ("move", "move_in_increments", "move_to_aligned_point"):
            # Update position after movement
            pass  # Position will be updated on next get_xyz_positions

        elif tool_name == "starting_tests":
            indents = arguments.get("indents") or {}
            nx = indents.get("X Indents", 5)
            ny = indents.get("Y Indents", 5)
            updates["test_progress"] = {
                "test_name": arguments.get("sample_name"),
                "expected_indents": nx * ny,
                "status": "configured",
            }

        elif tool_name == "start_test_normal":
            test_progress = dict(state.get("test_progress") or {})
            test_progress["status"] = "running"
            updates["test_progress"] = test_progress

        elif tool_name == "process_review_file":
            # Store processed results
            if result.get("export_path"):
                current_indent = {
                    "test_name": arguments.get("current_sample_file_name", ""),
                    "export_path": result.get("export_path"),
                    "timestamp": str(time.time()),
                    "position": state.get("current_position", (0.0, 0.0)),
                }
                updates["current_indent"] = current_indent
                updates["indent_history"] = [current_indent]

    return updates


def interpret_result_node(state: AgentState) -> dict[str, Any]:
    """Interpret the tool execution result."""
    pending = state.get("pending_proposed_action") or {}
    result = state.get("last_tool_result") or {}
    
    interpretation = {
        "tool": pending.get("tool_name", "unknown"),
        "success": result.get("ok", False),
        "recommendation": "continue" if result.get("ok") else "retry",
    }

    if not result.get("ok"):
        interpretation["error_analysis"] = result.get("error", "Unknown error")
        interpretation["recommendation"] = "conclude"

    updates: dict[str, Any] = {
        "last_instrument_interpretation": interpretation,
        "current_stage": "instrument_interpret",
    }

    # Check if we should conclude
    cfg = get_config()
    indent_count = len(state.get("indent_history", []))
    if indent_count >= cfg.experiment.min_indent_arrays_for_confirmation:
        updates["instrument_conclude"] = True

    return updates


def update_spatial_node(state: AgentState) -> dict[str, Any]:
    """Update spatial targeting based on results."""
    # Mark current target as visited
    current_target = state.get("current_target") or {}
    visited = list(state.get("visited_target_keys") or [])
    
    if current_target.get("label") and current_target["label"] not in visited:
        visited.append(current_target["label"])

    return {
        "visited_target_keys": visited,
        "current_stage": "instrument_spatial",
        "scratchpad": [f"spatial: visited {len(visited)} targets"],
    }


def route_after_approve(state: AgentState) -> Literal["execute", "plan", "instrument_done"]:
    """Route after approval node."""
    if state.get("instrument_approved"):
        return "execute"
    if (state.get("last_denied_action") or {}).get("tool_name"):
        return "plan"
    if state.get("instrument_conclude"):
        return "instrument_done"
    return "instrument_done"


def route_after_plan(state: AgentState) -> Literal["instrument_approve", "instrument_done"]:
    """Route after planning node."""
    if state.get("instrument_conclude"):
        return "instrument_done"
    if state.get("pending_proposed_action", {}).get("tool_name"):
        return "instrument_approve"
    return "instrument_done"


def route_after_execute(state: AgentState) -> Literal["interpret", "instrument_done"]:
    """Route after execution node."""
    if state.get("last_tool_result", {}).get("ok") is False:
        return "instrument_done"
    return "interpret"


def route_after_spatial(state: AgentState) -> Literal["plan", "instrument_done"]:
    """Route after spatial update."""
    cfg = get_config()
    
    if state.get("instrument_conclude"):
        return "instrument_done"
    
    ic_cycle = state.get("ic_cycle", 0)
    max_cycles = state.get("ic_max_cycles", cfg.experiment.ic_max_cycles)
    
    if ic_cycle >= max_cycles:
        return "instrument_done"
    
    return "plan"


def build_instrument_subgraph() -> CompiledStateGraph:
    """Build the instrument control subgraph."""
    g = StateGraph(AgentState)
    
    g.add_node("plan", plan_next_action_node)
    g.add_node("instrument_approve", instrument_approve_node)
    g.add_node("execute", execute_tool_node)
    g.add_node("interpret", interpret_result_node)
    g.add_node("spatial", update_spatial_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, {
        "instrument_approve": "instrument_approve",
        "instrument_done": END,
    })
    g.add_conditional_edges("instrument_approve", route_after_approve, {
        "execute": "execute",
        "plan": "plan",
        "instrument_done": END,
    })
    g.add_conditional_edges("execute", route_after_execute, {
        "interpret": "interpret",
        "instrument_done": END,
    })
    g.add_edge("interpret", "spatial")
    g.add_conditional_edges("spatial", route_after_spatial, {
        "plan": "plan",
        "instrument_done": END,
    })
    
    return g.compile()
