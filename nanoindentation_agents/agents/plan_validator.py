"""
plan_validator.py
=================
Validate instrument planner output against the tool catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instrument_control.schemas import ProposedAction, safety

from ..registry.tool_catalog import get_allowed_tool_names, load_instrument_catalog
from ..constants import DECISION_TYPES


@dataclass
class PlanValidation:
    ok: bool
    action: ProposedAction | None = None
    decision_type: str = ""
    measurement_mode: str = ""
    blocked_by: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_tool_name(name: str) -> str:
    if name.startswith("indent_"):
        return name[7:]
    return name


def validate_instrument_plan(
    decision: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> PlanValidation:
    """Validate LLM planner JSON before proposing for approval."""
    state = state or {}
    blocked: list[str] = []
    warnings: list[str] = []

    decision_type = str(decision.get("decision_type", "") or "").strip()
    if decision_type == "conclude" or decision.get("decision") == "conclude":
        return PlanValidation(ok=True, decision_type="conclude")

    tool_name = decision.get("tool_name")
    if not tool_name:
        blocked.append("conclude requested but decision_type is not conclude and tool_name is missing")
        return PlanValidation(ok=False, blocked_by=blocked)

    tool_name = _normalize_tool_name(str(tool_name))
    allowed = get_allowed_tool_names()
    if tool_name not in allowed:
        blocked.append(f"tool '{tool_name}' not in allowed list")
        return PlanValidation(ok=False, blocked_by=blocked)

    if decision_type and decision_type not in DECISION_TYPES:
        warnings.append(f"unknown decision_type '{decision_type}' — using tool mapping")

    arguments = decision.get("tool_arguments")
    if arguments is None:
        arguments = decision.get("arguments", {})
    if not isinstance(arguments, dict):
        blocked.append("tool_arguments must be a JSON object")
        return PlanValidation(ok=False, blocked_by=blocked)

    catalog = load_instrument_catalog()
    tool_meta = (catalog.get("tools") or {}).get(tool_name, {})
    params = tool_meta.get("parameters") or {}
    for pname, pspec in params.items():
        if pspec.get("required") and pname not in arguments:
            blocked.append(f"missing required parameter '{pname}' for {tool_name}")

    # Workflow guards from state
    test_progress = state.get("test_progress") or {}
    if tool_name == "start_test_normal" and not test_progress.get("test_name"):
        if not state.get("pending_test_configured"):
            warnings.append("start_test_normal before starting_tests — ensure array is configured")

    if tool_name == "process_review_file":
        status = test_progress.get("status", "")
        if status == "running":
            warnings.append("process_review_file while test may still be running")

    risk = str(decision.get("risk_level") or tool_meta.get("risk_level") or "low")
    measurement_mode = str(decision.get("measurement_mode") or _mode_for_tool(tool_name))

    action = ProposedAction(
        tool_name=tool_name,
        arguments=arguments,
        justification=str(decision.get("reasoning") or decision.get("justification") or ""),
        safety=safety(risk),
    )
    return PlanValidation(
        ok=True,
        action=action,
        decision_type=decision_type or _decision_type_for_tool(tool_name),
        measurement_mode=measurement_mode,
        warnings=warnings,
    )


def _decision_type_for_tool(tool_name: str) -> str:
    mapping = {
        "move": "move_to_location",
        "move_in_increments": "move_to_location",
        "move_to_aligned_point": "move_to_location",
        "starting_tests": "setup_test",
        "start_test_normal": "run_test",
        "capture_video_panel_remote": "capture_image",
        "process_review_file": "process_data",
        "align_focus": "align",
        "focus": "align",
        "restore_alignment": "align",
        "get_xyz_positions": "monitor",
        "get_all_ui_state": "monitor",
        "get_current_sample_and_test_ocr_passive": "monitor",
        "change_method": "align",
        "set_extension": "align",
        "engage": "align",
    }
    return mapping.get(tool_name, "move_to_location")


def _mode_for_tool(tool_name: str) -> str:
    if tool_name in ("capture_video_panel_remote",):
        return "imaging"
    if tool_name in ("starting_tests", "start_test_normal"):
        return "indentation"
    if tool_name in ("process_review_file",):
        return "data_processing"
    if tool_name in ("align_focus", "focus", "restore_alignment", "change_method"):
        return "alignment"
    return "indentation"
