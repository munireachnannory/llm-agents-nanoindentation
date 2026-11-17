"""
notebook_bridge.py
==================
Sync between notebook @dataclass state and LangGraph AgentState.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState, create_initial_state


def notebook_to_graph_state(nb: Any, *, cfg: Any = None) -> AgentState:
    """Build LangGraph initial state from notebook state."""
    rq = (
        getattr(nb, "experiment_context", None)
        or getattr(cfg, "research_question", None)
        or ""
    )
    if cfg and getattr(cfg, "research_question", None):
        rq = rq or cfg.research_question

    gs = create_initial_state(
        research_question=str(rq).strip(),
        experiment_name=getattr(cfg, "experiment", None).name if cfg else "nanoindent_notebook",
        max_iterations=getattr(cfg, "experiment", None).max_iterations if cfg else 10,
        ic_max_cycles=getattr(cfg, "experiment", None).ic_max_cycles if cfg else 12,
    )
    gs["experiment_context"] = str(getattr(nb, "experiment_context", "") or rq)
    if getattr(nb, "hypothesis", None):
        gs["hypothesis"] = nb.hypothesis
    if getattr(nb, "policy", None):
        gs["policy"] = dict(nb.policy)
    if getattr(nb, "current_image", None):
        gs["current_image"] = dict(nb.current_image)
    if getattr(nb, "indent_history", None):
        gs["indent_history"] = list(nb.indent_history)
    if getattr(nb, "decision_history", None):
        gs["decision_history"] = list(nb.decision_history)
    if getattr(nb, "operator_feedback", None):
        gs["operator_feedback"] = list(nb.operator_feedback)
    return gs


def graph_to_notebook(nb: Any, gs: AgentState) -> None:
    """Copy graph state back into notebook state for inspection cells."""
    if gs.get("hypothesis"):
        nb.hypothesis = gs["hypothesis"]
    if gs.get("hypothesis_data"):
        nb.hypothesis_data = gs["hypothesis_data"]
    if gs.get("current_image"):
        nb.current_image = dict(gs["current_image"])
    if gs.get("current_indent"):
        nb.current_indent = dict(gs["current_indent"])
    nb.indent_history = list(gs.get("indent_history") or [])
    nb.image_history = list(gs.get("image_history") or [])
    nb.current_position = gs.get("current_position", (0.0, 0.0))
    nb.current_extension = gs.get("current_extension", 0.0)
    nb.plane_params = list(gs.get("plane_params") or [])
    nb.alignment_data = dict(gs.get("alignment_data") or {})
    nb.instrument_status = dict(gs.get("instrument_status") or {})
    nb.spatial_options = list(gs.get("spatial_options") or [])
    nb.measurement_plan = list(gs.get("measurement_plan") or [])
    if gs.get("current_target"):
        nb.current_target = dict(gs["current_target"])
    if gs.get("visited_target_keys"):
        nb.visited_target_keys = list(gs["visited_target_keys"])
    nb.decision_history = list(gs.get("decision_history") or [])
    nb.test_progress = dict(gs.get("test_progress") or {})
    nb.policy = dict(gs.get("policy") or {})
    nb.validation_report = gs.get("measurement_critique", "") or ""
    nb.graph_state = dict(gs)
    nb.literature_context = gs.get("literature_context", "")
    nb.literature_sources = list(gs.get("literature_sources") or [])
    nb.literature_raw_context = gs.get("literature_raw_context", "")
    nb.hypothesis_critique = gs.get("hypothesis_critique", "")
    nb.supervisor_reasoning = gs.get("supervisor_reasoning", "")
    ia = gs.get("image_analysis") or {}
    nb.image_analysis = dict(ia) if ia else {}
    nb.image_analysis_artifacts = dict(gs.get("image_analysis_artifacts") or {})
    nb.last_instrument_interpretation = dict(gs.get("last_instrument_interpretation") or {})
