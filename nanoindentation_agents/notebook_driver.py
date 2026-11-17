"""
notebook_driver.py
==================
Run the LangGraph nanoindentation workflow from a Jupyter notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from instrument_control.approval import request_approval
from instrument_control.schemas import ProposedAction, safety

from .checkpointing import create_checkpointer
from .config import load_config, set_active_config
from .graph import build_nanoindent_graph
from .notebook_bridge import graph_to_notebook, notebook_to_graph_state
from .notebook_credentials import print_llm_backend_summary
from .notebook_display import NotebookRunDisplay
from .observability import setup_langsmith
from .state import AgentState
from .tools.indent_tools import IndentToolkit, set_toolkit

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


def _notebook_preset_approval() -> Optional[dict[str, Any]]:
    """
    Read HYPOTHESIS_APPROVAL from notebook globals (set before run cell).
    
    Usage in notebook cell before running workflow:
        HYPOTHESIS_APPROVAL = True          # Auto-approve
        HYPOTHESIS_APPROVAL = False         # Auto-reject
        HYPOTHESIS_APPROVAL = "y"           # Auto-approve  
        HYPOTHESIS_APPROVAL = "My hypothesis text"  # Approve with modified text
        HYPOTHESIS_APPROVAL = None          # Prompt for input (default)
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return None
        preset = ip.user_ns.get("HYPOTHESIS_APPROVAL")
    except Exception:
        return None

    if preset is True or preset in ("y", "yes", "approve"):
        print("[HYPOTHESIS] Auto-approved via HYPOTHESIS_APPROVAL preset")
        return {"user_approved": True}
    if preset is False or preset in ("n", "no", "reject"):
        print("[HYPOTHESIS] Auto-rejected via HYPOTHESIS_APPROVAL preset")
        return {"user_approved": False}
    if isinstance(preset, str) and preset.strip() and len(preset.strip()) > 10:
        # Looks like modified hypothesis text
        print(f"[HYPOTHESIS] Using modified text from HYPOTHESIS_APPROVAL ({len(preset)} chars)")
        return {"user_approved": True, "hypothesis": preset.strip()}
    return None


def _format_hypothesis_html(gs: AgentState) -> str:
    """Format hypothesis as a nice HTML card for Jupyter display."""
    # Color palette
    TEAL = "#0e6655"
    BLUE = "#2c3e50"
    GREEN = "#1e8449"
    ORANGE = "#e67e22"
    RED = "#c0392b"
    
    hyp = gs.get("hypothesis", "No hypothesis generated")
    hyp_data = gs.get("hypothesis_data", {}) or {}
    
    # Extract structured fields if available
    rationale = hyp_data.get("rationale") or gs.get("_hypothesis_rationale", "")
    measurements = hyp_data.get("measurements") or gs.get("_hypothesis_measurements", "")
    success = hyp_data.get("success_criteria") or gs.get("_hypothesis_success_criteria", "")
    failure = hyp_data.get("failure_criteria") or gs.get("_hypothesis_failure_criteria", "")
    critique = gs.get("hypothesis_critique", "")
    
    # Literature context indicator
    lit_ctx = gs.get("literature_context", "")
    lit_sources = gs.get("literature_sources", [])
    lit_badge = ""
    if lit_ctx:
        source_count = len(lit_sources) if lit_sources else 0
        lit_badge = (
            f'<span style="background:{GREEN};color:white;padding:2px 8px;'
            f'border-radius:12px;font-size:11px;margin-left:10px">'
            f'{source_count} literature source{"s" if source_count != 1 else ""}</span>'
        )
    
    # Iteration info
    iteration = gs.get("iteration", 0)
    max_iter = gs.get("max_iterations", 10)
    
    # Build expandable sections
    sections = ""
    
    if rationale:
        sections += (
            f'<details style="margin:8px 0">'
            f'<summary style="cursor:pointer;color:{TEAL};font-weight:bold">'
            f'Rationale</summary>'
            f'<div style="padding:8px 12px;background:#f8f9fa;border-radius:4px;'
            f'margin-top:4px;color:#333">{rationale}</div></details>'
        )
    
    if measurements:
        sections += (
            f'<details style="margin:8px 0">'
            f'<summary style="cursor:pointer;color:{BLUE};font-weight:bold">'
            f'Proposed Measurements</summary>'
            f'<div style="padding:8px 12px;background:#e3f2fd;border-radius:4px;'
            f'margin-top:4px;color:#333">{measurements}</div></details>'
        )
    
    if success:
        sections += (
            f'<details style="margin:8px 0">'
            f'<summary style="cursor:pointer;color:{GREEN};font-weight:bold">'
            f'Success Criteria</summary>'
            f'<div style="padding:8px 12px;background:#e8f5e9;border-radius:4px;'
            f'margin-top:4px;color:#333">{success}</div></details>'
        )
    
    if failure:
        sections += (
            f'<details style="margin:8px 0">'
            f'<summary style="cursor:pointer;color:{RED};font-weight:bold">'
            f'Falsification Criteria</summary>'
            f'<div style="padding:8px 12px;background:#ffebee;border-radius:4px;'
            f'margin-top:4px;color:#333">{failure}</div></details>'
        )
    
    # Critic feedback (if any)
    critic_section = ""
    if critique:
        critic_section = (
            f'<div style="background:#fff3e0;border-left:4px solid {ORANGE};'
            f'padding:10px 14px;margin:10px 0;border-radius:0 4px 4px 0">'
            f'<b style="color:{ORANGE}">Critic Feedback:</b><br>'
            f'<span style="color:#5d4037">{critique[:800]}</span></div>'
        )
    
    # Main card HTML
    html = f'''
    <div style="border:2px solid {TEAL};border-radius:10px;margin:12px 0;
                overflow:hidden;font-family:system-ui,-apple-system,sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,0.1)">
        <div style="background:{TEAL};color:white;padding:12px 16px;
                    display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:15px;font-weight:bold">
                Hypothesis for Approval
            </span>
            <span style="font-size:12px;opacity:0.9">
                Iteration {iteration + 1}/{max_iter} {lit_badge}
            </span>
        </div>
        <div style="padding:16px">
            <div style="background:linear-gradient(135deg,#f5f7fa 0%,#e4e8ec 100%);
                        border-radius:6px;padding:14px;margin-bottom:12px;
                        border-left:4px solid {TEAL}">
                <div style="font-size:14px;line-height:1.6;color:#1a1a2e;
                            white-space:pre-wrap">{hyp}</div>
            </div>
            {critic_section}
            {sections}
        </div>
    </div>
    '''
    return html


def _hypothesis_approval_simple(gs: AgentState) -> dict[str, Any]:
    """Simple text-based approval for Jupyter notebooks."""
    from IPython.display import display, HTML
    
    hyp = gs.get("hypothesis", "")
    
    # Display formatted hypothesis card
    display(HTML(_format_hypothesis_html(gs)))
    
    # Simple text prompt
    print("\n" + "-" * 50)
    print("DECISION REQUIRED")
    print("-" * 50)
    print("  y / yes    = Approve hypothesis")
    print("  n / no     = Reject and regenerate")
    print("  m / modify = Enter modified hypothesis")
    print("  c / cancel = Cancel workflow (raises exception)")
    print("-" * 50)
    
    while True:
        try:
            resp = input("Your choice [y/n/m/c]: ").strip().lower()
        except EOFError:
            # Handle Jupyter kernel restart
            raise KeyboardInterrupt("Hypothesis approval cancelled")
        
        if resp in ("y", "yes", "approve"):
            print("-> Approved")
            return {"user_approved": True}
        
        if resp in ("n", "no", "reject"):
            print("-> Rejected, will regenerate")
            return {"user_approved": False}
        
        if resp in ("m", "modify"):
            print("Enter modified hypothesis (press Enter twice when done):")
            lines = []
            empty_count = 0
            while empty_count < 1:
                try:
                    line = input()
                except EOFError:
                    break
                if not line:
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
            modified = "\n".join(lines).strip()
            if modified:
                print(f"-> Modified and approved ({len(modified)} chars)")
                return {"user_approved": True, "hypothesis": modified}
            else:
                print("-> No modification, approving original")
                return {"user_approved": True}
        
        if resp in ("c", "cancel", "quit", "exit", "q"):
            raise KeyboardInterrupt("Workflow cancelled by user")
        
        print("  Invalid input. Enter y, n, m, or c")


def _hypothesis_approval(gs: AgentState) -> dict[str, Any]:
    """Handle hypothesis approval with preset check first."""
    preset = _notebook_preset_approval()
    if preset is not None:
        print("\n[HYPOTHESIS APPROVAL] Using notebook HYPOTHESIS_APPROVAL preset.")
        return preset

    # Use simple text-based approval (works in both Jupyter and terminal)
    return _hypothesis_approval_simple(gs)


def _instrument_approval(gs: AgentState) -> dict[str, Any]:
    pending = gs.get("pending_proposed_action") or {}
    if not pending.get("tool_name"):
        return {"instrument_approved": False}

    proposed = ProposedAction(
        tool_name=pending["tool_name"],
        arguments=pending.get("arguments", {}),
        justification=pending.get("justification", ""),
        safety=safety(pending.get("risk_level", "low")),
    )
    decision = request_approval(proposed, run_mode="human_suggests_alternatives")

    if decision.decision == "approved":
        return {"instrument_approved": True, "last_denied_action": {}}

    notes = (decision.human_notes or decision.rationale or "").strip()
    feedback = list(gs.get("operator_feedback", []))
    if notes:
        feedback.append(f"{pending.get('tool_name')}: {notes}")
    print("\n→ Denied. Planner will propose a different action.\n")
    stop_words = ("stop", "conclude", "abort", "end experiment", "quit")
    if any(w in notes.lower() for w in stop_words):
        return {
            "instrument_approved": False,
            "instrument_conclude": True,
            "operator_feedback": feedback,
            "scratchpad": [f"operator stopped instrument loop: {notes[:200]}"],
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
        "scratchpad": [f"operator denied {pending.get('tool_name')}: {notes[:200]}"],
    }


def _interrupt_kind(snapshot: Any) -> Optional[str]:
    nxt = getattr(snapshot, "next", None) or ()
    text = " ".join(str(x) for x in nxt)
    if "user_approval" in text:
        return "hypothesis"
    if "instrument_approve" in text:
        return "instrument"
    tasks = getattr(snapshot, "tasks", None) or ()
    text = " ".join(str(t) for t in tasks)
    if "user_approval" in text:
        return "hypothesis"
    if "instrument_approve" in text:
        return "instrument"
    return None


def _merge_state(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in update.items():
        merged[key] = val
    return merged


def _parse_stream_chunk(chunk: Any) -> tuple[tuple, str, dict[str, Any]]:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        namespace, data = chunk
    else:
        namespace, data = (), chunk
    if not isinstance(data, dict):
        return namespace, "", {}
    for node_name, update in data.items():
        if isinstance(update, dict):
            return namespace, node_name, update
    return namespace, "", {}


def run_notebook_workflow(
    notebook_state: Any,
    *,
    config_path: Optional[str] = None,
    auto_approve_hypothesis: bool = False,
    auto_approve_instruments: bool = False,
    on_stage: Optional[Callable[..., None]] = None,
) -> AgentState:
    """
    Execute the full LangGraph workflow and sync results into notebook_state.

    Human gates: hypothesis approval and instrument action approval (unless auto flags).
    """
    cfg = load_config(config_path)
    ctx = getattr(notebook_state, "experiment_context", "") or ""
    if ctx:
        cfg.experiment_context = ctx
        if not cfg.research_question:
            cfg.research_question = ctx[:500]
    if auto_approve_instruments:
        cfg.graph.auto_approve_instruments = True
    set_active_config(cfg)

    mem = Path(cfg.memory.base_dir)
    cp = create_checkpointer(cfg.experiment.name, memory_dir=mem, cfg=cfg)

    toolkit = IndentToolkit(
        host=cfg.instrument.host,
        port=cfg.instrument.port,
        token=cfg.instrument.token,
        timeout=cfg.instrument.timeout,
        require_approval=cfg.instrument.require_approval and not auto_approve_instruments,
    )
    set_toolkit(toolkit)

    interrupt_before = list(cfg.graph.interrupt_nodes)
    if auto_approve_hypothesis:
        interrupt_before = [n for n in interrupt_before if n != "user_approval"]

    # graph = build_nanoindent_graph(checkpointer=cp, interrupt_before=interrupt_before, cfg=cfg)

    lg_checkpointer = InMemorySaver()
    graph = build_nanoindent_graph(
        checkpointer=lg_checkpointer,
        interrupt_before=interrupt_before,
        cfg=cfg,
    )
    initial = notebook_to_graph_state(notebook_state, cfg=cfg)
    initial["run_id"] = cp.run_id
    if auto_approve_hypothesis:
        initial["user_approved"] = True

    config = {**cp.get_config(), **setup_langsmith(cfg), "recursion_limit": 200}
    current: AgentState = initial

    print("Starting LangGraph nanoindentation workflow…")
    print_llm_backend_summary(cfg)
    print(f"  Instrument: {cfg.instrument.url}")
    print(f"  Memory: {mem}")
    print(f"  Run ID: {cp.run_id}")
    print("  Live agent narration below.\n")

    run_display = NotebookRunDisplay()
    stage_cb = on_stage if on_stage is not None else run_display.on_node

    stream_input: Any = initial
    running: dict[str, Any] = dict(initial)
    try:
        while True:
            for chunk in graph.stream(
                stream_input,
                config,
                stream_mode="updates",
                subgraphs=True,
            ):
                namespace, node_name, update = _parse_stream_chunk(chunk)
                if not update:
                    continue
                running = _merge_state(running, update)
                current = running  # type: ignore[assignment]
                if callable(stage_cb):
                    try:
                        stage_cb(node_name, running, namespace)
                    except TypeError:
                        stage_cb(running)

            snap = graph.get_state(config)
            if snap.values:
                running = _merge_state(running, dict(snap.values))
                current = running  # type: ignore[assignment]
            kind = _interrupt_kind(snap)
            if not kind:
                break

            stream_input = None
            if kind == "hypothesis":
                if auto_approve_hypothesis:
                    graph.update_state(config, {"user_approved": True})
                else:
                    graph.update_state(config, _hypothesis_approval(current))
            elif kind == "instrument":
                if auto_approve_instruments:
                    graph.update_state(config, {"instrument_approved": True})
                else:
                    graph.update_state(config, _instrument_approval(current))
            else:
                break

        graph_to_notebook(notebook_state, current)
        run_display.display_final_summary(current)
        if current.get("_conclusion_summary"):
            print(current["_conclusion_summary"])
        notebook_state.run_trace = run_display.trace  # type: ignore[attr-defined]
        return current
    finally:
        cp.close()
