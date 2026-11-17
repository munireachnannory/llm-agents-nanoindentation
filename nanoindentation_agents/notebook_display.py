"""
notebook_display.py
===================
Lightweight notebook narration during LangGraph runs.
"""

from __future__ import annotations

import json
from typing import Any

NODE_TO_STAGE: dict[str, str] = {
    "observe": "observe",
    "opening_image_analysis": "image_analysis",
    "hypothesis_image_analysis": "image_analysis",
    "literature": "literature_search",
    "hypothesis_gen": "hypothesis_generation",
    "hypothesis_critic": "hypothesis_critic",
    "user_approval": "awaiting_user_approval",
    "supervisor": "supervisor",
    "plan": "instrument_plan",
    "instrument_approve": "instrument_approve",
    "execute": "instrument_execute",
    "interpret": "instrument_interpret",
    "spatial": "instrument_spatial",
    "instrument_subgraph": "instrument_subgraph",
    "image_analysis_agent": "image_analysis",
    "measurement_critic": "measurement_critic",
    "policy_learning": "policy_learning",
    "conclude": "concluded",
}

AGENT_NAMES: dict[str, str] = {
    "observe": "Observe Agent",
    "literature_search": "Literature Agent",
    "hypothesis_generation": "Hypothesis Agent",
    "hypothesis_critic": "Hypothesis Critic",
    "awaiting_user_approval": "Human Checkpoint",
    "supervisor": "Supervisor Agent",
    "instrument_plan": "Instrument Planner",
    "instrument_approve": "Instrument Approval",
    "instrument_execute": "Instrument Executor",
    "instrument_interpret": "Instrument Interpreter",
    "instrument_spatial": "Spatial Analyst",
    "instrument_subgraph": "Instrument Loop",
    "image_analysis": "Image Analysis Agent",
    "measurement_critic": "Measurement Critic",
    "policy_learning": "Policy Learning Agent",
    "concluded": "Experiment Conclusion",
}


class NotebookRunDisplay:
    """Accumulates and prints agent narration during a LangGraph run."""

    def __init__(self) -> None:
        self.trace: list[dict[str, Any]] = []

    def on_node(self, node_name: str, state: dict[str, Any], namespace: Any = ()) -> None:
        stage = NODE_TO_STAGE.get(node_name, node_name)
        agent = AGENT_NAMES.get(stage, node_name)
        entry = {
            "node": node_name,
            "stage": stage,
            "iteration": state.get("iteration", 0),
            "ic_cycle": state.get("ic_cycle", 0),
        }
        self.trace.append(entry)

        print("\n" + "=" * 72)
        print(f"[{agent}]  stage={state.get('current_stage', stage)}")
        if state.get("supervisor_reasoning") and node_name == "supervisor":
            print(f"Routing: {state['supervisor_reasoning'][:300]}")
        
        # Show literature agent output
        if node_name == "literature":
            lit_ctx = state.get("literature_context", "")
            lit_query = state.get("literature_search_query", "")
            lit_sources = state.get("literature_sources", [])
            if lit_query:
                print(f"Query: {lit_query[:120]}")
            if lit_sources:
                print(f"Sources: {', '.join(lit_sources[:3])}")
            if lit_ctx:
                print(f"Literature context (→ hypothesis): {lit_ctx[:300]}...")
            else:
                print("WARNING: No literature context retrieved!")
        
        # Show hypothesis agent using literature
        if node_name == "hypothesis_gen":
            lit_ctx = state.get("literature_context", "")
            if lit_ctx:
                print(f"Using literature context ({len(lit_ctx)} chars)")
            else:
                print("NOTE: No literature context available for hypothesis")
        
        if state.get("hypothesis") and node_name in ("hypothesis_gen", "hypothesis_critic"):
            print(f"Hypothesis: {state['hypothesis'][:240]}")
        if state.get("pending_proposed_action") and node_name == "plan":
            pending = state["pending_proposed_action"]
            print(f"Planned tool: {pending.get('tool_name')}  args={json.dumps(pending.get('arguments', {}), default=str)[:200]}")
        if state.get("last_tool_result") and node_name == "execute":
            result = state["last_tool_result"]
            print(f"Tool result: ok={result.get('ok')}  summary={str(result)[:240]}")

    def display_final_summary(self, state: dict[str, Any]) -> None:
        print("\n" + "=" * 72)
        print("RUN SUMMARY")
        print("=" * 72)
        print(f"Stage: {state.get('current_stage')}")
        print(f"Iterations: {state.get('iteration', 0)}")
        print(f"Indent arrays: {len(state.get('indent_history', []))}")
        print(f"Instrument decisions: {len(state.get('decision_history', []))}")
        print(f"Visited targets: {len(state.get('visited_target_keys', []))}")
        if state.get("hypothesis"):
            print(f"\nHypothesis:\n{state['hypothesis'][:500]}")
