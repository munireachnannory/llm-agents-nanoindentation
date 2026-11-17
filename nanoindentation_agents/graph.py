"""
graph.py
========
Config-driven LangGraph assembly with supervisor Command routing.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents.hypothesis_agent import hypothesis_node
from .agents.image_analysis_agent import (
    hypothesis_image_analysis_node,
    image_analysis_node,
    opening_image_analysis_node,
)
from .agents.literature_agent import literature_node
from .agents.observe_agent import observe_node
from .agents.policy_learning_agent import policy_learning_node
from .agents.supervisor_agent import supervisor_node
from .config import AppConfig, get_config
from .critics.hypothesis_critic import hypothesis_critic_node
from .critics.measurement_critic import measurement_critic_node, get_routing_decision
from .state import AgentState, create_initial_state
from .subgraphs.instrument import build_instrument_subgraph


def route_after_hypothesis(state: AgentState) -> Literal["user_approval", "hypothesis_gen", "literature", "supervisor"]:
    """Route after hypothesis critic evaluation."""
    cfg = get_config()
    approved = state.get("hypothesis_critique_approved", False)
    history = state.get("hypothesis_history", [])
    max_rev = cfg.experiment.max_hypothesis_revisions
    mid_run_refinement = state.get("user_approved", False) and state.get("iteration", 0) > 0
    min_passes = max(1, int(getattr(cfg.experiment, "min_hypothesis_critic_passes", 1) or 1))
    critic_passes = int(state.get("hypothesis_critic_passes", 0) or 0)

    if state.get("literature_refresh_requested") and not approved:
        return "literature"
    if approved and critic_passes < min_passes and len(history) < max_rev:
        return "hypothesis_gen"
    if approved and mid_run_refinement:
        return "supervisor"
    if len(history) >= max_rev and cfg.routing.force_hypothesis_approval_after_revisions:
        return "supervisor" if mid_run_refinement else "user_approval"
    return "user_approval" if approved else "hypothesis_gen"


def route_after_user_approval(state: AgentState) -> Literal["supervisor", "hypothesis_gen"]:
    """Route after user approval."""
    return "supervisor" if state.get("user_approved", False) else "hypothesis_gen"


def route_after_literature(state: AgentState) -> Literal["hypothesis_gen", "supervisor"]:
    """Route after literature retrieval."""
    if state.get("hypothesis_refinement_requested"):
        return "hypothesis_gen"
    if state.get("iteration", 0) > 0 or state.get("user_approved"):
        return "supervisor"
    return "hypothesis_gen"


def route_after_hypothesis_node(state: AgentState) -> Literal["hypothesis_image_analysis", "supervisor"]:
    """Route after hypothesis generation."""
    return "hypothesis_image_analysis"


def route_after_measurement_critic(state: AgentState) -> Literal["supervisor", "hypothesis_gen", "literature", "policy_learning"]:
    """Route after measurement critique."""
    decision = get_routing_decision(state)
    if decision == "conclude":
        return "policy_learning"
    if state.get("hypothesis_refinement_requested") and state.get("literature_refresh_requested"):
        return "literature"
    if decision == "hypothesis_invalid":
        return "hypothesis_gen"
    return "supervisor"


def user_approval_node(state: AgentState) -> dict[str, Any]:
    """User approval gate node."""
    return {"current_stage": "awaiting_user_approval"}


def conclude_node(state: AgentState) -> dict[str, Any]:
    """Experiment conclusion node."""
    hypothesis = state.get("hypothesis", "No hypothesis")
    decision = state.get("measurement_critique_decision", "")
    iteration = state.get("iteration", 0)
    indent_history = state.get("indent_history", [])
    
    summary_parts = [
        "=" * 60,
        "EXPERIMENT CONCLUSION",
        "=" * 60,
        "",
        f"Final Hypothesis: {hypothesis[:200]}",
        f"Outcome: {decision.upper().replace('_', ' ')}",
        f"Total Iterations: {iteration}",
        "",
    ]
    
    if indent_history:
        summary_parts.append(f"Indent Arrays Collected: {len(indent_history)}")
        total_indents = sum(i.get("valid_indent_count", 0) for i in indent_history)
        summary_parts.append(f"Total Valid Indents: {total_indents}")
        
        # Calculate overall statistics
        h_values = [i.get("hardness_mean", 0) for i in indent_history if i.get("hardness_mean")]
        e_values = [i.get("modulus_mean", 0) for i in indent_history if i.get("modulus_mean")]
        
        if h_values:
            summary_parts.append(f"Hardness Range: {min(h_values):.2f} - {max(h_values):.2f} GPa")
        if e_values:
            summary_parts.append(f"Modulus Range: {min(e_values):.2f} - {max(e_values):.2f} GPa")
    
    if state.get("policy"):
        policy = state["policy"]
        if policy.get("recommendations"):
            summary_parts.append(f"\nRecommendations: {'; '.join(policy['recommendations'])}")
    
    summary_parts.append("-" * 60)
    
    return {
        "current_stage": "concluded",
        "should_continue": False,
        "_conclusion_summary": "\n".join(summary_parts),
    }


def build_nanoindent_graph(
    checkpointer=None,
    interrupt_before: list[str] | None = None,
    cfg: Optional[AppConfig] = None,
) -> CompiledStateGraph:
    """
    Build the nanoindentation multi-agent graph.
    
    Parameters
    ----------
    checkpointer : optional
        LangGraph checkpointer for state persistence
    interrupt_before : list[str], optional
        Nodes to interrupt before (for human-in-the-loop)
    cfg : AppConfig, optional
        Configuration object
        
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow
    """
    cfg = cfg or get_config()
    
    if interrupt_before is None:
        interrupt_before = list(cfg.graph.interrupt_nodes)

    # Build instrument subgraph
    instrument_subgraph = build_instrument_subgraph()

    # Build main graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("literature", literature_node)
    graph.add_node("hypothesis_gen", hypothesis_node)
    graph.add_node("hypothesis_critic", hypothesis_critic_node)
    graph.add_node("user_approval", user_approval_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("instrument_subgraph", instrument_subgraph)
    graph.add_node("hypothesis_image_analysis", hypothesis_image_analysis_node)
    graph.add_node("image_analysis_agent", image_analysis_node)
    graph.add_node("measurement_critic", measurement_critic_node)
    graph.add_node("policy_learning", policy_learning_node)
    graph.add_node("conclude", conclude_node)
    graph.add_node("observe", observe_node)
    graph.add_node("opening_image_analysis", opening_image_analysis_node)

    # Add edges
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "opening_image_analysis")
    graph.add_edge("opening_image_analysis", "literature")
    
    graph.add_conditional_edges(
        "literature",
        route_after_literature,
        {"hypothesis_gen": "hypothesis_gen", "supervisor": "supervisor"},
    )

    graph.add_conditional_edges(
        "hypothesis_gen",
        route_after_hypothesis_node,
        {
            "hypothesis_image_analysis": "hypothesis_image_analysis",
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("hypothesis_image_analysis", "hypothesis_critic")

    graph.add_conditional_edges(
        "hypothesis_critic",
        route_after_hypothesis,
        {
            "user_approval": "user_approval",
            "hypothesis_gen": "hypothesis_gen",
            "literature": "literature",
            "supervisor": "supervisor",
        },
    )

    graph.add_conditional_edges(
        "user_approval",
        route_after_user_approval,
        {"supervisor": "supervisor", "hypothesis_gen": "hypothesis_gen"},
    )

    graph.add_edge("instrument_subgraph", "image_analysis_agent")
    graph.add_edge("image_analysis_agent", "measurement_critic")

    graph.add_conditional_edges(
        "measurement_critic",
        route_after_measurement_critic,
        {
            "supervisor": "supervisor",
            "hypothesis_gen": "hypothesis_gen",
            "literature": "literature",
            "policy_learning": "policy_learning",
        },
    )
    
    graph.add_edge("policy_learning", "conclude")
    graph.add_edge("conclude", END)

    # Compile
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
        
    return graph.compile(**compile_kwargs)


def build_simple_graph(cfg: Optional[AppConfig] = None) -> CompiledStateGraph:
    """Build a simple graph without interrupts (auto-approve mode)."""
    cfg = cfg or get_config()
    cfg.graph.auto_approve_instruments = True
    return build_nanoindent_graph(checkpointer=None, interrupt_before=[], cfg=cfg)


def run_experiment(
    research_question: str,
    experiment_name: Optional[str] = None,
    max_iterations: Optional[int] = None,
    auto_approve: bool = False,
    config_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run a complete nanoindentation experiment.
    
    Parameters
    ----------
    research_question : str
        The scientific question to investigate
    experiment_name : str, optional
        Name for the experiment
    max_iterations : int, optional
        Maximum number of iterations
    auto_approve : bool
        Whether to auto-approve all actions
    config_path : str, optional
        Path to configuration file
        
    Returns
    -------
    dict
        Final experiment state
    """
    from .config import load_config, set_active_config

    cfg = load_config(config_path)
    if research_question:
        cfg.research_question = research_question
    if experiment_name:
        cfg.experiment.name = experiment_name
    if max_iterations is not None:
        cfg.experiment.max_iterations = max_iterations
    if auto_approve:
        cfg.graph.auto_approve_instruments = True
        cfg.graph.interrupt_nodes = []

    set_active_config(cfg)

    graph = build_nanoindent_graph(
        checkpointer=None,
        interrupt_before=[] if auto_approve else None,
        cfg=cfg,
    )

    initial = create_initial_state(
        research_question=cfg.research_question or research_question,
        experiment_name=cfg.experiment.name,
        max_iterations=cfg.experiment.max_iterations,
        ic_max_cycles=cfg.experiment.ic_max_cycles,
    )
    initial["user_approved"] = auto_approve

    config = {"configurable": {"thread_id": experiment_name or "nanoindent-1"}}
    
    final_state = graph.invoke(initial, config)
    return final_state or {}


def visualize_graph() -> str:
    """Return a Mermaid diagram of the graph structure."""
    return """
```mermaid
stateDiagram-v2
    [*] --> observe
    observe --> opening_image_analysis
    opening_image_analysis --> literature
    literature --> hypothesis
    hypothesis --> hypothesis_image_analysis
    hypothesis_image_analysis --> hypothesis_critic
    hypothesis_critic --> user_approval
    hypothesis_critic --> hypothesis
    hypothesis_critic --> literature
    hypothesis_critic --> supervisor
    user_approval --> supervisor
    supervisor --> instrument_subgraph
    supervisor --> image_analysis
    supervisor --> hypothesis
    supervisor --> literature
    supervisor --> policy_learning
    instrument_subgraph --> image_analysis
    image_analysis --> measurement_critic
    measurement_critic --> supervisor
    measurement_critic --> hypothesis
    measurement_critic --> literature
    measurement_critic --> policy_learning
    policy_learning --> conclude
    conclude --> [*]
```
""".strip()
