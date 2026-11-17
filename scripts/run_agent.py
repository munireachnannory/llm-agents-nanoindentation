#!/usr/bin/env python3
"""
run_agent.py
============
CLI entry point for the LangGraph Nanoindentation Multi-Agent System.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add repo root to path (scripts/ is one level below root)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nanoindentation_agents.notebook_credentials import load_project_env

load_project_env(REPO_ROOT)

from nanoindentation_agents.config import get_config, load_config, reset_config_cache, set_active_config
from nanoindentation_agents.graph import build_nanoindent_graph, visualize_graph
from nanoindentation_agents.state import create_initial_state, AgentState
from nanoindentation_agents.tools.indent_tools import IndentToolkit, set_toolkit


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     LangGraph Nanoindentation Multi-Agent System              ║
║     Autonomous Nanoindentation Experiment Control             ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_state_summary(state: AgentState):
    print("\n" + "=" * 60)
    print("CURRENT STATE")
    print("=" * 60)
    print(f"Stage: {state.get('current_stage', 'unknown')}")
    print(f"Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 10)}")
    print(f"IC cycle: {state.get('ic_cycle', 0)}/{state.get('ic_max_cycles', 8)}")
    if state.get("hypothesis"):
        print(f"\nHypothesis: {state['hypothesis'][:100]}...")
    print(f"\nIndent Arrays: {len(state.get('indent_history', []))}")
    print(f"Decisions: {len(state.get('decision_history', []))}")
    print(f"Visited Targets: {len(state.get('visited_target_keys', []))}")
    if state.get("error_message"):
        print(f"\nError: {state['error_message']}")
    print("-" * 60)


def get_user_approval(state: AgentState) -> dict[str, Any]:
    """Get user approval for hypothesis."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS APPROVAL REQUIRED")
    print("=" * 60)
    print(f"\nHypothesis:\n{state.get('hypothesis', '')}\n")
    if state.get("hypothesis_critique"):
        print(f"Critic:\n{state['hypothesis_critique']}\n")
    
    while True:
        response = input("Approve? [y/n/m/q]: ").strip().lower()
        if response in ("y", "yes"):
            return {"user_approved": True}
        if response in ("n", "no"):
            fb = input("Feedback (optional): ").strip()
            return {
                "user_approved": False,
                "hypothesis_critique": state.get("hypothesis_critique", "") + f"\n{fb}",
            }
        if response in ("m", "modify"):
            print("Enter modified hypothesis (blank line to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            return {"user_approved": True, "hypothesis": "\n".join(lines)}
        if response in ("q", "quit"):
            sys.exit(0)
        print("Invalid option.")


def get_instrument_approval(state: AgentState) -> dict[str, Any]:
    """Get user approval for instrument action."""
    pending = state.get("pending_proposed_action") or {}
    print("\n" + "=" * 60)
    print("INSTRUMENT ACTION APPROVAL")
    print("=" * 60)
    print(json.dumps(pending, indent=2, default=str))
    
    while True:
        response = input("Approve instrument action? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return {"instrument_approved": True}
        if response in ("n", "no"):
            notes = input("Notes (optional): ").strip()
            return {
                "instrument_approved": False,
                "instrument_conclude": True,
                "last_denied_action": {
                    "tool_name": pending.get("tool_name"),
                    "arguments": pending.get("arguments"),
                    "human_notes": notes,
                },
            }
        print("Enter y or n.")


def _init_runtime(cfg_path: Optional[str], args: argparse.Namespace):
    """Initialize runtime configuration and toolkit."""
    cfg = load_config(cfg_path or args.config)
    
    if args.question:
        cfg.research_question = args.question
    if args.name:
        cfg.experiment.name = args.name
    if args.iterations:
        cfg.experiment.max_iterations = args.iterations
    if args.memory_dir:
        cfg.memory.base_dir = args.memory_dir
    if args.auto:
        cfg.graph.auto_approve_instruments = True
        cfg.graph.interrupt_nodes = [n for n in cfg.graph.interrupt_nodes if n != "instrument_approve"]
    if getattr(args, "sim", False):
        cfg.instrument.sim_mode = True

    set_active_config(cfg)

    # Initialize toolkit
    toolkit = IndentToolkit(
        host=cfg.instrument.host,
        port=cfg.instrument.port,
        token=cfg.instrument.token,
        timeout=cfg.instrument.timeout,
        require_approval=cfg.instrument.require_approval,
    )
    set_toolkit(toolkit)

    return cfg


def run_interactive(cfg_path: Optional[str], args: argparse.Namespace):
    """Run in interactive mode with human approval gates."""
    cfg = _init_runtime(cfg_path, args)
    print_banner()
    print(f"Question: {cfg.research_question}")
    print(f"Instrument: {cfg.instrument.url}")
    if cfg.instrument.sim_mode:
        print("  (simulator mode — ensure scripts/run_sim_server.py is running)")
    print(f"Memory: {cfg.memory_dir}")

    graph = build_nanoindent_graph(checkpointer=None, cfg=cfg)
    
    state = create_initial_state(
        research_question=cfg.research_question,
        experiment_name=cfg.experiment.name,
        max_iterations=cfg.experiment.max_iterations,
        ic_max_cycles=cfg.experiment.ic_max_cycles,
    )

    config = {"configurable": {"thread_id": cfg.experiment.name}}
    print("Starting experiment...")

    try:
        for event in graph.stream(state, config, stream_mode="values"):
            stage = event.get("current_stage", "")
            print(f"\n[{stage}]")
            
            if stage == "awaiting_user_approval":
                graph.update_state(config, get_user_approval(event))
            elif stage == "instrument_approve":
                if not cfg.graph.auto_approve_instruments:
                    graph.update_state(config, get_instrument_approval(event))
            
            state = event
        
        print_state_summary(state)
        if state.get("_conclusion_summary"):
            print(state["_conclusion_summary"])
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        reset_config_cache()


def run_automated(cfg_path: Optional[str], args: argparse.Namespace):
    """Run in fully automated mode without human intervention."""
    cfg = _init_runtime(cfg_path, args)
    cfg.graph.auto_approve_instruments = True
    cfg.graph.interrupt_nodes = []
    set_active_config(cfg)

    print_banner()
    print(f"Running automated experiment: {cfg.experiment.name}")
    print(f"Question: {cfg.research_question}")

    graph = build_nanoindent_graph(checkpointer=None, interrupt_before=[], cfg=cfg)
    
    state = create_initial_state(
        research_question=cfg.research_question,
        experiment_name=cfg.experiment.name,
        max_iterations=cfg.experiment.max_iterations,
        ic_max_cycles=cfg.experiment.ic_max_cycles,
    )
    state["user_approved"] = True
    state["instrument_approved"] = True

    config = {"configurable": {"thread_id": cfg.experiment.name}}
    
    try:
        final = graph.invoke(state, config)
        print_state_summary(final)
        if final.get("_conclusion_summary"):
            print(final["_conclusion_summary"])
    finally:
        reset_config_cache()


def check_connection():
    """Check connection to instrument server."""
    cfg = get_config()
    print(f"Checking connection to {cfg.instrument.url}...")
    
    try:
        toolkit = IndentToolkit(
            host=cfg.instrument.host,
            port=cfg.instrument.port,
            token=cfg.instrument.token,
            timeout=10,
        )
        if toolkit.ping():
            print("Connection successful!")
            return True
        else:
            print("Connection failed - server not responding")
            print("  Start simulator: python scripts/run_sim_server.py")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Nanoindentation Multi-Agent System"
    )
    parser.add_argument("-q", "--question", type=str, help="Research question")
    parser.add_argument("-n", "--name", type=str, help="Experiment name")
    parser.add_argument("-i", "--iterations", type=int, help="Max iterations")
    parser.add_argument("--config", type=str, help="Path to experiment YAML/JSON config")
    parser.add_argument("--host", type=str, help="Instrument server host")
    parser.add_argument("--port", type=int, help="Instrument server port")
    parser.add_argument("--memory-dir", type=str, help="Experiment memory directory")
    parser.add_argument("--auto", action="store_true", help="Automated mode")
    parser.add_argument("--visualize", action="store_true", help="Print graph diagram")
    parser.add_argument("--check", action="store_true", help="Check instrument connection")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Expect instrument simulator (start with: python scripts/run_sim_server.py)",
    )
    args = parser.parse_args()

    if args.host:
        os.environ["NANOINDENT_HOST"] = args.host
    if args.port:
        os.environ["NANOINDENT_PORT"] = str(args.port)

    if args.visualize:
        print(visualize_graph())
        return
    
    if args.check:
        check_connection()
        return

    if not args.config and not args.question:
        parser.print_help()
        print("\nError: Provide either --config or --question")
        sys.exit(1)

    if args.config and not args.question:
        pre = load_config(args.config)
        if not pre.research_question:
            print("Error: set research_question in config or pass -q.")
            sys.exit(1)

    if args.auto:
        run_automated(args.config, args)
    else:
        run_interactive(args.config, args)


if __name__ == "__main__":
    main()
