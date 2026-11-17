"""
approval.py
===========
Human approval workflow for instrument actions.
"""

from __future__ import annotations

from typing import Any

from .schemas import ProposedAction, ApprovalDecision


def request_approval(
    proposed: ProposedAction,
    run_mode: str = "human_approves",
) -> ApprovalDecision:
    """
    Request human approval for a proposed action.
    
    In notebook/CLI context, this displays the action and waits for input.
    In automated mode, this returns auto-approval.
    
    Parameters
    ----------
    proposed : ProposedAction
        The proposed instrument action
    run_mode : str
        The run mode: "human_approves", "human_suggests_alternatives", "auto"
        
    Returns
    -------
    ApprovalDecision
        The approval decision
    """
    if run_mode == "auto":
        return ApprovalDecision(
            decision="approved",
            rationale="Auto-approved in automated mode",
        )
    
    # Display proposed action
    print("\n" + "=" * 50)
    print("PROPOSED INSTRUMENT ACTION")
    print("=" * 50)
    print(f"Tool: {proposed.tool_name}")
    print(f"Arguments: {proposed.arguments}")
    print(f"Justification: {proposed.justification}")
    print(f"Risk Level: {proposed.safety.risk_level}")
    print("-" * 50)
    
    while True:
        response = input("Approve? [y/n/m]: ").strip().lower()
        
        if response in ("y", "yes"):
            return ApprovalDecision(
                decision="approved",
                rationale="Human approved",
            )
        elif response in ("n", "no"):
            notes = input("Notes (why rejected): ").strip()
            return ApprovalDecision(
                decision="rejected",
                rationale="Human rejected",
                human_notes=notes,
            )
        elif response in ("m", "modify"):
            print("Enter modified arguments as JSON (or 'cancel'):")
            import json
            mod_input = input().strip()
            if mod_input.lower() == "cancel":
                continue
            try:
                modified = json.loads(mod_input)
                return ApprovalDecision(
                    decision="modified",
                    rationale="Human modified arguments",
                    modified_arguments=modified,
                )
            except json.JSONDecodeError:
                print("Invalid JSON. Try again.")
        else:
            print("Enter y (yes), n (no), or m (modify)")


def display_action_for_approval(proposed: ProposedAction) -> str:
    """Format a proposed action for display."""
    lines = [
        "=" * 50,
        "PROPOSED INSTRUMENT ACTION",
        "=" * 50,
        f"Tool: {proposed.tool_name}",
        f"Arguments: {proposed.arguments}",
        f"Justification: {proposed.justification}",
        f"Risk Level: {proposed.safety.risk_level}",
        "-" * 50,
    ]
    return "\n".join(lines)
