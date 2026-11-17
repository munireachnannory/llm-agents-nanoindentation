"""
Approval policies — the safety model in one file, not five scattered switches.

Design rules (fixing the drift found in notebook 5):
* The default is ALWAYS manual. Nothing auto-approves because an environment
  variable happened to be set.
* Simulator auto-approval refuses to construct against real hardware.
* Autonomous REAL-HARDWARE approval exists, but only behind an explicit
  acknowledgement phrase typed by the operator in the session — an act of
  commission, never a default.

Manual approval keeps the decision-variable style (MY_DECISION = "approve" /
"reject: reason") rather than input() prompts.
"""
from __future__ import annotations

from typing import Optional

AUTONOMOUS_ACK = "I ACCEPT AUTONOMOUS INDENTS ON REAL HARDWARE"


def parse_decision(text: Optional[str]) -> tuple[Optional[str], str]:
    """'approve[: note]' / 'reject[: reason]' -> ('approve'|'reject'|None, note)."""
    t = (text or "").strip()
    if not t:
        return None, ""
    low = t.lower()
    note = t.partition(":")[2].strip()
    if low.startswith("approve"):
        return "approve", note
    if low.startswith("reject"):
        return "reject", note
    return None, ""


def apply_decision(gate, decision: Optional[str], operator: str) -> bool:
    """Apply a manual decision string to an ApprovalGate.

    Returns True when a decision (either way) was applied; False when the gate
    is left PENDING — the caller must treat pending as a hard stop.
    """
    verdict, note = parse_decision(decision)
    if verdict == "approve":
        gate.approve(approved_by=operator, note=note or "Approved by operator")
        return True
    if verdict == "reject":
        gate.reject(reason=note or "Rejected by operator", approved_by=operator)
        return True
    print("PAUSED — this gate is still PENDING and the workflow will not continue.")
    print('  Set the decision to "approve" or "reject: <reason>" and re-run.')
    return False


class AutoApproval:
    """Auto-approve every gate — SIMULATOR ONLY by construction."""

    interactive = False
    allows_agent_recipe_changes = True

    def __init__(self, *, use_simulator: bool):
        if not use_simulator:
            raise PermissionError(
                "AutoApproval is simulator-only. For a hands-free run on real "
                "hardware, construct AutonomousHardwareApproval(operator, ack=...) "
                "with the explicit acknowledgement phrase.")
        self.approved_by = "AUTO (unattended simulator run)"
        self.note = "Auto-approved: simulator auto policy"

    def approve(self, gate) -> None:
        gate.approve(approved_by=self.approved_by, note=self.note)


class AutonomousHardwareApproval:
    """Auto-approve irreversible rounds on REAL hardware.

    Requires the operator to pass the exact acknowledgement phrase
    (campaign.AUTONOMOUS_ACK) — the notebook never enables this by default,
    and no environment variable can turn it on.
    """

    interactive = False
    # Construction requires the exact acknowledgement phrase below. That
    # explicit session-level consent covers validated agent-selected recipes as
    # well as fixed recipes; all downstream motion and method checks still run.
    allows_agent_recipe_changes = True

    def __init__(self, operator: str, *, ack: str):
        if ack != AUTONOMOUS_ACK:
            raise PermissionError(
                "Autonomous hardware approval needs the exact acknowledgement "
                f"phrase:\n    ack={AUTONOMOUS_ACK!r}\n"
                "This is deliberate friction: every round it approves places "
                "irreversible indents with no per-round human gate.")
        self.operator = operator
        self.approved_by = f"AUTONOMOUS (real hardware, acknowledged by {operator})"
        self.note = "Auto-approved: operator enabled the autonomous campaign this session"

    def approve(self, gate) -> None:
        gate.approve(approved_by=self.approved_by, note=self.note)
