"""
schemas.py
==========
Data structures for instrument control workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SafetyInfo:
    """Safety assessment for a proposed action."""
    risk_level: str = "low"
    expected_side_effects: list[str] = field(default_factory=list)
    estimated_duration_s: Optional[float] = None
    requires_supervision: bool = False


def safety(
    risk_level: str = "low",
    expected_side_effects: Optional[list[str]] = None,
    estimated_duration_s: Optional[float] = None,
) -> SafetyInfo:
    """Create a SafetyInfo object."""
    return SafetyInfo(
        risk_level=risk_level,
        expected_side_effects=expected_side_effects or [],
        estimated_duration_s=estimated_duration_s,
        requires_supervision=risk_level in ("medium", "high"),
    )


@dataclass
class ProposedAction:
    """A proposed instrument action awaiting approval."""
    tool_name: str
    arguments: dict[str, Any]
    justification: str = ""
    safety: SafetyInfo = field(default_factory=SafetyInfo)
    success_criteria: list[str] = field(default_factory=list)


@dataclass
class ApprovalDecision:
    """Human approval decision for a proposed action."""
    decision: str  # "approved", "rejected", "modified"
    rationale: str = ""
    human_notes: str = ""
    modified_arguments: Optional[dict[str, Any]] = None


@dataclass
class ToolOutcome:
    """Result of executing an instrument tool."""
    ok: bool
    result: Any = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    # Set when the SERVER answered but the automation method itself failed.
    # The Flask server returns HTTP 400 with {"message", "traceback"} for any
    # exception raised inside the method (instrument_server.py /call_auto), so
    # a 400 means "the instrument PC tried and threw" — NOT a network problem.
    # Callers must be able to tell those apart without parsing English.
    status_code: Optional[int] = None
    server_message: Optional[str] = None
    server_traceback: Optional[str] = None

    @property
    def server_side_failure(self) -> bool:
        """True when the server answered and the method raised (vs transport)."""
        return self.status_code is not None and not self.ok

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status_code": self.status_code,
            "server_message": self.server_message,
        }
