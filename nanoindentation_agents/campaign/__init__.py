"""
Campaign layer — composes the instrument layer into gated, journaled rounds.

    journal.py   CampaignJournal: append-only JSONL event log; the durable
                 record that survives kernel restarts and enables recovery.
    approval.py  Approval policies: manual decision strings, simulator auto,
                 and explicit-opt-in autonomous hardware approval.
    runner.py    CampaignRunner: ONE round implementation shared by round 1,
                 manual rounds, and the autonomous loop.
"""
from .journal import CampaignJournal
from .approval import (AutoApproval, AutonomousHardwareApproval,
                       apply_decision, AUTONOMOUS_ACK)
from .runner import CampaignConfig, CampaignRunner, RoundProposal

__all__ = [
    "CampaignJournal", "AutoApproval", "AutonomousHardwareApproval",
    "apply_decision", "AUTONOMOUS_ACK",
    "CampaignConfig", "CampaignRunner", "RoundProposal",
]
