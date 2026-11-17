"""
episodic_store.py
=================
Cross-run episodic memory using SQLite for policies, learnings, and run summaries.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig, get_config


class EpisodicStore:
    """Lightweight episodic memory for policies and run summaries."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._conn.commit()

    def put(self, namespace: str, key: str, value: Any) -> None:
        """Store a value in the episodic memory."""
        self._conn.execute(
            "INSERT OR REPLACE INTO episodic (namespace, key, value) VALUES (?, ?, ?)",
            (namespace, key, json.dumps(value, default=str)),
        )
        self._conn.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve a value from episodic memory."""
        row = self._conn.execute(
            "SELECT value FROM episodic WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    def list_keys(self, namespace: str) -> list[str]:
        """List all keys in a namespace."""
        rows = self._conn.execute(
            "SELECT key FROM episodic WHERE namespace = ? ORDER BY updated_at DESC",
            (namespace,),
        ).fetchall()
        return [r[0] for r in rows]

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a key from episodic memory."""
        cursor = self._conn.execute(
            "DELETE FROM episodic WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace."""
        cursor = self._conn.execute(
            "DELETE FROM episodic WHERE namespace = ?",
            (namespace,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# Namespace constants
NS_POLICIES = "policies"
NS_RUN_SUMMARIES = "run_summaries"
NS_LEARNINGS = "learnings"
NS_MATERIAL_KNOWLEDGE = "material_knowledge"


def get_episodic_store(cfg: AppConfig | None = None) -> EpisodicStore:
    """Get or create the episodic store from config."""
    cfg = cfg or get_config()
    mem_dir = Path(cfg.memory.base_dir)
    db_path = mem_dir / cfg.memory.store_db
    return EpisodicStore(db_path)


def store_run_summary(
    store: EpisodicStore,
    run_id: str,
    summary: dict[str, Any],
) -> None:
    """Store a run summary for future reference."""
    store.put(NS_RUN_SUMMARIES, run_id, summary)


def get_recent_run_summaries(
    store: EpisodicStore,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Get the most recent run summaries."""
    keys = store.list_keys(NS_RUN_SUMMARIES)[:limit]
    summaries = []
    for key in keys:
        summary = store.get(NS_RUN_SUMMARIES, key)
        if summary:
            summaries.append({"run_id": key, **summary})
    return summaries


def store_policy(
    store: EpisodicStore,
    policy_name: str,
    policy: dict[str, Any],
) -> None:
    """Store a learned policy."""
    store.put(NS_POLICIES, policy_name, policy)


def get_policy(
    store: EpisodicStore,
    policy_name: str,
) -> Optional[dict[str, Any]]:
    """Retrieve a stored policy."""
    return store.get(NS_POLICIES, policy_name)


def store_learning(
    store: EpisodicStore,
    material_type: str,
    learning: dict[str, Any],
) -> None:
    """Store a learning about a material type."""
    existing = store.get(NS_LEARNINGS, material_type) or {"learnings": []}
    if not isinstance(existing, dict):
        existing = {"learnings": []}
    existing.setdefault("learnings", []).append(learning)
    # Keep only recent learnings
    existing["learnings"] = existing["learnings"][-20:]
    store.put(NS_LEARNINGS, material_type, existing)


def get_learnings(
    store: EpisodicStore,
    material_type: str,
) -> list[dict[str, Any]]:
    """Get learnings for a material type."""
    data = store.get(NS_LEARNINGS, material_type)
    if not data or not isinstance(data, dict):
        return []
    return data.get("learnings", [])


def store_material_knowledge(
    store: EpisodicStore,
    material_name: str,
    knowledge: dict[str, Any],
) -> None:
    """Store knowledge about a specific material."""
    store.put(NS_MATERIAL_KNOWLEDGE, material_name, knowledge)


def get_material_knowledge(
    store: EpisodicStore,
    material_name: str,
) -> Optional[dict[str, Any]]:
    """Get stored knowledge about a material."""
    return store.get(NS_MATERIAL_KNOWLEDGE, material_name)


def build_run_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Build a run summary from the final state."""
    hypothesis = state.get("hypothesis", "")
    hypothesis_data = state.get("hypothesis_data", {}) or {}
    indent_history = state.get("indent_history") or []
    
    # Calculate statistics
    h_values = [i.get("hardness_mean") for i in indent_history if i.get("hardness_mean")]
    e_values = [i.get("modulus_mean") for i in indent_history if i.get("modulus_mean")]
    
    summary = {
        "hypothesis": hypothesis[:500],
        "hypothesis_rationale": hypothesis_data.get("rationale", "")[:300],
        "iterations": state.get("iteration", 0),
        "indent_arrays": len(indent_history),
        "total_indents": sum(i.get("valid_indent_count", 0) for i in indent_history),
        "decision": state.get("measurement_critique_decision", ""),
        "outcome": state.get("current_stage", ""),
    }
    
    if h_values:
        summary["hardness_range"] = {
            "min": min(h_values),
            "max": max(h_values),
            "mean": sum(h_values) / len(h_values),
        }
    
    if e_values:
        summary["modulus_range"] = {
            "min": min(e_values),
            "max": max(e_values),
            "mean": sum(e_values) / len(e_values),
        }
    
    # Extract key findings
    if state.get("policy"):
        policy = state["policy"]
        summary["recommendations"] = policy.get("recommendations", [])[:5]
        summary["key_findings"] = policy.get("key_findings", [])[:5]
    
    return summary
