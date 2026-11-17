"""
checkpointing.py
================
LangGraph state persistence for notebook and CLI runs.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None  # type: ignore[misc, assignment]

from .config import AppConfig, get_config

_runtime_checkpointer: Optional["NotebookCheckpointer"] = None


class NotebookCheckpointer:
    """LangGraph state persistence for nanoindentation experiments."""

    def __init__(
        self,
        memory_dir: Path | str,
        experiment_name: str,
        run_id: Optional[str] = None,
        *,
        checkpoint_path: Optional[Path] = None,
        use_sqlite: bool = True,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.run_id = run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if use_sqlite and SqliteSaver is not None and checkpoint_path:
            import sqlite3

            conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
            self.langgraph_saver = SqliteSaver(conn)
        else:
            self.langgraph_saver = MemorySaver()

    def close(self) -> None:
        if hasattr(self.langgraph_saver, "conn"):
            try:
                self.langgraph_saver.conn.close()
            except Exception:
                pass

    def get_config(self, thread_id: Optional[str] = None) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id or self.run_id}}


def get_runtime_checkpointer() -> Optional[NotebookCheckpointer]:
    return _runtime_checkpointer


def set_runtime_checkpointer(cp: NotebookCheckpointer) -> None:
    global _runtime_checkpointer
    _runtime_checkpointer = cp


def create_checkpointer(
    experiment_name: str,
    memory_dir: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
) -> NotebookCheckpointer:
    cfg = cfg or get_config()
    base = Path(memory_dir) if memory_dir else cfg.memory_dir
    base.mkdir(parents=True, exist_ok=True)
    cp = NotebookCheckpointer(
        memory_dir=base,
        experiment_name=experiment_name,
        run_id=run_id,
        checkpoint_path=base / cfg.memory.checkpoint_db,
        use_sqlite=True,
    )
    set_runtime_checkpointer(cp)
    return cp
