"""
observability.py
================
LangSmith tracing setup driven by configuration and environment.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .config import AppConfig, get_config


def setup_langsmith(cfg: Optional[AppConfig] = None) -> dict[str, Any]:
    """Configure LangSmith tracing from config and environment."""
    cfg = cfg or get_config()
    enabled = cfg.langsmith.enabled or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not enabled:
        return {}

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    project = os.environ.get("LANGCHAIN_PROJECT") or cfg.langsmith.project
    os.environ.setdefault("LANGCHAIN_PROJECT", project)

    return {
        "tags": ["nanoindent-agent", cfg.experiment.name],
        "metadata": {
            "experiment_name": cfg.experiment.name,
            "project": project,
        },
    }
