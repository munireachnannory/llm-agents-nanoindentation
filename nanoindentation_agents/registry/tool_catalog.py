"""
registry/tool_catalog.py
========================
Load and format the instrument tool catalog for LLM prompts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from ..tools.indent_tools import AGENT_SAFE_TOOL_NAMES, TOOL_NAME_MAP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CATALOG_PATH = _REPO_ROOT / "config" / "instrument_tools.yaml"


def _normalize_tool_name(name: str) -> str:
    if name.startswith("indent_"):
        name = name[7:]
    return TOOL_NAME_MAP.get(name, name)


@lru_cache(maxsize=1)
def load_instrument_catalog() -> dict[str, Any]:
    if not _CATALOG_PATH.exists():
        return {"tools": {}, "workflows": {}, "decision_types": {}}
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def get_allowed_tool_names() -> set[str]:
    return set(AGENT_SAFE_TOOL_NAMES)


def format_workflows_text(catalog: Optional[dict[str, Any]] = None) -> str:
    catalog = catalog or load_instrument_catalog()
    workflows = catalog.get("workflows") or {}
    if not workflows:
        return "(no workflow definitions)"
    lines = []
    for name, wf in workflows.items():
        desc = (wf.get("description") or "").strip().replace("\n", " ")
        steps = wf.get("steps") or []
        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        lines.append("Steps: " + " → ".join(steps))
        lines.append("")
    return "\n".join(lines).strip()


def format_tool_catalog_text(catalog: Optional[dict[str, Any]] = None) -> str:
    """Format full tool catalog for injection into planner prompts."""
    catalog = catalog or load_instrument_catalog()
    tools = catalog.get("tools") or {}
    decision_types = catalog.get("decision_types") or {}
    lines = []

    if decision_types:
        lines.append("### Decision types")
        for dt, desc in decision_types.items():
            lines.append(f"- {dt}: {desc}")
        lines.append("")

    lines.append("### Allowed tools")
    for name in AGENT_SAFE_TOOL_NAMES:
        meta = tools.get(name, {})
        desc = meta.get("description", f"Execute {name}")
        risk = meta.get("risk_level", "low")
        lines.append(f"- `{name}` ({risk}): {desc}")
        params = meta.get("parameters") or {}
        if params:
            param_parts = []
            for pname, pspec in params.items():
                req = "required" if pspec.get("required") else "optional"
                pdesc = pspec.get("description", pspec.get("type", ""))
                enum = pspec.get("enum")
                if enum:
                    pdesc = f"{pdesc} one of {enum}" if pdesc else f"one of {enum}"
                param_parts.append(f"{pname} ({req}): {pdesc}")
            lines.append("  Parameters: " + "; ".join(param_parts))
        notes = meta.get("notes")
        if notes:
            lines.append(f"  Notes: {notes}")
        prereq = meta.get("prerequisites")
        if prereq:
            lines.append(f"  Prerequisites: {', '.join(prereq)}")
        typical = meta.get("typical_next")
        if typical:
            lines.append(f"  Typical next: {', '.join(typical)}")
    return "\n".join(lines)


def format_valid_tools_short() -> str:
    return ", ".join(f"`{n}`" for n in AGENT_SAFE_TOOL_NAMES)


def merge_toolkit_catalog(toolkit_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich live toolkit listing with YAML metadata."""
    catalog = load_instrument_catalog()
    yaml_tools = catalog.get("tools") or {}
    merged = []
    for tool in toolkit_tools:
        raw_name = tool.get("name", "")
        name = _normalize_tool_name(raw_name)
        meta = yaml_tools.get(name, {})
        merged.append({
            **tool,
            "name": name,
            "description": meta.get("description") or tool.get("description", ""),
            "risk_level": meta.get("risk_level", "low"),
            "parameters": meta.get("parameters") or tool.get("parameters", {}),
        })
    return merged


def parse_llm_json(response: str) -> dict[str, Any]:
    text = response.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()
    else:
        # Try to extract first JSON object
        i = text.find("{")
        j = text.rfind("}")
        if i >= 0 and j > i:
            text = text[i : j + 1]
    return json.loads(text)
