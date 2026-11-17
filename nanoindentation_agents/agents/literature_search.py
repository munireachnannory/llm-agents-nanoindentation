"""
literature_search.py
====================
Plan literature search queries and keywords from experiment input.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..config import AppConfig, get_config
from ..registry.models import get_llm_for_role
from ..registry.tool_catalog import parse_llm_json
from ..state import AgentState

_DEFAULT_QUERY = "nanoindentation hardness elastic modulus microstructure grain boundary"

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "use",
    "through", "then", "that", "this", "from", "into", "across", "identify",
    "generate", "testable", "hypothesis", "approved", "instrument", "actions",
})


def _heuristic_keywords(text: str, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    seen: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= max_keywords:
            break
    defaults = ["nanoindentation", "hardness", "modulus", "microstructure"]
    for d in defaults:
        if d not in seen:
            seen.append(d)
    return seen[:max_keywords]


def _query_from_image_analysis(state: AgentState) -> str:
    ia = state.get("image_analysis") or {}
    parts: list[str] = []
    interp = (ia.get("interpretation") or "").strip()
    if interp:
        parts.append(interp[:400])
    for f in (ia.get("features_detected") or [])[:4]:
        parts.append(str(f))
    for t in (ia.get("target_locations") or [])[:3]:
        label = t.get("label") or t.get("reason")
        if label:
            parts.append(str(label))
    return " ".join(parts).strip()


def _build_literature_query_prompt(state: AgentState) -> str:
    parts = [
        "## Research question",
        (state.get("research_question") or "").strip() or "(none)",
        "## Experiment context",
        (state.get("experiment_context") or "").strip() or "(none)",
    ]
    ia_seed = _query_from_image_analysis(state)
    if ia_seed:
        parts.extend(["## Image analysis seed", ia_seed])
    if state.get("hypothesis"):
        parts.extend(["## Current hypothesis", state["hypothesis"][:500]])
    parts.append(
        'Respond with JSON only: {"literature_search_query": "<Crossref-style query>", '
        '"literature_search_keywords": ["term1", "term2", ...], "reasoning": "<brief>"}'
    )
    return "\n\n".join(parts)


def plan_literature_search(
    state: AgentState,
    cfg: Optional[AppConfig] = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Derive literature search query and keywords from experiment input.
    Uses LLM when HF token is available; falls back to heuristics.
    """
    cfg = cfg or get_config()
    existing = (state.get("literature_search_query") or "").strip()
    if existing and not force and not state.get("literature_refresh_requested"):
        return {}

    base = (
        (state.get("research_question") or "").strip()
        or (state.get("experiment_context") or "").strip()
        or _query_from_image_analysis(state)
    )

    query = ""
    keywords: list[str] = []
    reasoning = ""

    try:
        llm = get_llm_for_role("literature", cfg)
        response = llm.chat(
            user_message=_build_literature_query_prompt(state),
            system_message=(
                "You plan bibliographic literature searches for nanoindentation experiments. "
                "Extract searchable terms from the user's research question. Focus on: "
                "material type, mechanical properties (hardness, modulus), microstructure "
                "(grains, phases, boundaries), and measurement technique."
            ),
        )
        parsed = parse_llm_json(response)
        query = (parsed.get("literature_search_query") or "").strip()
        raw_kw = parsed.get("literature_search_keywords") or []
        if isinstance(raw_kw, list):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        reasoning = str(parsed.get("reasoning") or "")
    except Exception as exc:
        reasoning = f"heuristic fallback: {exc}"

    if not query:
        query = base[:300] if base else _DEFAULT_QUERY
    if not keywords:
        keywords = _heuristic_keywords(base or query)

    return {
        "literature_search_query": query,
        "literature_search_keywords": keywords,
        "literature_refresh_requested": False,
        "scratchpad": [f"literature query: {query[:100]}; keywords={', '.join(keywords[:5])}"],
    }
