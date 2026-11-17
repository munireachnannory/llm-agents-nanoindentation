"""
literature_agent.py
===================
Literature retrieval for nanoindentation experiments.
Searches: Crossref (peer-reviewed), arXiv, ChemRxiv, RSS feeds, and local files.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..rag.literature_retriever import LiteratureRetriever
from ..registry.models import get_llm_for_role
from ..state import AgentState
from .literature_search import _query_from_image_analysis, plan_literature_search
from .literature_manager_bridge import (
    format_candidates_for_context,
    search_literature_online,
    source_labels,
)


def _resolve_literature_query(state: AgentState) -> str:
    for key in ("literature_search_query", "research_question", "experiment_context", "hypothesis"):
        q = (state.get(key) or "").strip()
        if q and not q.startswith("[LLM Error"):
            return q
    ia_query = _query_from_image_analysis(state)
    return ia_query or "nanoindentation hardness modulus microstructure"


def literature_node(state: AgentState) -> dict[str, Any]:
    """Search literature from online sources (Crossref, arXiv, ChemRxiv) and local files."""
    cfg = get_config()
    lit_cfg = cfg.literature

    print("[literature] Starting literature search...")
    print(f"[literature] Backend: {lit_cfg.backend} | use_manager: {lit_cfg.use_manager}")
    
    plan_error = ""
    try:
        plan_updates = plan_literature_search(state, cfg)
        print(f"[literature] Query planned: {plan_updates.get('literature_search_query', '')[:80]}")
    except Exception as exc:
        plan_updates = {}
        plan_error = str(exc)
        print(f"[literature] Query planning fallback: {plan_error}")

    merged = {**state, **plan_updates}
    query = _resolve_literature_query(merged)
    keywords = list(merged.get("literature_search_keywords") or [])
    
    print(f"[literature] Final query: {query[:100]}")
    print(f"[literature] Keywords: {keywords[:5]}")

    raw_context = ""
    sources: list[str] = []
    search_meta: dict[str, Any] = {"query": query}
    if plan_error:
        search_meta["plan_error"] = plan_error

    # Try online search first if enabled
    if lit_cfg.use_manager or lit_cfg.backend == "manager":
        print("[literature] Searching online sources (Crossref, arXiv, ChemRxiv, RSS)...")
        try:
            candidates, online_meta = search_literature_online(query, lit_cfg, keywords=keywords)
            if candidates:
                raw_context = format_candidates_for_context(candidates, limit=lit_cfg.max_chunks)
                sources = source_labels(candidates)
                search_meta = {**search_meta, "mode": "online", **online_meta}
                print(f"[literature] Online search found {len(candidates)} papers")
            else:
                print("[literature] No online results, falling back to local files")
        except Exception as exc:
            print(f"[literature] Online search error: {exc}")
            search_meta["online_error"] = str(exc)

    # Fallback to local files if no online results or online not enabled
    if not raw_context.strip() and lit_cfg.fallback_to_local_files:
        print("[literature] Searching local ./literature/ files...")
        retriever = LiteratureRetriever(lit_cfg)
        local_context, local_sources = retriever.retrieve(query, keywords=keywords)
        raw_context = local_context
        sources = local_sources
        search_meta["mode"] = "local_files"
        print(f"[literature] Local search found {len(local_sources)} files")
    
    if not sources:
        print("[literature] WARNING: No literature found (online or local)")
        raw_context = (
            f"No literature sources found for query: {query}\n"
            "Check network connectivity for online search, or add .txt/.md files to ./literature/"
        )

    # Summarize retrieved content with LLM
    summary = ""
    try:
        llm = get_llm_for_role("literature", cfg)
        summary = llm.chat(
            user_message=(
                f"Search query: {query}\n"
                f"Keywords: {', '.join(keywords)}\n"
                f"Sources: {len(sources)} ({search_meta.get('mode', 'unknown')})\n\n"
                f"Retrieved literature:\n{raw_context[:6000]}\n\n"
                "Summarize findings relevant to nanoindentation experiment design: "
                "expected hardness/modulus ranges, spacing rules, phase/grain effects, "
                "and what measurements would discriminate competing hypotheses. "
                "Cite specific papers when possible."
            ),
            system_message=(
                "You are a materials science literature assistant for nanoindentation. "
                "Be concise and cite which paper/source each finding comes from."
            ),
        )
        print(f"[literature] Generated summary ({len(summary)} chars) → will pass to hypothesis agent")
    except Exception as exc:
        summary = (
            f"Literature search completed (query: {query}; sources: {len(sources)}). "
            f"LLM summary unavailable ({exc}). Raw context:\n{raw_context[:2000]}"
        )
        print(f"[literature] LLM summary unavailable ({exc}) - using raw context")

    # Build scratchpad entry with search details
    mode = search_meta.get("mode", "unknown")
    scratchpad_entry = (
        f"literature: mode={mode} query={query[:60]} "
        f"sources={len(sources)}"
    )
    if search_meta.get("crossref_count"):
        scratchpad_entry += f" crossref={search_meta['crossref_count']}"
    if search_meta.get("arxiv_count"):
        scratchpad_entry += f" arxiv={search_meta['arxiv_count']}"
    if search_meta.get("chemrxiv_count"):
        scratchpad_entry += f" chemrxiv={search_meta['chemrxiv_count']}"

    return {
        "literature_search_query": query,
        "literature_search_keywords": keywords,
        "literature_context": summary,
        "literature_sources": sources,
        "literature_raw_context": raw_context,
        "literature_search_meta": search_meta,
        "literature_refresh_requested": False,
        "current_stage": "literature_search",
        "scratchpad": [scratchpad_entry],
    }
