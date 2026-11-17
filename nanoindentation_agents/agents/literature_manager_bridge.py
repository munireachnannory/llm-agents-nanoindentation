"""
literature_manager_bridge.py
============================
Adapter from LangGraph state to literature search (Crossref, arXiv, ChemRxiv, RSS).
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import LiteratureSettings
from . import literature_manager_search as search


def search_literature_online(
    query: str,
    lit_cfg: LiteratureSettings,
    keywords: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Search multiple online sources: Crossref, arXiv, ChemRxiv, and RSS feeds.
    
    Returns
    -------
    tuple[list[dict], dict]
        (merged_candidates, metadata_dict)
    """
    year_min = lit_cfg.year_min
    year_max = lit_cfg.year_max
    errors: list[str] = []
    
    # Crossref search (peer-reviewed publications)
    crossref_results: list[dict[str, Any]] = []
    try:
        crossref_results = search.crossref_search(
            query,
            year_min,
            year_max,
            rows_per_page=search.CROSSREF_ROWS_PER_PAGE,
            max_pages=min(search.CROSSREF_MAX_PAGES, 3),
        )
        print(f"[literature] Crossref: {len(crossref_results)} results")
    except Exception as exc:
        errors.append(f"crossref: {type(exc).__name__}: {exc}")
        print(f"[literature] Crossref error: {exc}")
    
    # arXiv search (preprints)
    arxiv_results: list[dict[str, Any]] = []
    try:
        arxiv_results = search.arxiv_search(query, year_min, year_max, max_results=15)
        print(f"[literature] arXiv: {len(arxiv_results)} results")
    except Exception as exc:
        errors.append(f"arxiv: {type(exc).__name__}: {exc}")
        print(f"[literature] arXiv error: {exc}")
    
    # ChemRxiv search (chemistry/materials preprints)
    chemrxiv_results: list[dict[str, Any]] = []
    try:
        chemrxiv_results = search.chemrxiv_search(query, year_min, year_max, max_results=10)
        print(f"[literature] ChemRxiv: {len(chemrxiv_results)} results")
    except Exception as exc:
        errors.append(f"chemrxiv: {type(exc).__name__}: {exc}")
        print(f"[literature] ChemRxiv error: {exc}")
    
    # RSS feeds (journal TOCs)
    rss_results: list[dict[str, Any]] = []
    rss_kwargs: dict[str, Any] = {"keywords": keywords}
    if lit_cfg.rss_feeds:
        rss_kwargs["feeds"] = lit_cfg.rss_feeds
    try:
        rss_results = search.rss_search(query, year_min, year_max, **rss_kwargs)
        print(f"[literature] RSS feeds: {len(rss_results)} results")
    except Exception as exc:
        errors.append(f"rss: {type(exc).__name__}: {exc}")
        print(f"[literature] RSS error: {exc}")
    
    # Merge all results (Crossref preferred, then arXiv, ChemRxiv, RSS)
    merged = search.merge_candidates(crossref_results, arxiv_results, chemrxiv_results, rss_results)
    
    meta = {
        "query": query,
        "keywords": keywords or [],
        "year_min": year_min,
        "year_max": year_max,
        "crossref_count": len(crossref_results),
        "arxiv_count": len(arxiv_results),
        "chemrxiv_count": len(chemrxiv_results),
        "rss_count": len(rss_results),
        "merged_count": len(merged),
        "search_terms": search.query_terms(query, keywords),
        "errors": errors,
    }
    
    return merged[: lit_cfg.manager_max_candidates], meta


def format_candidates_for_context(candidates: list[dict[str, Any]], limit: int = 10) -> str:
    """Format search results as text for LLM context."""
    blocks: list[str] = []
    
    for i, c in enumerate(candidates[:limit], 1):
        title = c.get("title") or "Untitled"
        doi = c.get("doi") or c.get("arxiv_id") or ""
        year = c.get("year") or ""
        journal = c.get("journal") or ""
        source = c.get("source") or ""
        abstract = (c.get("abstract") or "").strip()
        
        if len(abstract) > 1200:
            abstract = abstract[:1200] + "..."
            
        id_line = f"DOI: {doi}" if doi else ""
        if c.get("arxiv_id"):
            id_line = f"arXiv: {c['arxiv_id']}"
            
        blocks.append(
            f"[{i}] {title}\n"
            f"    {id_line} | Year: {year} | Journal: {journal} | Source: {source}\n"
            f"    Abstract: {abstract or '(none)'}"
        )
    
    if not blocks:
        return "(No online literature results found)"
        
    return "\n\n".join(blocks)


def source_labels(candidates: list[dict[str, Any]]) -> list[str]:
    """Extract unique source labels for citation."""
    labels: list[str] = []
    seen: set[str] = set()
    
    for c in candidates:
        doi = (c.get("doi") or "").strip()
        arxiv_id = (c.get("arxiv_id") or "").strip()
        title = (c.get("title") or "").strip()
        year = c.get("year") or ""
        
        if doi:
            label = doi
        elif arxiv_id:
            label = f"arXiv:{arxiv_id}"
        elif title and year:
            label = f"{title[:50]} ({year})"
        else:
            label = title[:60] if title else ""
            
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
            
    return labels
