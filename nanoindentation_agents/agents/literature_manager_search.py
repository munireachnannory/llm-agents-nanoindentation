"""
literature_manager_search.py
============================
Crossref + RSS + arXiv literature search for nanoindentation research.
"""

from __future__ import annotations

import html as ihtml
import re
import time
from typing import Any, Optional

import requests

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

CROSSREF_ROWS_PER_PAGE = 50
CROSSREF_MAX_PAGES = 4
REQUEST_TIMEOUT = 45
SLEEP_BETWEEN_REQUESTS = 0.2
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Nanoindentation-Literature-Agent)",
    "Accept-Language": "en-US,en;q=0.9",
}

# RSS feeds for materials science journals
RSS_FEEDS = [
    # arXiv materials science
    "https://rss.arxiv.org/rss/cond-mat.mtrl-sci",
    # ChemRxiv - materials
    "https://chemrxiv.org/engage/chemrxiv/public-api/v1/rss/subject-area/materials-science",
    # Journal of Materials Research
    "https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=43578&channel-name=Journal+of+Materials+Research",
    # Acta Materialia (ScienceDirect)
    "https://rss.sciencedirect.com/publication/science/13596454",
    # Materials Science and Engineering A
    "https://rss.sciencedirect.com/publication/science/09215093",
]

TITLE_BLOCK_RE = re.compile(
    r"(^|\s)(corrigendum|erratum|editorial|cover art|front cover|table of contents)(\s|$)",
    re.I,
)

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "into", "about", "using", "via",
    "between", "over", "under", "after", "before", "than", "then", "also", "can",
    "may", "our", "their", "they", "them", "its", "how", "what", "when", "where",
    "which", "while", "during", "through", "among", "within", "without",
})


def strip_html_text(value: Any) -> str:
    if not value:
        return ""
    text = ihtml.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_terms(query: str, keywords: Optional[list[str]] = None) -> list[str]:
    """Build deduplicated search terms from query + optional keyword list."""
    raw: list[str] = []
    for source in (keywords or []) + [query]:
        raw.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", str(source).lower()))
    seen: set[str] = set()
    terms: list[str] = []
    for term in raw:
        if len(term) <= 2 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def is_allowed_crossref_item(item: dict[str, Any]) -> bool:
    title = " ".join(item.get("title", [])).strip()
    if not title or TITLE_BLOCK_RE.search(title):
        return False
    doc_type = item.get("type")
    if doc_type in ("journal-issue", "proceedings-series"):
        return False
    return True


def crossref_search(
    query: str,
    year_min: int,
    year_max: int,
    rows_per_page: int = CROSSREF_ROWS_PER_PAGE,
    max_pages: int = CROSSREF_MAX_PAGES,
) -> list[dict[str, Any]]:
    """Search Crossref API for scholarly articles."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    
    for page in range(max_pages):
        offset = page * rows_per_page
        url = (
            "https://api.crossref.org/works?"
            f"query.bibliographic={requests.utils.quote(query)}"
            f"&rows={rows_per_page}&offset={offset}"
            f"&filter=from-pub-date:{year_min}-01-01,until-pub-date:{year_max}-12-31"
        )
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
        except Exception:
            break
            
        if not items:
            break
            
        for item in items:
            if not is_allowed_crossref_item(item):
                continue
            title = " ".join(item.get("title", [])).strip() or None
            doi = item.get("DOI")
            key = doi or title
            if not key or key in seen:
                continue
            seen.add(key)
            
            journal = " ".join(item.get("container-title", [])).strip() or None
            abstract = strip_html_text(item.get("abstract"))
            resource_url = item.get("URL")
            
            year = None
            if item.get("published-print", {}).get("date-parts"):
                year = item["published-print"]["date-parts"][0][0]
            elif item.get("published-online", {}).get("date-parts"):
                year = item["published-online"]["date-parts"][0][0]
                
            author = None
            if item.get("author"):
                first = item["author"][0]
                author = " ".join([first.get("given", ""), first.get("family", "")]).strip() or None
                
            out.append({
                "title": title,
                "doi": doi,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "lead_author": author,
                "source": "crossref",
                "landing_page": resource_url,
            })
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return out


def arxiv_search(
    query: str,
    year_min: int,
    year_max: int,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search arXiv API for preprints."""
    # arXiv search query - focus on materials science category
    search_query = f"all:{requests.utils.quote(query)}+AND+cat:cond-mat.mtrl-sci"
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={search_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
    except Exception:
        return out
    
    if not HAS_FEEDPARSER:
        return out
        
    feed = feedparser.parse(r.text)
    
    for entry in getattr(feed, "entries", []):
        title = getattr(entry, "title", "").replace("\n", " ").strip()
        if not title or TITLE_BLOCK_RE.search(title):
            continue
            
        arxiv_id = getattr(entry, "id", "")
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        
        # Extract year from published date
        year = None
        published = getattr(entry, "published", "")
        if published:
            match = re.search(r"(\d{4})", published)
            if match:
                year = int(match.group(1))
                if year < year_min or year > year_max:
                    continue
        
        summary = strip_html_text(getattr(entry, "summary", ""))
        
        # Get first author
        authors = getattr(entry, "authors", [])
        lead_author = authors[0].get("name", "") if authors else None
        
        out.append({
            "title": title,
            "doi": None,
            "arxiv_id": arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id,
            "journal": "arXiv preprint",
            "year": year,
            "abstract": summary,
            "lead_author": lead_author,
            "source": "arxiv",
            "landing_page": arxiv_id,
        })
    
    return out


def chemrxiv_search(
    query: str,
    year_min: int,
    year_max: int,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """Search ChemRxiv API for chemistry/materials preprints."""
    url = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
    params = {
        "term": query,
        "limit": max_results,
        "searchDateFrom": f"{year_min}-01-01",
        "searchDateTo": f"{year_max}-12-31",
    }
    
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return out
    
    for item in data.get("itemHits", []):
        item_data = item.get("item", {})
        title = item_data.get("title", "").strip()
        if not title or TITLE_BLOCK_RE.search(title):
            continue
            
        item_id = item_data.get("id", "")
        if item_id in seen:
            continue
        seen.add(item_id)
        
        doi = item_data.get("doi")
        abstract = strip_html_text(item_data.get("abstract", ""))
        
        # Get year from posted date
        year = None
        posted = item_data.get("statusDate", "")
        if posted:
            match = re.search(r"(\d{4})", posted)
            if match:
                year = int(match.group(1))
        
        # Get first author
        authors = item_data.get("authors", [])
        lead_author = None
        if authors:
            first = authors[0]
            lead_author = f"{first.get('firstName', '')} {first.get('lastName', '')}".strip()
        
        out.append({
            "title": title,
            "doi": doi,
            "journal": "ChemRxiv preprint",
            "year": year,
            "abstract": abstract,
            "lead_author": lead_author,
            "source": "chemrxiv",
            "landing_page": f"https://chemrxiv.org/engage/chemrxiv/article-details/{item_id}",
        })
    
    return out


def rss_search(
    query: str,
    year_min: int,
    year_max: int,
    keywords: Optional[list[str]] = None,
    *,
    min_term_matches: Optional[int] = None,
    feeds: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Match RSS feed entries using terms from query and keywords."""
    if not HAS_FEEDPARSER:
        return []

    terms = query_terms(query, keywords)
    if not terms:
        return []

    if min_term_matches is None:
        min_term_matches = 2 if len(terms) >= 2 else 1

    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for feed_url in feeds or RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            continue
            
        for entry in getattr(feed, "entries", []):
            title = getattr(entry, "title", None) or ""
            summary = getattr(entry, "summary", None) or ""
            blob = f"{title} {summary}".lower()
            
            if TITLE_BLOCK_RE.search(title):
                continue
            if "review" in blob and "review" not in terms:
                continue
                
            matches = sum(1 for term in terms if term in blob)
            if matches < min_term_matches:
                continue
                
            year = None
            for attr in ("published_parsed", "updated_parsed"):
                parsed = getattr(entry, attr, None)
                if parsed:
                    year = parsed[0]
                    break
            if year is not None and (year < year_min or year > year_max):
                continue
                
            link = getattr(entry, "link", None)
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)
            
            items.append({
                "title": strip_html_text(title),
                "doi": None,
                "journal": getattr(feed.feed, "title", None),
                "year": year,
                "abstract": strip_html_text(summary),
                "lead_author": None,
                "source": "rss",
                "landing_page": link,
                "rss_match_terms": matches,
            })
    return items


def merge_candidates(
    *candidate_lists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge multiple candidate lists, preferring earlier lists on duplicates."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    
    for candidates in candidate_lists:
        for item in candidates:
            key = (item.get("doi") or "").lower() or (item.get("title") or "").lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out
