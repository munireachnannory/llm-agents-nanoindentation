"""
literature_retriever.py
=======================
Local literature retrieval from ./literature text files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import LiteratureSettings


class LiteratureRetriever:
    def __init__(self, settings: LiteratureSettings):
        self.settings = settings
        root = Path(__file__).resolve().parent.parent.parent
        p = Path(settings.local_dir)
        self.local_dir = p if p.is_absolute() else root / p

    def retrieve(self, query: str, keywords: list[str] | None = None) -> tuple[str, list[str]]:
        search_text = " ".join([query, *(keywords or [])]).strip()
        chunks: list[tuple[str, str, int]] = []
        sources: list[str] = []

        if self.local_dir.exists():
            for path in sorted(self.local_dir.glob("**/*")):
                if path.suffix.lower() not in (".txt", ".md", ".rst"):
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                score = _relevance_score(search_text, text)
                chunks.append((str(path), _excerpt(text, 1200), score))
                sources.append(str(path.name))

        if not chunks:
            return (
                f"No .txt/.md files under {self.local_dir}. "
                "Add nanoindentation primers there.",
                [],
            )

        chunks.sort(key=lambda c: c[2], reverse=True)
        if search_text and chunks[0][2] > 0:
            selected = [c for c in chunks if c[2] > 0][: self.settings.max_chunks]
        else:
            selected = chunks[: self.settings.max_chunks]
        if not selected:
            selected = chunks[: self.settings.max_chunks]

        context = "\n\n---\n\n".join(f"[{src}]\n{body}" for src, body, _ in selected)
        return context, [src for src, _, _ in selected]


def _relevance_score(query: str, text: str) -> int:
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q_tokens & t_tokens)


def _excerpt(text: str, max_len: int) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
