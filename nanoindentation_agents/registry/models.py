"""
registry/models.py
==================
LLM client factory per configured role.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import AppConfig, get_config
from ..llm_config import LLMClient, LLMClientLike, create_llm_client


def clear_llm_client_cache() -> None:
    _client_cache.cache_clear()


@lru_cache(maxsize=32)
def _client_cache(
    role: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMClientLike:
    return create_llm_client(model=model, max_tokens=max_tokens, temperature=temperature)


def get_llm_for_role(role: str, cfg: Optional[AppConfig] = None) -> LLMClientLike:
    cfg = cfg or get_config()
    spec = cfg.models.role(role)
    max_tokens = spec.max_tokens or cfg.models.default_max_tokens
    return _client_cache(role, spec.model, spec.temperature, max_tokens)
