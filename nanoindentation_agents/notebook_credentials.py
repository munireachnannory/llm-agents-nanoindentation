"""
notebook_credentials.py
=====================
Resolve Hugging Face credentials for notebook LLM / vision calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AppConfig
else:
    AppConfig = object

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


def _apply_env_file(env_path: Path) -> None:
    """Set env vars from file when missing or blank (never print values)."""
    try:
        from dotenv import dotenv_values

        for key, val in dotenv_values(env_path).items():
            if not key or val is None:
                continue
            val = str(val).strip()
            if val and not os.environ.get(key, "").strip():
                os.environ[key] = val
        return
    except ImportError:
        pass

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and val and not os.environ.get(key, "").strip():
            os.environ[key] = val


def load_project_env(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Load ``.env`` from the repo root into ``os.environ``.

    Returns the path when a file was found and loaded, else ``None``.
    Safe to call multiple times. Fills keys that are missing or empty only.
    """
    env_path = (repo_root or _REPO_ROOT) / ".env"
    if not env_path.is_file():
        return None

    _apply_env_file(env_path)
    return env_path


def resolve_instrument_token() -> str:
    """Instrument auth token from env / ``.env`` (after ``load_project_env``)."""
    load_project_env()
    return (
        os.environ.get("NANOINDENT_TOKEN", "").strip()
        or os.environ.get("INSTRUMENT_TOKEN", "").strip()
    )


def _token_from_notebook_globals() -> Optional[str]:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return None
        ns = ip.user_ns
    except Exception:
        return None

    raw = ns.get("HF_TOKEN")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def resolve_hf_token() -> Optional[str]:
    """HF token from env vars, .env, or notebook ``HF_TOKEN`` global."""
    load_project_env()
    for key in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val

    return _token_from_notebook_globals()


def sync_notebook_hf_credentials() -> bool:
    """Copy HF credentials into os.environ. Returns True when a token is available."""
    token = resolve_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    try:
        from .registry.models import clear_llm_client_cache

        clear_llm_client_cache()
    except Exception:
        pass
    return bool(token)


def print_llm_backend_summary(cfg: AppConfig) -> None:
    """One-line status for notebook driver startup."""
    token_ok = bool(resolve_hf_token())
    print(f"  HF Inference API: {'ready' if token_ok else 'NO TOKEN — set HF_TOKEN'}")
    try:
        spec = cfg.models.role("image_analysis")
        print(f"  Vision LLM: {spec.model} (vision={spec.vision}, provider={spec.provider})")
    except Exception as exc:
        print(f"  Vision LLM: (config error: {exc})")
