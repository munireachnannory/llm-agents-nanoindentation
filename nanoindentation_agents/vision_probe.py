"""
vision_probe.py
===============
Check which Hugging Face vision models are usable with your token.
"""

from __future__ import annotations

from typing import Any, Optional

from .notebook_credentials import resolve_hf_token

_PROBE_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

DEFAULT_VISION_CANDIDATES = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]


def probe_hf_model(
    model_id: str,
    *,
    hf_token: Optional[str] = None,
    vision: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Test one model on the Hugging Face Inference router."""
    token = (hf_token or resolve_hf_token() or "").strip()
    row: dict[str, Any] = {
        "model": model_id,
        "vision": vision,
        "status": "skip",
        "detail": "",
    }
    if not token:
        row["status"] = "auth"
        row["detail"] = "No HF token"
        return row

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        row["status"] = "error"
        row["detail"] = "huggingface_hub not installed"
        return row

    client = InferenceClient(token=token, timeout=timeout_s)
    try:
        if vision:
            client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _PROBE_IMAGE_DATA_URL}},
                        {"type": "text", "text": "Reply with OK."},
                    ],
                }],
                max_tokens=8,
            )
        else:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=8,
            )
        row["status"] = "ok"
        row["detail"] = "Probe succeeded"
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        low = msg.lower()
        if "401" in msg or "unauthorized" in low:
            row["status"] = "auth"
        elif "not supported" in low or "model_not_supported" in low:
            row["status"] = "unsupported"
        else:
            row["status"] = "error"
        row["detail"] = msg[:200]
    return row


def probe_vision_models(
    *,
    hf_token: Optional[str] = None,
    models: Optional[list[str]] = None,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Probe a list of vision models."""
    candidates = models or DEFAULT_VISION_CANDIDATES
    return [
        probe_hf_model(m, hf_token=hf_token, vision=True, timeout_s=timeout_s)
        for m in candidates
    ]


def print_vision_report(rows: list[dict[str, Any]]) -> None:
    """Print a compact table of probe results."""
    print(f"{'Model':<42} {'Status':<12} Detail")
    print("-" * 90)
    for row in rows:
        print(f"{row['model']:<42} {row['status']:<12} {row.get('detail', '')[:40]}")
