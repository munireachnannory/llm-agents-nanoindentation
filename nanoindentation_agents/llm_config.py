"""
llm_config.py
=============
Minimal Hugging Face LLM client for nanoindentation agents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol

try:
    from huggingface_hub import InferenceClient
except ModuleNotFoundError:
    InferenceClient = None  # type: ignore[misc, assignment]

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3


class LLMClientLike(Protocol):
    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
    ) -> str: ...
    
    def chat_with_images(
        self,
        user_message: str,
        image_paths: list[str],
        system_message: Optional[str] = None,
    ) -> str: ...


def _api_key_from_env() -> Optional[str]:
    for key in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            from nanoindentation_agents.notebook_credentials import load_project_env

            load_project_env(env_path.parent)
        except Exception:
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
            except Exception:
                pass
        for key in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
            val = os.environ.get(key, "").strip()
            if val:
                return val
    return None


class LLMClient:
    """Text LLM via Hugging Face Inference API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or _api_key_from_env()
        if not self._api_key:
            raise ValueError("HF_TOKEN required for LLM calls")
        if InferenceClient is None:
            raise ModuleNotFoundError("Install huggingface_hub for LLM support")
        self._client = InferenceClient(model=model, token=self._api_key, timeout=120.0)

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def chat_with_images(
        self,
        user_message: str,
        image_paths: list[str],
        system_message: Optional[str] = None,
    ) -> str:
        """Chat with images using a vision-language model."""
        import base64
        
        # Build content with images
        content: list[dict] = []
        
        # Add images first
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                continue
            
            # Read and encode image
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine mime type
            suffix = img_path.suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/png")
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_data}"
                }
            })
        
        # Add text content
        content.append({
            "type": "text",
            "text": user_message
        })
        
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


def create_llm_client(
    *,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMClient:
    return LLMClient(model=model, max_tokens=max_tokens, temperature=temperature)
