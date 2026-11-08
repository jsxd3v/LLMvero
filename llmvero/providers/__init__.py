"""Provider registry: which model APIs exist and which ones have keys in .env."""

from __future__ import annotations

import httpx

from ..config import DEFAULT_BASE_URLS, DEFAULT_MODELS, Settings
from .anthropic import AnthropicProvider
from .base import Completion, Provider, ProviderError
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai import OpenAICompatibleProvider, OpenAIProvider

__all__ = ["Completion", "Provider", "ProviderError", "build_provider", "provider_settings", "PROVIDERS"]

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "mock": MockProvider,
}


def provider_settings(name: str, s: Settings) -> tuple[str, str, str, bool]:
    """Return (api_key, model, base_url, configured) for a provider.

    The main provider uses API_KEY / MODEL / BASE_URL. Others use their own <NAME>_API_KEY etc. if set.
    """
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    if name == "mock":
        return "", DEFAULT_MODELS["mock"], "", True
    own_key = getattr(s, f"{name}_api_key", "")
    own_model = getattr(s, f"{name}_model", "")
    own_base = getattr(s, f"{name}_base_url", "")
    if name == s.provider:
        api_key, model, base_url = s.api_key or own_key, s.model or own_model, s.base_url or own_base
    else:
        api_key, model, base_url = own_key, own_model, own_base
    model = model or DEFAULT_MODELS[name]
    base_url = base_url or DEFAULT_BASE_URLS[name]
    configured = bool(base_url and model) if name == "openai_compatible" else bool(api_key)
    return api_key, model, base_url, configured


def build_provider(
    name: str | None,
    settings: Settings,
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Provider:
    name = name or settings.provider
    api_key, default_model, base_url, configured = provider_settings(name, settings)
    if not configured:
        raise ValueError(f"provider '{name}' is not configured: set API_KEY (and MODEL) in .env")
    return PROVIDERS[name](
        api_key=api_key,
        model=model or default_model,
        base_url=base_url,
        timeout=settings.llmvero_request_timeout,
        max_retries=settings.llmvero_max_retries,
        client=client,
    )
