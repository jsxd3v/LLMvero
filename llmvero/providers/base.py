from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class Provider(ABC):
    """A chat model behind an HTTP API. Subclasses only build the request and read the response."""

    name: str = "base"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> Completion: ...

    async def _post(self, url: str, *, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on rate limits, overloads and network errors."""
        attempt = 0
        while True:
            try:
                resp = await self._client.post(url, headers=headers, json=body)
            except httpx.TransportError as exc:
                error = ProviderError(f"{self.name}: network error: {exc}", retryable=True)
                retry_after = None
            else:
                if resp.status_code < 400:
                    return resp.json()
                error = ProviderError(
                    f"{self.name}: HTTP {resp.status_code}: {_error_message(resp)}",
                    status=resp.status_code,
                    retryable=resp.status_code in RETRYABLE_STATUS,
                )
                retry_after = _retry_after(resp)

            if not error.retryable or attempt >= self.max_retries:
                raise error
            delay = retry_after if retry_after is not None else min(2**attempt, 30) + random.random()
            await asyncio.sleep(delay)
            attempt += 1


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        return str(err)[:300]
    return str(data)[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return min(float(value), 60.0)
    except ValueError:
        return None
