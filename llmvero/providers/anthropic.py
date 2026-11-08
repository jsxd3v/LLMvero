from __future__ import annotations

from .base import Completion, Provider


class AnthropicProvider(Provider):
    """Claude models through the Anthropic Messages API."""

    name = "anthropic"

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> Completion:
        body: dict = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = await self._post(f"{self.base_url}/v1/messages", headers=headers, body=body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
