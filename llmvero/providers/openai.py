from __future__ import annotations

from .base import Completion, Provider


class OpenAIProvider(Provider):
    """OpenAI Chat Completions API."""

    name = "openai"
    # OpenAI's own API takes max_completion_tokens and JSON mode; many compatible servers don't.
    use_max_completion_tokens = True
    supports_json_mode = True

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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        body["max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if json_mode and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        data = await self._post(f"{self.base_url}/chat/completions", headers=headers, body=body)
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """Any server that speaks the OpenAI Chat Completions protocol (Ollama, vLLM, Groq, OpenRouter, ...)."""

    name = "openai_compatible"
    use_max_completion_tokens = False
    supports_json_mode = False
