from __future__ import annotations

from .base import Completion, Provider


class GeminiProvider(Provider):
    """Google Gemini through the Generative Language API."""

    name = "gemini"

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
        generation: dict = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            generation["temperature"] = temperature
        if json_mode:
            generation["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        url = f"{self.base_url}/models/{model or self.model}:generateContent"
        data = await self._post(url, headers=headers, body=body)
        candidates = data.get("candidates") or [{}]
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )
