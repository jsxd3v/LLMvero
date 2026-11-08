import json

import httpx
import pytest

from llmvero.config import Settings
from llmvero.providers import ProviderError, build_provider


def make(name: str, handler, **env):
    settings = Settings(_env_file=None, llmvero_max_retries=2, **env)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return build_provider(name, settings, client=client)


@pytest.mark.anyio
async def test_anthropic_request_and_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"examples": []}'}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    p = make("anthropic", handler, anthropic_api_key="sk-test", anthropic_model="claude-test")
    out = await p.complete(system="sys", prompt="hi", max_tokens=100)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["body"]["system"] == "sys" and seen["body"]["model"] == "claude-test"
    assert "temperature" not in seen["body"]
    assert out.text == '{"examples": []}' and out.input_tokens == 10 and out.output_tokens == 5


@pytest.mark.anyio
async def test_openai_uses_json_mode_and_bearer():
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    p = make("openai", handler, openai_api_key="sk-o")
    out = await p.complete(system="sys", prompt="hi", temperature=0.2)
    assert seen["headers"]["authorization"] == "Bearer sk-o"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["max_completion_tokens"] == 4096 and seen["body"]["temperature"] == 0.2
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert out.input_tokens == 3


@pytest.mark.anyio
async def test_openai_compatible_points_at_custom_server():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    p = make(
        "openai_compatible",
        handler,
        openai_compatible_base_url="http://localhost:11434/v1",
        openai_compatible_model="llama3.1",
    )
    await p.complete(system="s", prompt="p")
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 4096 and "response_format" not in seen["body"]


@pytest.mark.anyio
async def test_gemini_request_and_skips_thoughts():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "{}"}]}}
                ],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
            },
        )

    p = make("gemini", handler, gemini_api_key="g-key", gemini_model="gemini-test")
    out = await p.complete(system="sys", prompt="hi")
    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "g-key"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert out.text == "{}" and out.output_tokens == 2


@pytest.mark.anyio
async def test_retries_rate_limits_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def no_sleep(_):
        return None

    monkeypatch.setattr("llmvero.providers.base.asyncio.sleep", no_sleep)

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    p = make("anthropic", handler, anthropic_api_key="k")
    assert (await p.complete(system="s", prompt="p")).text == "ok"
    assert calls["n"] == 3


@pytest.mark.anyio
async def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    p = make("anthropic", handler, anthropic_api_key="bad")
    with pytest.raises(ProviderError, match="invalid x-api-key"):
        await p.complete(system="s", prompt="p")
    assert calls["n"] == 1


def test_unconfigured_provider_is_rejected():
    with pytest.raises(ValueError, match="not configured"):
        build_provider("openai", Settings(_env_file=None, openai_api_key=""))


def test_simple_env_provider_model_api_key():
    from llmvero.providers import provider_settings

    s = Settings(_env_file=None, provider="openai", api_key="sk-main", model="")
    assert provider_settings("openai", s) == ("sk-main", "gpt-6.1-terra", "https://api.openai.com/v1", True)
    s = Settings(_env_file=None, provider="anthropic", api_key="sk-a", model="claude-haiku-5-5")
    assert provider_settings("anthropic", s)[:2] == ("sk-a", "claude-haiku-5-5")
    assert provider_settings("gemini", s)[3] is False  # API_KEY belongs to PROVIDER only
    s = Settings(
        _env_file=None, provider="openai_compatible", base_url="http://localhost:11434/v1", model="llama3.1"
    )
    assert provider_settings("openai_compatible", s)[3] is True
