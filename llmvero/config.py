from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-6.1-terra",
    "gemini": "gemini-2.5-flash",
    "openai_compatible": "",
    "mock": "mock-1",
}

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai_compatible": "",
    "mock": "",
}


class Settings(BaseSettings):
    """Settings come from environment variables or a `.env` file. Only PROVIDER and API_KEY are needed."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The model that writes the training data.
    provider: str = "anthropic"  # anthropic, openai, gemini, openai_compatible or mock
    model: str = ""  # empty = the provider's default (DEFAULT_MODELS)
    api_key: str = ""
    base_url: str = ""  # only for openai_compatible (Ollama, vLLM, Groq...) or a custom endpoint

    # Optional.
    access_key: str = ""  # when set, callers must send "Authorization: Bearer <access_key>"
    system_prompt: str = ""  # system message stored in every training example

    # Rarely needed; the defaults work.
    llmvero_data_dir: Path = Path("./data")
    llmvero_cors_origins: str = "http://localhost:3000"
    llmvero_max_concurrency: int = Field(4, ge=1, le=64)
    llmvero_max_document_chars: int = 200_000
    llmvero_webhook_secret: str = ""
    llmvero_request_timeout: float = 120.0
    llmvero_max_retries: int = 3

    # Extra providers, only if one request should use a different provider than PROVIDER.
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    gemini_api_key: str = ""
    gemini_model: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""
    openai_compatible_base_url: str = ""

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.llmvero_cors_origins.split(",") if o.strip()]

    @property
    def db_path(self) -> Path:
        return self.llmvero_data_dir / "llmvero.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
