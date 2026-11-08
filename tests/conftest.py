import pytest
from fastapi.testclient import TestClient

from llmvero.config import Settings
from llmvero.db import Store
from llmvero.main import create_app

ARTICLE = (
    "Acme Cloud stores backups for 30 days. Customers on the Pro plan can extend retention to 1 year.\n\n"
    "Backups are encrypted with AES-256 at rest. Restores can be started from the dashboard or the CLI.\n\n"
    "Support is available 24/7 by chat. Enterprise customers also get a dedicated account manager."
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llmvero_data_dir=tmp_path,
        provider="mock",
        access_key="",
        llmvero_max_retries=0,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings, store=Store(":memory:"))
    with TestClient(app) as c:
        yield c
