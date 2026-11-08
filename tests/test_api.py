import json
import time

from fastapi.testclient import TestClient

from llmvero.db import Store
from llmvero.main import create_app

from .conftest import ARTICLE


def wait_for(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def make_dataset(client, docs=None):
    ds = client.post("/v1/datasets", json={"name": "Help center"}).json()
    docs = docs or [{"text": ARTICLE, "external_id": "article-1", "metadata": {"lang": "en"}}]
    r = client.post(f"/v1/datasets/{ds['id']}/documents", json={"documents": docs})
    assert r.status_code == 200, r.text
    return ds["id"]


def test_health_and_providers(client):
    assert client.get("/health").json()["status"] == "ok"
    providers = {p["name"]: p for p in client.get("/v1/providers").json()}
    assert providers["mock"]["configured"] and providers["mock"]["default"]
    assert not providers["anthropic"]["configured"]
    assert "sharegpt" in client.get("/v1/formats").json()


def test_documents_are_deduplicated(client):
    ds_id = make_dataset(client)
    r = client.post(
        f"/v1/datasets/{ds_id}/documents",
        json={"documents": [{"text": ARTICLE}, {"messages": [{"role": "user", "content": "hi"}]}]},
    )
    assert r.json() == {"inserted": 1, "skipped_duplicates": 1, "document_count": 2}


def test_document_needs_content(client):
    ds_id = make_dataset(client)
    r = client.post(f"/v1/datasets/{ds_id}/documents", json={"documents": [{"text": "  "}]})
    assert r.status_code == 422


def test_full_job_flow_and_exports(client):
    ds_id = make_dataset(client)
    r = client.post(
        f"/v1/datasets/{ds_id}/jobs",
        json={
            "task": "qa",
            "examples_per_chunk": 4,
            "system_prompt": "You are Acme support.",
            "quality_check": True,
        },
    )
    assert r.status_code == 202, r.text
    job = wait_for(client, r.json()["id"])
    assert job["status"] == "completed", job
    assert job["example_count"] == 4 and job["done_chunks"] == 1
    assert job["input_tokens"] > 0

    page = client.get(f"/v1/jobs/{job['id']}/examples").json()
    assert page["total"] == 4
    first = page["items"][0]
    assert first["system"] == "You are Acme support." and first["score"] == 5
    assert first["metadata"]["external_id"] == "article-1" and first["metadata"]["lang"] == "en"

    lines = client.get(f"/v1/jobs/{job['id']}/export?format=openai").text.strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["messages"][0]["role"] == "system"

    gemini = json.loads(client.get(f"/v1/jobs/{job['id']}/export?format=gemini").text.splitlines()[0])
    assert "contents" in gemini

    r = client.delete(f"/v1/jobs/{job['id']}/examples/{first['id']}")
    assert r.status_code == 204
    assert client.get(f"/v1/jobs/{job['id']}").json()["example_count"] == 3


def test_duplicates_are_rejected_and_counted(client):
    # 8 examples requested from 3 short paragraphs: the mock repeats questions, which must be dropped.
    ds_id = make_dataset(client)
    job_id = client.post(f"/v1/datasets/{ds_id}/jobs", json={"examples_per_chunk": 8}).json()["id"]
    job = wait_for(client, job_id)
    assert job["example_count"] == 6
    assert job["rejected"]["duplicate"] == 2


def test_max_examples_and_validation_split(client):
    docs = [
        {"text": f"Fact number {i} is that item {i} weighs {i * 3} kilograms. It ships in {i} days."}
        for i in range(30)
    ]
    ds_id = make_dataset(client, docs)
    job_id = client.post(
        f"/v1/datasets/{ds_id}/jobs",
        json={"examples_per_chunk": 2, "max_examples": 20, "validation_ratio": 0.3},
    ).json()["id"]
    job = wait_for(client, job_id)
    assert job["example_count"] == 20
    train = client.get(f"/v1/jobs/{job_id}/examples?split=train").json()["total"]
    val = client.get(f"/v1/jobs/{job_id}/examples?split=validation").json()["total"]
    assert train + val == 20 and val > 0


def test_classification_and_conversation_tasks(client):
    ds_id = make_dataset(client)
    r = client.post(f"/v1/datasets/{ds_id}/jobs", json={"task": "classification"})
    assert r.status_code == 422  # labels required

    job = wait_for(
        client,
        client.post(
            f"/v1/datasets/{ds_id}/jobs", json={"task": "classification", "labels": ["backup", "billing"]}
        ).json()["id"],
    )
    item = client.get(f"/v1/jobs/{job['id']}/examples").json()["items"][0]
    assert item["messages"][1]["content"] == "backup"

    job = wait_for(
        client, client.post(f"/v1/datasets/{ds_id}/jobs", json={"task": "conversation"}).json()["id"]
    )
    msgs = client.get(f"/v1/jobs/{job['id']}/examples").json()["items"][0]["messages"]
    assert len(msgs) == 4


def test_unconfigured_provider_returns_400(client):
    ds_id = make_dataset(client)
    r = client.post(f"/v1/datasets/{ds_id}/jobs", json={"provider": "anthropic"})
    assert r.status_code == 400 and "not configured" in r.json()["detail"]
    r = client.post(f"/v1/datasets/{ds_id}/jobs", json={"provider": "nope"})
    assert r.status_code == 400


def test_upload_formats(client):
    ds_id = client.post("/v1/datasets", json={"name": "uploads"}).json()["id"]
    jsonl = '{"text": "First record about refunds."}\n{"messages": [{"role": "user", "content": "Where is my order?"}]}\n'
    r = client.post(f"/v1/datasets/{ds_id}/documents/upload", files={"file": ("data.jsonl", jsonl)})
    assert r.json()["inserted"] == 2
    csv_data = "id,body\n1,Shipping takes three days.\n2,Returns are free.\n"
    r = client.post(
        f"/v1/datasets/{ds_id}/documents/upload",
        files={"file": ("faq.csv", csv_data)},
        data={"text_column": "body"},
    )
    assert r.json()["inserted"] == 2
    r = client.post(
        f"/v1/datasets/{ds_id}/documents/upload", files={"file": ("notes.md", "# Title\n\nBody text.")}
    )
    assert r.json()["document_count"] == 5
    r = client.post(f"/v1/datasets/{ds_id}/documents/upload", files={"file": ("x.pdf", "binary")})
    assert r.status_code == 422


def test_generate_one_shot(client):
    r = client.post(
        "/v1/generate",
        json={
            "documents": [{"text": ARTICLE}],
            "examples_per_chunk": 3,
            "format": "sharegpt",
            "redact_pii": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "mock" and len(body["examples"]) == 3
    assert body["examples"][0]["conversations"][0]["from"] == "human"


def test_api_key_is_enforced(settings):
    settings.access_key = "secret"
    with TestClient(create_app(settings, store=Store(":memory:"))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/datasets").status_code == 401
        assert c.get("/v1/datasets", headers={"authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/datasets", headers={"authorization": "Bearer secret"}).status_code == 200
        assert c.get("/v1/datasets", headers={"x-api-key": "secret"}).status_code == 200


def test_failed_chunks_can_be_retried(settings, monkeypatch):
    from llmvero.providers.mock import MockProvider

    original = MockProvider.complete
    state = {"fail": True}

    async def flaky(self, **kwargs):
        if state["fail"]:
            raise RuntimeError("provider down")
        return await original(self, **kwargs)

    monkeypatch.setattr(MockProvider, "complete", flaky)
    with TestClient(create_app(settings, store=Store(":memory:"))) as c:
        ds_id = make_dataset(c)
        job = wait_for(c, c.post(f"/v1/datasets/{ds_id}/jobs", json={}).json()["id"])
        assert job["status"] == "failed" and "provider down" in job["error"]

        state["fail"] = False
        c.post(f"/v1/jobs/{job['id']}/retry")
        job = wait_for(c, job["id"])
        assert job["status"] == "completed" and job["example_count"] > 0 and job["failed_chunks"] == 0


def test_jobs_resume_after_restart(settings):
    store = Store(settings.llmvero_data_dir / "resume.db")
    ds = store.create_dataset("resume")
    store.add_documents(ds["id"], [(ARTICLE, None, {})])
    job = store.create_job(ds["id"], "mock", "mock-1", {"task": "qa", "examples_per_chunk": 2})
    store.update_job(job["id"], status="running")  # as if the server died mid-job

    with TestClient(create_app(settings, store=store)) as c:
        done = wait_for(c, job["id"])
    assert done["status"] == "completed" and done["example_count"] == 2


def test_webhook_is_signed(settings):
    import hashlib
    import hmac

    import httpx

    received = {}

    def handler(request: httpx.Request):
        received["body"] = request.content
        received["sig"] = request.headers.get("x-llmvero-signature")
        return httpx.Response(200)

    settings.llmvero_webhook_secret = "whsec"
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(settings, store=Store(":memory:"), http_client=http)) as c:
        ds_id = make_dataset(c)
        job_id = c.post(
            f"/v1/datasets/{ds_id}/jobs", json={"webhook_url": "https://app.example/hook"}
        ).json()["id"]
        wait_for(c, job_id)
        deadline = time.time() + 5
        while "body" not in received and time.time() < deadline:
            time.sleep(0.05)

    payload = json.loads(received["body"])
    assert payload["event"] == "job.completed" and payload["job"]["id"] == job_id
    expected = hmac.new(b"whsec", received["body"], hashlib.sha256).hexdigest()
    assert received["sig"] == f"sha256={expected}"


def test_cancel_stops_job_and_sends_webhook(settings, monkeypatch):
    import asyncio

    import httpx

    from llmvero.providers.mock import MockProvider

    async def slow(self, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(MockProvider, "complete", slow)
    events = []
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: events.append(json.loads(r.content)["event"]) or httpx.Response(200)
        )
    )
    with TestClient(create_app(settings, store=Store(":memory:"), http_client=http)) as c:
        ds_id = make_dataset(c)
        job_id = c.post(
            f"/v1/datasets/{ds_id}/jobs", json={"webhook_url": "https://app.example/hook"}
        ).json()["id"]
        time.sleep(0.1)
        assert c.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        assert c.post(f"/v1/jobs/{job_id}/cancel").status_code == 409
    assert events == ["job.cancelled"]


def test_records_endpoint_takes_the_file_as_is(client):
    import pathlib

    raw = (pathlib.Path(__file__).parent.parent / "examples" / "login.json").read_bytes()
    r = client.post("/v1/training-data", content=raw)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["facts"]["overview"].startswith("# Login activity on Friday, 2026-11-06")
    assert "Peak hour: 09:00-09:59" in out["facts"]["overview"]
    assert len(out["examples"]) > 10 and not out["errors"]

    lines = (
        client.post("/v1/training-data?output=jsonl&format=sharegpt", content=raw).text.strip().splitlines()
    )
    assert len(lines) == len(out["examples"]) and "conversations" in json.loads(lines[0])


def test_records_endpoint_accepts_other_shapes(client):
    shapes = [
        '[{"userId": "ana", "createdAt": "2026-11-06T09:00:00Z"}, {"userId": "bo", "createdAt": 1793955600}]',
        '{"username": "ana", "timestamp": "2026-11-06 09:00"}\n{"username": "bo", "timestamp": "2026-11-06 10:00"}',
        "email;logged_in_at\nana@x.example;2026-11-06 09:00\nbo@x.example;2026-11-06 10:00\n",
        '{"data": [{"user": {"id": 5, "name": "Ana"}, "at": "2026-11-06 09:00:00"}]}',
    ]
    for body in shapes:
        r = client.post("/v1/training-data", content=body)
        assert r.status_code == 200, (body, r.text)
        assert "Peak hour" in r.json()["facts"]["overview"]
    csv_out = client.post("/v1/training-data", content=shapes[2]).json()
    assert csv_out["facts"]["overview"].startswith("# Login activity")  # guessed from logged_in_at


def test_records_endpoint_explains_what_it_cannot_read(client):
    r = client.post("/v1/training-data", content='[{"a": 1, "b": "x"}]')
    assert r.status_code == 422 and "Fields found: a, b" in r.json()["detail"]
    r = client.post("/v1/training-data?user_field=a&time_field=b", content='[{"a": 1, "b": "x"}]')
    assert r.status_code == 422 and "can't read" in r.json()["detail"]
    assert client.post("/v1/training-data", content="").status_code == 422


def test_generate_with_records_in_json_body(client):
    rows = [
        {"user_id": 1, "date_time": "2026-11-06 09:00:00"},
        {"user_id": 2, "date_time": "2026-11-06 09:30:00"},
    ]
    r = client.post("/v1/generate", json={"records": rows, "event": "login"})
    assert r.status_code == 200 and "Total logins: 2" in r.json()["facts"]["overview"]
    assert client.post("/v1/generate", json={}).status_code == 422


def _login_file():
    import pathlib

    return (pathlib.Path(__file__).parent.parent / "examples" / "login.json").read_bytes()


def test_tool_call_style_teaches_the_model_to_look_numbers_up(client):
    out = client.post("/v1/training-data?style=tool_calls", content=_login_file()).json()
    assert [t["function"]["name"] for t in out["tools"]] == [
        "get_activity_summary",
        "get_hourly_counts",
        "get_top_users",
        "get_daily_counts",
        "get_user_activity",
    ]
    by_tool = {}
    for ex in out["examples"]:
        user, call, tool, answer = ex["messages"]
        assert user["role"] == "user" and answer["role"] == "assistant" and answer["content"]
        fn = call["tool_calls"][0]["function"]
        assert tool["role"] == "tool" and tool["tool_call_id"] == call["tool_calls"][0]["id"]
        assert ex["tools"] == out["tools"]
        by_tool.setdefault(fn["name"], []).append((json.loads(fn["arguments"]), json.loads(tool["content"])))

    args, result = by_tool["get_activity_summary"][0]
    assert args == {"date_from": "2026-11-06", "date_to": "2026-11-06"}
    assert result["total"] == 145 and result["peak_hour"]["hour"] == "09:00-09:59"
    assert by_tool["get_top_users"][0][1]["top_users"][0] == {
        "rank": 1,
        "user_id": "7",
        "count": 14,
        "share_percent": 10,
    }
    for args, result in by_tool["get_user_activity"]:
        assert args["user_id"] == result["user_id"]  # the tool result is for the user the question names


def test_tool_call_style_in_sharegpt_and_unsupported_formats(client):
    out = client.post("/v1/training-data?style=tool_calls&format=sharegpt", content=_login_file()).json()
    turns = [c["from"] for c in out["examples"][0]["conversations"]]
    assert turns == ["human", "function_call", "observation", "gpt"]
    assert json.loads(out["examples"][0]["tools"])[0]["name"] == "get_activity_summary"

    r = client.post("/v1/training-data?style=tool_calls&format=alpaca", content=_login_file())
    assert r.status_code == 422 and "can't hold tool calls" in r.text
    r = client.post("/v1/generate", json={"documents": [{"text": "hello there"}], "style": "tool_calls"})
    assert r.status_code == 422


def test_eval_format_gives_question_answer_and_facts(client):
    lines = client.post("/v1/training-data?format=eval&output=jsonl", content=_login_file()).text.splitlines()
    cases = [json.loads(line) for line in lines]
    assert all({"question", "expected_answer", "facts"} <= case.keys() for case in cases)
    overview = next(c for c in cases if c["metadata"]["section"] == "overview")
    assert overview["facts"]["total"] == 145

    tool_cases = client.post("/v1/training-data?format=eval&style=tool_calls", content=_login_file()).json()
    assert tool_cases["examples"][0]["expected_tool_call"]["name"] == "get_activity_summary"

    text = client.post(
        "/v1/generate", json={"documents": [{"text": "Backups are kept for 30 days."}], "format": "eval"}
    ).json()
    assert text["examples"][0]["facts"] == "Backups are kept for 30 days."
