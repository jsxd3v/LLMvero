from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from ..auth import require_api_key
from ..config import Settings, get_settings
from ..core.chunking import chunk_text
from ..core.exporters import FORMAT_NOTES, convert
from ..core.pipeline import JobRunner, generate_for_chunk
from ..core.records import (
    DEFAULT_RECORD_INSTRUCTIONS,
    PROFILE_INSTRUCTIONS,
    Analysis,
    RecordError,
    analyze_records,
    is_single_user_question,
    load_rows,
    names_a_known_user,
    profile_users,
    tool_definitions,
    ungrounded_numbers,
)
from ..db import Store
from ..providers import PROVIDERS, build_provider, provider_settings
from ..schemas import (
    Dataset,
    DatasetCreate,
    DocumentIn,
    DocumentsAdd,
    DocumentsAdded,
    ExamplePage,
    ExportFormat,
    GenerateRequest,
    GenerateResponse,
    GenerationConfig,
    Job,
    JobCreate,
    ProviderInfo,
    Style,
)

log = logging.getLogger("llmvero")

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_GENERATE_CHARS = 60_000
MAX_RECORDS_BYTES = 100 * 1024 * 1024


def _event_word(sections: list[tuple[str, str]]) -> str:
    """Read the event name back from the overview title, e.g. '# Login activity on ...' -> 'login'."""
    title = sections[0][1].splitlines()[0].lstrip("# ")
    return title.split(" activity")[0].lower()


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_runner(request: Request) -> JobRunner:
    return request.app.state.runner


def _dataset_or_404(store: Store, ds_id: str) -> dict:
    ds = store.get_dataset(ds_id)
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return ds


def _job_or_404(store: Store, job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


def _resolve_provider(cfg: GenerationConfig, settings: Settings) -> tuple[str, str]:
    name = cfg.provider or settings.provider
    try:
        _, default_model, _, configured = provider_settings(name, settings)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not configured:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"provider '{name}' is not configured: set API_KEY in .env"
        )
    return name, cfg.model or default_model


# Info ------------------------------------------------------------------------


@router.get("/providers", response_model=list[ProviderInfo], tags=["info"])
def list_providers(settings: Settings = Depends(get_settings)):
    out = []
    for name in PROVIDERS:
        _, model, _, configured = provider_settings(name, settings)
        out.append(
            ProviderInfo(
                name=name,
                configured=configured,
                model=model,
                default=name == settings.provider,
            )
        )
    return out


@router.get("/formats", tags=["info"])
def list_formats() -> dict[str, str]:
    return FORMAT_NOTES


# Datasets --------------------------------------------------------------------


@router.post("/datasets", response_model=Dataset, status_code=201, tags=["datasets"])
def create_dataset(body: DatasetCreate, store: Store = Depends(get_store)):
    return store.create_dataset(body.name, body.description)


@router.get("/datasets", response_model=list[Dataset], tags=["datasets"])
def list_datasets(store: Store = Depends(get_store)):
    return store.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=Dataset, tags=["datasets"])
def get_dataset(dataset_id: str, store: Store = Depends(get_store)):
    return _dataset_or_404(store, dataset_id)


@router.delete("/datasets/{dataset_id}", status_code=204, tags=["datasets"])
async def delete_dataset(
    dataset_id: str, store: Store = Depends(get_store), runner: JobRunner = Depends(get_runner)
):
    _dataset_or_404(store, dataset_id)
    for job in store.list_jobs(dataset_id):
        if job["status"] in ("queued", "running"):
            await runner.cancel(job["id"])
    store.delete_dataset(dataset_id)


def _save_documents(store: Store, settings: Settings, ds_id: str, docs: list[DocumentIn]) -> DocumentsAdded:
    rows = []
    limit = settings.llmvero_max_document_chars
    for i, doc in enumerate(docs):
        text = doc.as_text()
        if len(text) > limit:
            raise HTTPException(
                413,
                f"document {i} has {len(text)} characters; the limit is {limit} (LLMVERO_MAX_DOCUMENT_CHARS)",
            )
        rows.append((text, doc.external_id, doc.metadata))
    inserted, skipped = store.add_documents(ds_id, rows)
    return DocumentsAdded(
        inserted=inserted,
        skipped_duplicates=skipped,
        document_count=store.get_dataset(ds_id)["document_count"],
    )


@router.post("/datasets/{dataset_id}/documents", response_model=DocumentsAdded, tags=["datasets"])
def add_documents(
    dataset_id: str,
    body: DocumentsAdd,
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    _dataset_or_404(store, dataset_id)
    return _save_documents(store, settings, dataset_id, body.documents)


@router.post("/datasets/{dataset_id}/documents/upload", response_model=DocumentsAdded, tags=["datasets"])
async def upload_documents(
    dataset_id: str,
    file: UploadFile = File(..., description=".txt, .md, .json, .jsonl or .csv"),
    text_column: str = Form("text", description="CSV column that holds the text."),
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    _dataset_or_404(store, dataset_id)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 50 MB")
    try:
        docs = parse_upload(file.filename or "", raw.decode("utf-8-sig"), text_column)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, f"could not read file: {exc}") from exc
    if not docs:
        raise HTTPException(422, "the file has no usable records")
    return _save_documents(store, settings, dataset_id, docs)


async def _read_body_capped(request: Request, limit: int, message: str) -> bytes:
    """Read the request body, stopping as soon as it passes `limit` bytes."""
    too_large = HTTPException(413, message)
    try:
        declared = int(request.headers.get("content-length", "0"))
    except ValueError:
        declared = 0
    if declared > limit:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_large
    return bytes(body)


def parse_upload(filename: str, content: str, text_column: str = "text") -> list[DocumentIn]:
    name = filename.lower()
    source = {"source_file": filename}

    def record(obj: Any, n: int) -> DocumentIn | None:
        if isinstance(obj, str):
            return DocumentIn(text=obj, metadata={**source, "row": n}) if obj.strip() else None
        if not isinstance(obj, dict):
            raise ValueError(f"record {n} is not an object or string")
        if not (obj.get("text") or obj.get("messages")):
            return None
        meta = {**source, "row": n, **(obj.get("metadata") or {})}
        return DocumentIn(
            text=obj.get("text"),
            messages=obj.get("messages"),
            external_id=obj.get("external_id"),
            metadata=meta,
        )

    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        items = [json.loads(line) for line in content.splitlines() if line.strip()]
        return [d for n, item in enumerate(items) if (d := record(item, n))]
    if name.endswith(".json"):
        data = json.loads(content)
        items = data if isinstance(data, list) else data.get("documents", [data])
        return [d for n, item in enumerate(items) if (d := record(item, n))]
    if name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content))
        if text_column not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV has no '{text_column}' column (found: {', '.join(reader.fieldnames or [])})"
            )
        docs = []
        for n, row in enumerate(reader):
            text = (row.pop(text_column) or "").strip()
            if text:
                docs.append(DocumentIn(text=text, metadata={**source, "row": n, **row}))
        return docs
    if name.endswith((".txt", ".md", ".markdown", ".rst", ".html", ".htm")) or "." not in name:
        return [DocumentIn(text=content, metadata=source)] if content.strip() else []
    raise ValueError("unsupported file type; use .txt, .md, .json, .jsonl or .csv")


# Jobs ------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/jobs", response_model=Job, status_code=202, tags=["jobs"])
async def create_job(
    dataset_id: str,
    body: JobCreate,
    store: Store = Depends(get_store),
    runner: JobRunner = Depends(get_runner),
    settings: Settings = Depends(get_settings),
):
    ds = _dataset_or_404(store, dataset_id)
    if ds["document_count"] == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "add documents to the dataset first")
    provider, model = _resolve_provider(body, settings)
    job = store.create_job(dataset_id, provider, model, body.model_dump(mode="json"))
    runner.start(job["id"])
    return job


@router.get("/datasets/{dataset_id}/jobs", response_model=list[Job], tags=["jobs"])
def list_dataset_jobs(dataset_id: str, store: Store = Depends(get_store)):
    _dataset_or_404(store, dataset_id)
    return store.list_jobs(dataset_id)


@router.get("/jobs", response_model=list[Job], tags=["jobs"])
def list_jobs(store: Store = Depends(get_store)):
    return store.list_jobs()


@router.get("/jobs/{job_id}", response_model=Job, tags=["jobs"])
def get_job(job_id: str, store: Store = Depends(get_store)):
    return _job_or_404(store, job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job, tags=["jobs"])
async def cancel_job(job_id: str, store: Store = Depends(get_store), runner: JobRunner = Depends(get_runner)):
    job = _job_or_404(store, job_id)
    if job["status"] not in ("queued", "running"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is already {job['status']}")
    await runner.cancel(job_id)
    return store.get_job(job_id)


@router.post("/jobs/{job_id}/retry", response_model=Job, tags=["jobs"])
async def retry_job(job_id: str, store: Store = Depends(get_store), runner: JobRunner = Depends(get_runner)):
    job = _job_or_404(store, job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(status.HTTP_409_CONFLICT, "job is still running")
    store.reset_failed_chunks(job_id)
    runner.start(job_id)
    return store.get_job(job_id)


@router.get("/jobs/{job_id}/examples", response_model=ExamplePage, tags=["examples"])
def list_examples(
    job_id: str,
    split: str | None = Query(None, pattern="^(train|validation)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: Store = Depends(get_store),
):
    _job_or_404(store, job_id)
    total, items = store.list_examples(job_id, split=split, limit=limit, offset=offset)
    return {"total": total, "items": items}


@router.delete("/jobs/{job_id}/examples/{example_id}", status_code=204, tags=["examples"])
def delete_example(job_id: str, example_id: str, store: Store = Depends(get_store)):
    if not store.delete_example(job_id, example_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "example not found")


@router.get("/jobs/{job_id}/export", tags=["examples"])
def export_job(
    job_id: str,
    format: ExportFormat = "openai",
    split: str | None = Query(None, pattern="^(train|validation)$"),
    store: Store = Depends(get_store),
):
    """Download the training data as JSONL, ready to upload to a fine-tuning service."""
    _job_or_404(store, job_id)

    def lines():
        for ex in store.iter_examples(job_id, split=split):
            yield json.dumps(convert(format, ex), ensure_ascii=False) + "\n"

    filename = f"{job_id}-{format}{'-' + split if split else ''}.jsonl"
    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"content-disposition": f'attachment; filename="{filename}"'},
    )


# One-shot ----------------------------------------------------------------------


@router.post("/training-data", response_model=GenerateResponse, tags=["generate"])
async def create_training_data(
    request: Request,
    event: str | None = Query(None, description="What one row is, e.g. login. Guessed when empty."),
    user_field: str | None = Query(None, description="Field holding the user. Detected when empty."),
    time_field: str | None = Query(None, description="Field holding the date/time. Detected when empty."),
    examples_per_chunk: int = Query(5, ge=1, le=50),
    system_prompt: str | None = Query(None, description="Defaults to SYSTEM_PROMPT in .env."),
    provider: str | None = Query(None),
    model: str | None = Query(None),
    quality_check: bool = Query(False),
    style: Style = Query(
        "qa",
        description="tool_calls: the assistant looks numbers up with a function call instead of stating them.",
    ),
    format: ExportFormat = Query("openai"),
    output: str = Query(
        "json", pattern="^(json|jsonl)$", description="jsonl = only the training data lines."
    ),
    settings: Settings = Depends(get_settings),
):
    """Send your data file as-is (JSON, JSON Lines or CSV), get training data back.

    Example: `curl --data-binary @login.json localhost:8000/v1/training-data`
    """
    raw = await _read_body_capped(request, MAX_RECORDS_BYTES, "the file is larger than 100 MB")
    try:
        rows, hint = load_rows(raw)
        body = GenerateRequest(
            records=rows,
            event=event or hint,
            user_field=user_field,
            time_field=time_field,
            examples_per_chunk=examples_per_chunk,
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            quality_check=quality_check,
            style=style,
            format=format,
        )
    except (RecordError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    result = await generate(body, request, settings)
    if output == "jsonl":
        lines = "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in result.examples)
        return Response(
            lines,
            media_type="application/x-ndjson",
            headers={"content-disposition": 'attachment; filename="training-result.jsonl"'},
        )
    return result


@router.post("/generate", response_model=GenerateResponse, tags=["generate"])
async def generate(body: GenerateRequest, request: Request, settings: Settings = Depends(get_settings)):
    """One call: send documents or raw records (e.g. a day of logins), get training examples back."""
    facts: dict[str, str] = {}
    analysis: Analysis | None = None
    if not body.system_prompt and settings.system_prompt:
        body = body.model_copy(update={"system_prompt": settings.system_prompt})
    if body.records:
        try:
            analysis = analyze_records(
                body.records, user_field=body.user_field, time_field=body.time_field, event=body.event
            )
        except RecordError as exc:
            raise HTTPException(422, str(exc)) from exc
        sections = analysis.sections
        facts = dict(sections)
        body = body.model_copy(
            update={
                "documents": [
                    DocumentIn(text=text, external_id=name, metadata={"section": name})
                    for name, text in sections
                ],
                "instructions": body.instructions
                or DEFAULT_RECORD_INSTRUCTIONS.format(event=_event_word(sections)),
            }
        )
    texts = [d.as_text() for d in body.documents]
    if sum(len(t) for t in texts) > MAX_GENERATE_CHARS:
        raise HTTPException(
            413,
            f"/v1/generate accepts up to {MAX_GENERATE_CHARS} characters; create a dataset and a job instead",
        )
    name, model = _resolve_provider(body, settings)
    provider = build_provider(
        name, settings, model=model, client=getattr(request.app.state, "http_client", None)
    )

    rejected: Counter = Counter()
    seen: set[str] = set()
    examples: list[dict] = []
    errors: list[str] = []
    tokens_in = tokens_out = 0
    pieces = [
        (doc, chunk)
        for doc, text in zip(body.documents, texts, strict=True)
        for chunk in chunk_text(text, body.chunk_size, body.chunk_overlap)
    ]
    semaphore = asyncio.Semaphore(settings.llmvero_max_concurrency)
    done = 0

    def config_for(doc: DocumentIn) -> GenerateRequest:
        """User-habit sheets only describe some users: one question per user, about that user."""
        users = (
            profile_users(doc.text or "")
            if doc.metadata.get("section", "").startswith("user_profiles")
            else []
        )
        if not users:
            return body
        title = (doc.text or "").splitlines()[0]
        period = title.split(" habits ", 1)[-1] if " habits " in title else "period"
        event = title.removeprefix("# User ").split(" habits")[0]
        return body.model_copy(
            update={
                "examples_per_chunk": len(users),
                "instructions": PROFILE_INSTRUCTIONS.format(period=period, event=event),
            }
        )

    async def run(doc: DocumentIn, chunk: str):
        nonlocal done
        async with semaphore:
            try:
                return await generate_for_chunk(provider, config_for(doc), chunk)
            except Exception as exc:  # noqa: BLE001 - report and continue with the other chunks
                return exc
            finally:
                done += 1
                log.info("generate: %d/%d model calls done", done, len(pieces))

    log.info("generate: %d model calls to %s/%s", len(pieces), name, model)
    try:
        # Calls run in parallel (LLMVERO_MAX_CONCURRENCY at a time); results are merged in order.
        results = await asyncio.gather(*(run(doc, chunk) for doc, chunk in pieces))
        for (doc, chunk), result in zip(pieces, results, strict=True):
            if isinstance(result, Exception):
                errors.append(str(result)[:500])
                continue
            if users := (
                profile_users(chunk) if doc.metadata.get("section", "").startswith("user_profiles") else []
            ):
                kept = [
                    ex
                    for ex in result.examples
                    if is_single_user_question(ex["messages"][0]["content"], users)
                ]
                if len(kept) < len(result.examples):
                    result.rejected["not_about_one_user"] += len(result.examples) - len(kept)
                result.examples = kept
            elif analysis is not None:
                # One user's activity is looked up with get_user_activity, never through the summary sheets.
                kept = [
                    ex
                    for ex in result.examples
                    if not names_a_known_user(ex["messages"][0]["content"], analysis.users)
                ]
                if len(kept) < len(result.examples):
                    result.rejected["single_user_on_summary_sheet"] += len(result.examples) - len(kept)
                result.examples = kept
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            rejected.update(result.rejected)
            for ex in result.examples:
                if ex["dedupe_hash"] in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(ex["dedupe_hash"])
                ex["metadata"] = {"external_id": doc.external_id, "task": body.task, **doc.metadata}
                if analysis is None:
                    ex["context"] = chunk
                else:
                    call = analysis.tool_call_for(
                        doc.metadata.get("section", ""), ex["messages"][0]["content"]
                    )
                    if call is None:
                        rejected["no_matching_tool"] += 1
                        continue
                    ex["context"] = call[2]
                    if ungrounded_numbers(ex["messages"][-1]["content"], call[2]):
                        rejected["number_not_in_facts"] += 1  # e.g. the model added numbers up itself
                        continue
                    if body.style == "tool_calls":
                        ex["messages"] = with_tool_call(ex["messages"], *call, ex["dedupe_hash"])
                        ex["tools"] = tool_definitions(analysis.event)
                examples.append(convert(body.format, ex))
    finally:
        await provider.aclose()

    if errors and not examples:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, {"message": "generation failed", "errors": errors})
    return GenerateResponse(
        provider=name,
        model=model,
        examples=examples,
        rejected=dict(rejected),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        errors=errors,
        facts=facts,
        tools=tool_definitions(analysis.event) if analysis and body.style == "tool_calls" else None,
    )


def with_tool_call(messages: list[dict], name: str, arguments: dict, result: dict, key: str) -> list[dict]:
    """[user, assistant] -> [user, assistant calls the tool, tool result, assistant answer]."""
    call_id = f"call_{key[:24]}"
    question, answer = messages[0], messages[-1]
    return [
        question,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)},
        answer,
    ]
