"""The processing core: source text -> model calls -> checked, de-duplicated training examples."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections import Counter
from dataclasses import dataclass, field

import httpx

from ..config import Settings
from ..db import Store, now
from ..providers import Provider, build_provider
from ..schemas import GenerationConfig, JobCreate
from .chunking import chunk_text
from .parsing import ParseError, extract_items, to_messages
from .prompts import GENERATOR_SYSTEM, REVIEWER_SYSTEM, build_generation_prompt, build_review_prompt
from .quality import check_example, dedupe_key, redact_pii

log = logging.getLogger("llmvero")


@dataclass
class ChunkResult:
    examples: list[dict] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0


async def generate_for_chunk(provider: Provider, cfg: GenerationConfig, text: str) -> ChunkResult:
    """Run one chunk through the model and return the examples that pass every check."""
    result = ChunkResult()
    if cfg.redact_pii:
        text = redact_pii(text)

    prompt = build_generation_prompt(
        task=cfg.task,
        document=text,
        n=cfg.examples_per_chunk,
        turns=cfg.turns,
        labels=cfg.labels,
        language=cfg.language,
        instructions=cfg.instructions,
    )

    items: list[dict] | None = None
    last_error: Exception | None = None
    for _ in range(2):  # one extra try if the model returns broken JSON
        completion = await provider.complete(
            system=GENERATOR_SYSTEM,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        result.input_tokens += completion.input_tokens
        result.output_tokens += completion.output_tokens
        try:
            items = extract_items(completion.text)
            break
        except ParseError as exc:
            last_error = exc
    if items is None:
        raise ParseError(f"model did not return usable JSON: {last_error}")

    min_chars = 1 if cfg.task == "classification" else cfg.min_answer_chars
    candidates: list[list[dict]] = []
    seen: set[str] = set()
    for item in items:
        messages = to_messages(cfg.task, item, document=text, labels=cfg.labels)
        if messages is None:
            result.rejected["invalid"] += 1
            continue
        reason = check_example(messages, min_chars=min_chars, max_chars=cfg.max_example_chars)
        if reason:
            result.rejected[reason] += 1
            continue
        key = dedupe_key(messages)
        if key in seen:
            result.rejected["duplicate"] += 1
            continue
        seen.add(key)
        candidates.append(messages)

    scores: list[int | None] = [None] * len(candidates)
    if cfg.quality_check and candidates:
        scores = await _review(provider, cfg, text, candidates, result)

    for messages, score in zip(candidates, scores, strict=False):
        if score is not None and score < cfg.min_score:
            result.rejected["low_score"] += 1
            continue
        result.examples.append(
            {
                "system": cfg.system_prompt or None,
                "messages": messages,
                "score": score,
                "dedupe_hash": dedupe_key(messages),
            }
        )
    return result


async def _review(
    provider: Provider, cfg: GenerationConfig, text: str, candidates: list[list[dict]], result: ChunkResult
) -> list[int | None]:
    completion = await provider.complete(
        system=REVIEWER_SYSTEM,
        prompt=build_review_prompt(text, candidates),
        max_tokens=min(cfg.max_tokens, 4096),
        temperature=0,
    )
    result.input_tokens += completion.input_tokens
    result.output_tokens += completion.output_tokens
    scores: list[int | None] = [None] * len(candidates)
    try:
        for row in extract_items(completion.text, key="scores"):
            idx, score = row.get("index"), row.get("score")
            if isinstance(idx, int) and 0 <= idx < len(candidates) and isinstance(score, (int, float)):
                scores[idx] = int(score)
    except ParseError:
        log.warning("review reply was not valid JSON; keeping examples unscored")
    return scores


class JobRunner:
    """Runs jobs as background asyncio tasks. Progress lives in SQLite, so jobs resume after a restart."""

    def __init__(self, store: Store, settings: Settings, client: httpx.AsyncClient | None = None):
        self.store = store
        self.settings = settings
        self.client = client  # injected in tests
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False

    def start(self, job_id: str) -> None:
        if job_id in self._tasks and not self._tasks[job_id].done():
            return
        task = asyncio.create_task(self._run(job_id), name=f"llmvero-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    def resume_unfinished(self) -> None:
        for job_id in self.store.jobs_with_status("queued", "running"):
            log.info("resuming %s", job_id)
            self.start(job_id)

    async def cancel(self, job_id: str) -> None:
        self.store.update_job(job_id, status="cancelled", finished_at=now())
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        job = self.store.get_job(job_id)
        await self._notify(job_id, (job or {}).get("config", {}).get("webhook_url"))

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def wait(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job or job["status"] in ("completed", "cancelled"):
            return
        cfg = JobCreate(**job["config"])
        provider: Provider | None = None
        try:
            self.store.update_job(job_id, status="running", started_at=job["started_at"] or now())
            if not self.store.has_chunks(job_id):
                self._plan_chunks(job_id, job["dataset_id"], cfg)
            provider = build_provider(job["provider"], self.settings, model=job["model"], client=self.client)

            semaphore = asyncio.Semaphore(self.settings.llmvero_max_concurrency)
            pending = self.store.pending_chunks(job_id)

            async def work(chunk: dict) -> None:
                async with semaphore:
                    if cfg.max_examples and self.store.get_job(job_id)["example_count"] >= cfg.max_examples:
                        return
                    await self._process_chunk(job_id, chunk, provider, cfg)

            await asyncio.gather(*(work(c) for c in pending))
            self._finish(job_id)
        except asyncio.CancelledError:
            if not self._shutting_down:
                self.store.update_job(job_id, status="cancelled", finished_at=now())
            raise
        except Exception as exc:  # noqa: BLE001 - any failure ends the job with a readable error
            log.exception("job %s failed", job_id)
            self.store.update_job(job_id, status="failed", error=str(exc)[:1000], finished_at=now())
        finally:
            if provider:
                await provider.aclose()
        if not self._shutting_down:
            await self._notify(job_id, cfg.webhook_url)

    def _plan_chunks(self, job_id: str, dataset_id: str, cfg: GenerationConfig) -> None:
        chunks: list[tuple[str, str]] = []
        for doc in self.store.iter_documents(dataset_id):
            for piece in chunk_text(doc["text"], cfg.chunk_size, cfg.chunk_overlap):
                chunks.append((doc["id"], piece))
        if not chunks:
            raise ValueError("the dataset has no documents")
        self.store.add_chunks(job_id, chunks)

    async def _process_chunk(self, job_id: str, chunk: dict, provider: Provider, cfg: JobCreate) -> None:
        try:
            result = await generate_for_chunk(provider, cfg, chunk["text"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not stop the job
            log.warning("job %s chunk %s failed: %s", job_id, chunk["idx"], exc)
            self.store.fail_chunk(job_id, chunk["idx"], str(exc))
            return
        doc = self.store.get_document(chunk["document_id"]) or {}
        for ex in result.examples:
            ex["document_id"] = chunk["document_id"]
            ex["metadata"] = {
                "document_id": chunk["document_id"],
                "external_id": doc.get("external_id"),
                "chunk": chunk["idx"],
                "task": cfg.task,
                **(doc.get("metadata") or {}),
            }
        if cfg.max_examples:
            room = cfg.max_examples - self.store.get_job(job_id)["example_count"]
            result.examples = result.examples[: max(room, 0)]
        self.store.finish_chunk(
            job_id,
            chunk["idx"],
            examples=result.examples,
            rejected=dict(result.rejected),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            validation_ratio=cfg.validation_ratio,
        )

    def _finish(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job["status"] == "cancelled":
            return
        if job["done_chunks"] == 0 and job["failed_chunks"] > 0:
            status, error = "failed", job["error"] or "every chunk failed"
        else:
            status, error = "completed", None
            if job["failed_chunks"]:
                error = (
                    f"{job['failed_chunks']} chunk(s) failed; POST /v1/jobs/{job_id}/retry to run them again"
                )
        self.store.update_job(job_id, status=status, error=error, finished_at=now())

    async def _notify(self, job_id: str, url: str | None) -> None:
        if not url:
            return
        job = self.store.get_job(job_id)
        body = json.dumps({"event": f"job.{job['status']}", "job": job}, default=str).encode()
        headers = {"content-type": "application/json", "user-agent": "LLMvero-webhook"}
        if self.settings.llmvero_webhook_secret:
            digest = hmac.new(self.settings.llmvero_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
            headers["x-llmvero-signature"] = f"sha256={digest}"
        client = self.client or httpx.AsyncClient(timeout=15)
        try:
            for attempt in range(3):
                try:
                    resp = await client.post(url, content=body, headers=headers)
                    if resp.status_code < 500:
                        return
                except httpx.HTTPError as exc:
                    log.warning("webhook for %s failed: %s", job_id, exc)
                await asyncio.sleep(2**attempt)
        finally:
            if client is not self.client:
                await client.aclose()
