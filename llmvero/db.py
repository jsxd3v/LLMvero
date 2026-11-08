"""SQLite storage. One file under LLMVERO_DATA_DIR; WAL mode so reads don't block the job runner."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    external_id TEXT,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, content_hash)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT NOT NULL,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    done_chunks INTEGER NOT NULL DEFAULT 0,
    failed_chunks INTEGER NOT NULL DEFAULT 0,
    example_count INTEGER NOT NULL DEFAULT 0,
    rejected TEXT NOT NULL DEFAULT '{}',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS examples (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    document_id TEXT,
    system TEXT,
    messages TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    score INTEGER,
    dedupe_hash TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'train',
    created_at TEXT NOT NULL,
    UNIQUE (job_id, dedupe_hash)
);
CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset_id);
CREATE INDEX IF NOT EXISTS idx_examples_job ON examples(job_id, split);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _exec(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    # Datasets ---------------------------------------------------------------

    def create_dataset(self, name: str, description: str = "") -> dict:
        ds_id = new_id("ds")
        self._exec(
            "INSERT INTO datasets (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (ds_id, name, description, now()),
        )
        return self.get_dataset(ds_id)  # type: ignore[return-value]

    def get_dataset(self, ds_id: str) -> dict | None:
        row = self._one(
            "SELECT d.*, (SELECT COUNT(*) FROM documents WHERE dataset_id = d.id) AS document_count "
            "FROM datasets d WHERE d.id = ?",
            (ds_id,),
        )
        return dict(row) if row else None

    def list_datasets(self) -> list[dict]:
        rows = self._all(
            "SELECT d.*, (SELECT COUNT(*) FROM documents WHERE dataset_id = d.id) AS document_count "
            "FROM datasets d ORDER BY d.created_at DESC"
        )
        return [dict(r) for r in rows]

    def delete_dataset(self, ds_id: str) -> bool:
        return self._exec("DELETE FROM datasets WHERE id = ?", (ds_id,)) > 0

    # Documents --------------------------------------------------------------

    def add_documents(self, ds_id: str, docs: list[tuple[str, str | None, dict]]) -> tuple[int, int]:
        """Insert (text, external_id, metadata) rows. Identical text in the same dataset is skipped."""
        inserted = 0
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                for text, external_id, metadata in docs:
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO documents "
                        "(id, dataset_id, external_id, text, metadata, content_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (new_id("doc"), ds_id, external_id, text, json.dumps(metadata), content_hash, now()),
                    )
                    inserted += cur.rowcount
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return inserted, len(docs) - inserted

    def iter_documents(self, ds_id: str) -> Iterator[dict]:
        for row in self._all(
            "SELECT id, external_id, text, metadata FROM documents WHERE dataset_id = ? ORDER BY created_at, id",
            (ds_id,),
        ):
            yield {**dict(row), "metadata": json.loads(row["metadata"])}

    def get_document(self, doc_id: str) -> dict | None:
        row = self._one("SELECT id, external_id, metadata FROM documents WHERE id = ?", (doc_id,))
        return {**dict(row), "metadata": json.loads(row["metadata"])} if row else None

    # Jobs -------------------------------------------------------------------

    def create_job(self, ds_id: str, provider: str, model: str, config: dict) -> dict:
        job_id = new_id("job")
        self._exec(
            "INSERT INTO jobs (id, dataset_id, status, provider, model, config, created_at) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (job_id, ds_id, provider, model, json.dumps(config), now()),
        )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict | None:
        row = self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not row:
            return None
        job = dict(row)
        job["config"] = json.loads(job["config"])
        job["rejected"] = json.loads(job["rejected"])
        return job

    def list_jobs(self, ds_id: str | None = None) -> list[dict]:
        if ds_id:
            rows = self._all("SELECT id FROM jobs WHERE dataset_id = ? ORDER BY created_at DESC", (ds_id,))
        else:
            rows = self._all("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 200")
        return [j for r in rows if (j := self.get_job(r["id"]))]

    def jobs_with_status(self, *statuses: str) -> list[str]:
        marks = ",".join("?" * len(statuses))
        return [r["id"] for r in self._all(f"SELECT id FROM jobs WHERE status IN ({marks})", statuses)]

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def add_chunks(self, job_id: str, chunks: list[tuple[str, str]]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.executemany(
                "INSERT INTO chunks (job_id, idx, document_id, text) VALUES (?, ?, ?, ?)",
                [(job_id, i, doc_id, text) for i, (doc_id, text) in enumerate(chunks)],
            )
            self._conn.execute("UPDATE jobs SET total_chunks = ? WHERE id = ?", (len(chunks), job_id))
            self._conn.execute("COMMIT")

    def pending_chunks(self, job_id: str) -> list[dict]:
        rows = self._all(
            "SELECT idx, document_id, text FROM chunks WHERE job_id = ? AND status IN ('pending', 'running') "
            "ORDER BY idx",
            (job_id,),
        )
        return [dict(r) for r in rows]

    def has_chunks(self, job_id: str) -> bool:
        return self._one("SELECT 1 FROM chunks WHERE job_id = ? LIMIT 1", (job_id,)) is not None

    def reset_failed_chunks(self, job_id: str) -> int:
        with self._lock:
            n = self._conn.execute(
                "UPDATE chunks SET status = 'pending', error = NULL WHERE job_id = ? AND status = 'failed'",
                (job_id,),
            ).rowcount
            self._conn.execute(
                "UPDATE jobs SET failed_chunks = 0, status = 'queued', error = NULL, finished_at = NULL "
                "WHERE id = ?",
                (job_id,),
            )
        return n

    def finish_chunk(
        self,
        job_id: str,
        idx: int,
        *,
        examples: list[dict],
        rejected: dict[str, int],
        input_tokens: int,
        output_tokens: int,
        validation_ratio: float,
    ) -> int:
        """Store a chunk's examples and update the job counters in one transaction. Returns kept count."""
        from .core.quality import split_for

        with self._lock:
            self._conn.execute("BEGIN")
            try:
                kept = 0
                for ex in examples:
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO examples (id, job_id, document_id, system, messages, metadata, "
                        "score, dedupe_hash, split, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            new_id("ex"),
                            job_id,
                            ex.get("document_id"),
                            ex.get("system"),
                            json.dumps(ex["messages"], ensure_ascii=False),
                            json.dumps(ex.get("metadata") or {}, ensure_ascii=False),
                            ex.get("score"),
                            ex["dedupe_hash"],
                            split_for(ex["dedupe_hash"], validation_ratio),
                            now(),
                        ),
                    )
                    kept += cur.rowcount
                dupes = len(examples) - kept
                row = self._conn.execute("SELECT rejected FROM jobs WHERE id = ?", (job_id,)).fetchone()
                totals = json.loads(row["rejected"])
                for reason, n in {**rejected, "duplicate": rejected.get("duplicate", 0) + dupes}.items():
                    if n:
                        totals[reason] = totals.get(reason, 0) + n
                self._conn.execute(
                    "UPDATE chunks SET status = 'done', error = NULL WHERE job_id = ? AND idx = ?",
                    (job_id, idx),
                )
                self._conn.execute(
                    "UPDATE jobs SET done_chunks = done_chunks + 1, example_count = example_count + ?, "
                    "rejected = ?, input_tokens = input_tokens + ?, output_tokens = output_tokens + ? "
                    "WHERE id = ?",
                    (kept, json.dumps(totals), input_tokens, output_tokens, job_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return kept

    def fail_chunk(
        self, job_id: str, idx: int, error: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET status = 'failed', error = ? WHERE job_id = ? AND idx = ?",
                (error[:1000], job_id, idx),
            )
            self._conn.execute(
                "UPDATE jobs SET failed_chunks = failed_chunks + 1, error = ?, "
                "input_tokens = input_tokens + ?, output_tokens = output_tokens + ? WHERE id = ?",
                (error[:1000], input_tokens, output_tokens, job_id),
            )

    # Examples ---------------------------------------------------------------

    def _example(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "system": row["system"],
            "messages": json.loads(row["messages"]),
            "metadata": json.loads(row["metadata"]),
            "score": row["score"],
            "split": row["split"],
        }

    def list_examples(
        self, job_id: str, *, split: str | None, limit: int, offset: int
    ) -> tuple[int, list[dict]]:
        where, params = "job_id = ?", [job_id]
        if split:
            where += " AND split = ?"
            params.append(split)
        total = self._one(f"SELECT COUNT(*) AS n FROM examples WHERE {where}", tuple(params))["n"]
        rows = self._all(
            f"SELECT * FROM examples WHERE {where} ORDER BY created_at, id LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        return total, [self._example(r) for r in rows]

    def iter_examples(self, job_id: str, split: str | None = None, batch: int = 500) -> Iterator[dict]:
        offset = 0
        while True:
            _, items = self.list_examples(job_id, split=split, limit=batch, offset=offset)
            yield from items
            if len(items) < batch:
                return
            offset += batch

    def delete_example(self, job_id: str, example_id: str) -> bool:
        with self._lock:
            n = self._conn.execute(
                "DELETE FROM examples WHERE id = ? AND job_id = ?", (example_id, job_id)
            ).rowcount
            if n:
                self._conn.execute(
                    "UPDATE jobs SET example_count = example_count - 1 WHERE id = ?", (job_id,)
                )
        return n > 0
