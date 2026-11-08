# Advanced

The [README](../README.md) covers the main flow: send event rows (who + when) to `POST /v1/training-data`, get
training data back. This page covers everything else.

- [Text mode](#text-mode): articles, docs, chat logs
- [Datasets and background jobs](#datasets-and-background-jobs): large inputs, progress, resume
- [Task types](#task-types)
- [Quality checks](#quality-checks)
- [Export formats](#export-formats)
- [Webhooks](#webhooks)
- [All settings](#all-settings)
- [API reference](#api-reference)

## Text mode

For data that is text rather than event rows (help articles, product pages, documentation, support chats),
send documents instead of records. The model writes examples straight from the text, so there are no computed
facts here.

```bash
curl -X POST http://localhost:8000/v1/generate -H "content-type: application/json" -d '{
  "task": "qa",
  "examples_per_chunk": 5,
  "documents": [
    {"text": "To reset your password, open Settings > Security...", "external_id": "kb-12"},
    {"messages": [{"role": "user", "content": "I was charged twice"}, {"role": "assistant", "content": "..."}]}
  ]
}'
```

A document has `text` or `messages` (an existing conversation), plus optional `external_id` and `metadata`, which
are kept on every example made from it. Long text is split into overlapping chunks (`chunk_size`,
`chunk_overlap`). `/v1/generate` takes up to 60,000 characters; for more, use a dataset and a job.

`/v1/generate` also accepts event rows inside a JSON body (`{"records": [...]}`), the same as `/v1/training-data`.

## Datasets and background jobs

For large text inputs, store documents in a dataset and process them in a background job. Progress is saved
per chunk in SQLite (`LLMVERO_DATA_DIR`), so a job continues after a restart without paying again for finished
chunks.

```bash
# 1. Create a dataset
curl -X POST localhost:8000/v1/datasets -H 'content-type: application/json' -d '{"name": "Help center"}'
# → {"id": "ds_..."}

# 2. Add documents (JSON, up to 5,000 per call) or upload a file (.txt, .md, .json, .jsonl, .csv)
curl -X POST localhost:8000/v1/datasets/ds_.../documents -H 'content-type: application/json' \
  -d '{"documents": [{"text": "..."}]}'
curl -X POST localhost:8000/v1/datasets/ds_.../documents/upload -F file=@articles.jsonl

# 3. Start a job (returns 202 at once)
curl -X POST localhost:8000/v1/datasets/ds_.../jobs -H 'content-type: application/json' \
  -d '{"task": "qa", "examples_per_chunk": 5, "validation_ratio": 0.1, "quality_check": true}'
# → {"id": "job_...", "status": "queued"}

# 4. Check progress: status, chunks done, examples kept, rejections, tokens
curl localhost:8000/v1/jobs/job_...

# 5. Download the result, split into train and validation
curl "localhost:8000/v1/jobs/job_.../export?format=openai&split=train" -o train.jsonl
curl "localhost:8000/v1/jobs/job_.../export?format=openai&split=validation" -o validation.jsonl
```

Identical documents in a dataset are stored once. A failed chunk (bad key, model down) doesn't stop the job;
`POST /v1/jobs/{id}/retry` re-runs only the failed ones. You can page through examples with
`GET /v1/jobs/{id}/examples` and delete bad ones before exporting.

Job statuses: `queued` → `running` → `completed`, `failed` (every chunk failed) or `cancelled`.

## Task types

Text mode and jobs take a `task`:

| `task` | Each example looks like | Extra options |
|---|---|---|
| `qa` | user question → grounded answer | |
| `instruction` | instruction (+ input) → output | |
| `conversation` | multi-turn user/assistant chat | `turns` |
| `summarization` | "Summarize: <chunk>" → summary | |
| `classification` | chunk → one of your labels | `labels` (required) |
| `custom` | whatever your instructions describe | `instructions` (required) |

All tasks also take `instructions` (tone, focus, what to avoid), `system_prompt`, `language`,
`examples_per_chunk`, `chunk_size`, `chunk_overlap`, `temperature`, `max_tokens`, `redact_pii`, `quality_check`,
`min_score`, `validation_ratio`. Jobs also take `max_examples` and `webhook_url`.

Records mode always uses `qa`, in one of two styles: `qa` (the answer states the numbers) or `tool_calls` (the
assistant calls a function, then answers from its result). See [tool-calling.md](tool-calling.md).

## Quality checks

Every generated example goes through these steps:

| Step | What it does | Option |
|---|---|---|
| PII masking | Emails, phone numbers, card numbers, SSNs and IPs become `[EMAIL]`, `[PHONE]`... before text is sent to the model (text mode) | `redact_pii` |
| JSON parsing | Reads the reply even with code fences or text around it; one retry if it isn't JSON | |
| Structure | Starts with the user, alternates roles, ends with the assistant, no empty turns | |
| Length | Drops replies that are too short and examples that are too long | `min_answer_chars`, `max_example_chars` |
| Refusals | Drops "As an AI language model..." and "the text does not mention..." replies | |
| Numbers from the facts | Records mode: every number and time in an answer must appear in the facts it came from; otherwise it's dropped as `number_not_in_facts` | |
| One user per question | Records mode: questions from a user-habits sheet must name exactly one user from that sheet | |
| De-duplication | The same question (ignoring case, spacing and punctuation) is kept once | |
| Model review | A second call scores each example 1–5 against the source; scores below `min_score` are dropped | `quality_check`, `min_score` |
| Train/validation split | Deterministic, so exporting again gives the same split | `validation_ratio` |

Every response or job reports what was dropped and why, e.g. `{"duplicate": 4, "too_short": 1}`.

PII masking is pattern-based: treat it as a safety net, not a guarantee. Records mode doesn't need it, because
raw rows are never sent to the model.

## Export formats

Choose with `?format=` on `/v1/training-data` and job exports, or `"format"` in a `/v1/generate` body.

| `format` | Shape | Use it with |
|---|---|---|
| `openai` | `{"messages": [{role, content}...]}` | OpenAI and Azure OpenAI fine-tuning, Together, Fireworks, most open-source trainers |
| `anthropic` | `{"system", "messages"}` | Claude fine-tuning on Amazon Bedrock |
| `gemini` | `{"systemInstruction", "contents": [{role, parts}]}` | Vertex AI Gemini tuning |
| `sharegpt` | `{"conversations": [{from, value}]}` | Axolotl, LLaMA-Factory, Unsloth |
| `alpaca` | `{"instruction", "input", "output", "history"}` | Alpaca-style instruction tuning |
| `raw` | everything stored, including metadata and scores | Your own pipeline |
| `eval` | `{"question", "expected_answer", "expected_tool_call", "facts"}` | Testing a chatbot or agent ([details](tool-calling.md#testing-it)) |

With `style=tool_calls` (records only), use `openai`, `sharegpt`, `raw` or `eval`; the other formats can't hold
function calls. See [tool-calling.md](tool-calling.md).

## Webhooks

Give a job a `webhook_url` and LLMvero POSTs to it when the job ends:

```json
{"event": "job.completed", "job": {"id": "job_...", "status": "completed", "example_count": 120, ...}}
```

Events: `job.completed`, `job.failed`, `job.cancelled`. With `LLMVERO_WEBHOOK_SECRET` set, the body is signed with
HMAC-SHA256 in the header `X-LLMvero-Signature: sha256=<hex>`. Check it against the raw body before trusting the
call.

## All settings

The README lists the common ones. The full list:

| Variable | Default | |
|---|---|---|
| `PROVIDER` | `anthropic` | `anthropic`, `openai`, `gemini`, `openai_compatible` or `mock` |
| `MODEL` | provider default | |
| `API_KEY` | empty | Key for `PROVIDER` |
| `BASE_URL` | provider default | For `openai_compatible` or a custom endpoint |
| `ACCESS_KEY` | empty | Callers must send `Authorization: Bearer <key>` (or `X-API-Key`). `/health` stays open |
| `SYSTEM_PROMPT` | empty | System message stored in every example |
| `LLMVERO_DATA_DIR` | `./data` | SQLite file for datasets and jobs |
| `LLMVERO_MAX_CONCURRENCY` | `4` | Model calls running at the same time |
| `LLMVERO_WEBHOOK_SECRET` | empty | Signs webhook calls |
| `LLMVERO_CORS_ORIGINS` | `http://localhost:3000` | Browser origins allowed to call the API |
| `LLMVERO_MAX_DOCUMENT_CHARS` | `200000` | Largest single document in text mode |
| `LLMVERO_REQUEST_TIMEOUT` | `120` | Seconds per model call |
| `LLMVERO_MAX_RETRIES` | `3` | Retries on rate limits, overloads and network errors |

**Several providers at once.** `API_KEY` and `MODEL` belong to `PROVIDER`. To let single calls use another
provider (`?provider=gemini`), also set that provider's own key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` (and optionally `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`), or
`OPENAI_COMPATIBLE_BASE_URL` + `OPENAI_COMPATIBLE_MODEL` (+ `OPENAI_COMPATIBLE_API_KEY`).

## API reference

Interactive docs: `http://localhost:8000/docs`.

| Method | Path | |
|---|---|---|
| `POST` | `/v1/training-data` | Event rows as a file body → training data |
| `POST` | `/v1/generate` | Documents or records in a JSON body → training data |
| `POST` | `/v1/datasets` | Create a dataset |
| `GET` | `/v1/datasets` · `/v1/datasets/{id}` | List / get |
| `DELETE` | `/v1/datasets/{id}` | Delete it with its documents, jobs and examples |
| `POST` | `/v1/datasets/{id}/documents` | Add documents |
| `POST` | `/v1/datasets/{id}/documents/upload` | Upload a file (`text_column` for CSV) |
| `POST` | `/v1/datasets/{id}/jobs` | Start a job |
| `GET` | `/v1/datasets/{id}/jobs` · `/v1/jobs` · `/v1/jobs/{id}` | List / get jobs |
| `POST` | `/v1/jobs/{id}/cancel` · `/v1/jobs/{id}/retry` | Cancel, or re-run failed chunks |
| `GET` | `/v1/jobs/{id}/examples` | Page through examples (`split`, `limit`, `offset`) |
| `DELETE` | `/v1/jobs/{id}/examples/{example_id}` | Remove one example |
| `GET` | `/v1/jobs/{id}/export` | Download JSONL (`format`, `split`) |
| `GET` | `/v1/providers` · `/v1/formats` · `/health` | Info |
