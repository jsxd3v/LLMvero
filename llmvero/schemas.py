from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Task = Literal["qa", "instruction", "conversation", "summarization", "classification", "custom"]
ExportFormat = Literal["openai", "anthropic", "gemini", "sharegpt", "alpaca", "raw", "eval"]
Style = Literal["qa", "tool_calls"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class ChatMessage(BaseModel):
    role: str
    content: str


class DocumentIn(BaseModel):
    """One record from your app: an article, a support ticket, a product page, a chat log..."""

    text: str | None = None
    messages: list[ChatMessage] | None = Field(
        None, description="An existing conversation. It is turned into a transcript and used as the source."
    )
    external_id: str | None = Field(None, max_length=200, description="Your own id for this record.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_content(self) -> DocumentIn:
        if not (self.text and self.text.strip()) and not self.messages:
            raise ValueError("a document needs `text` or `messages`")
        return self

    def as_text(self) -> str:
        if self.text and self.text.strip():
            return self.text.strip()
        return "\n".join(f"{m.role.capitalize()}: {m.content.strip()}" for m in self.messages or [])


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""


class Dataset(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    created_at: str


class DocumentsAdd(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=5000)


class DocumentsAdded(BaseModel):
    inserted: int
    skipped_duplicates: int
    document_count: int


class GenerationConfig(BaseModel):
    """How to turn source documents into training examples."""

    task: Task = "qa"
    provider: str | None = Field(None, description="anthropic, openai, gemini, openai_compatible or mock.")
    model: str | None = Field(None, description="Overrides the provider's model from .env.")
    examples_per_chunk: int = Field(5, ge=1, le=50)
    turns: int = Field(3, ge=2, le=10, description="User turns per conversation (task=conversation).")
    labels: list[str] | None = Field(None, description="Allowed labels (task=classification).")
    instructions: str | None = Field(None, max_length=5000, description="Extra guidance for the generator.")
    system_prompt: str | None = Field(
        None, max_length=5000, description="System message placed in every training example."
    )
    language: str | None = None
    chunk_size: int = Field(4000, ge=500, le=50_000)
    chunk_overlap: int = Field(200, ge=0, le=5000)
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int = Field(4096, ge=256, le=64_000)
    redact_pii: bool = Field(False, description="Mask emails, phones, cards and IPs before sending text out.")
    quality_check: bool = Field(False, description="Ask the model to score each example and drop weak ones.")
    min_score: int = Field(4, ge=1, le=5)
    min_answer_chars: int = Field(2, ge=1)
    max_example_chars: int = Field(20_000, ge=100)
    validation_ratio: float = Field(
        0.0, ge=0, le=0.5, description="Share of examples held out for validation."
    )

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        if self.task == "classification" and not self.labels:
            raise ValueError("task=classification needs `labels`")
        if self.task == "custom" and not self.instructions:
            raise ValueError("task=custom needs `instructions`")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class JobCreate(GenerationConfig):
    max_examples: int | None = Field(None, ge=1, description="Stop once this many examples are kept.")
    webhook_url: str | None = Field(None, description="Called with the job when it finishes.")


class Job(BaseModel):
    id: str
    dataset_id: str
    status: JobStatus
    provider: str
    model: str
    config: dict[str, Any]
    total_chunks: int
    done_chunks: int
    failed_chunks: int
    example_count: int
    rejected: dict[str, int]
    input_tokens: int
    output_tokens: int
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class Example(BaseModel):
    id: str
    system: str | None
    messages: list[ChatMessage]
    metadata: dict[str, Any]
    score: int | None
    split: str


class ExamplePage(BaseModel):
    total: int
    items: list[Example]


class GenerateRequest(GenerationConfig):
    """One-shot generation for small inputs: no dataset, no job, the examples come back in the response."""

    documents: list[DocumentIn] | None = Field(None, min_length=1, max_length=50)
    records: list[dict[str, Any]] | None = Field(
        None,
        min_length=1,
        max_length=1_000_000,
        description="Raw event rows such as logins: [{user_id, date_time}, ...]. Stats are computed exactly, "
        "then the model writes training data from them.",
    )
    user_field: str | None = Field(None, description="Field holding the user. Detected when empty.")
    time_field: str | None = Field(None, description="Field holding the date/time. Detected when empty.")
    event: str | None = Field(None, max_length=50, description="What one row is (e.g. login), for wording.")
    style: Style = Field(
        "qa",
        description="qa: answers state the numbers. tool_calls: the assistant calls a function to get them "
        "(records only), so a fine-tuned model looks numbers up instead of memorizing them.",
    )
    format: ExportFormat = "openai"

    @model_validator(mode="after")
    def _one_source(self) -> GenerateRequest:
        if bool(self.documents) == bool(self.records):
            raise ValueError("send either `documents` or `records`")
        if self.style == "tool_calls":
            if not self.records:
                raise ValueError("style=tool_calls needs `records` (event rows)")
            if self.format not in ("openai", "sharegpt", "raw", "eval"):
                raise ValueError(
                    f"format={self.format} can't hold tool calls; use openai, sharegpt, raw or eval"
                )
        return self


class GenerateResponse(BaseModel):
    provider: str
    model: str
    examples: list[dict[str, Any]]
    rejected: dict[str, int]
    input_tokens: int
    output_tokens: int
    errors: list[str]
    facts: dict[str, str] = Field(default_factory=dict, description="Fact sheets computed from `records`.")
    tools: list[dict[str, Any]] | None = Field(
        None,
        description="With style=tool_calls: the functions the trained model calls. Implement them in your app.",
    )


class ProviderInfo(BaseModel):
    name: str
    configured: bool
    model: str
    default: bool
