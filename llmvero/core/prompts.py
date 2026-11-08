"""Prompts sent to the user's model. Each task asks for a JSON object with an `examples` list."""

from __future__ import annotations

import json

GENERATOR_SYSTEM = (
    "You create high-quality training data for fine-tuning language models. "
    "Every example must be grounded in the source document: never invent facts that are not in it. "
    "Write natural, varied user messages and complete, accurate, well-written assistant replies. "
    "Do not mention 'the document', 'the text' or 'the passage' in the examples unless the task asks for it. "
    "Reply with a single JSON object and nothing else."
)

REVIEWER_SYSTEM = (
    "You review training examples for a fine-tuning dataset. Score each example from 1 to 5: "
    "5 = accurate, grounded in the source, helpful and well written; "
    "3 = usable but weak; 1 = wrong, ungrounded, unhelpful or malformed. "
    "Reply with a single JSON object and nothing else."
)

TASKS: dict[str, dict[str, str]] = {
    "qa": {
        "goal": "Write {n} question and answer pairs. Each question is something a real user might ask, "
        "and the answer is fully supported by the document. Cover different facts; avoid near-duplicates.",
        "schema": '{"examples": [{"question": "...", "answer": "..."}]}',
    },
    "instruction": {
        "goal": "Write {n} diverse instruction-following examples based on the document (explain, compare, "
        "list, rewrite, extract, reason step by step...). `input` holds extra context for the instruction "
        "and may be an empty string.",
        "schema": '{"examples": [{"instruction": "...", "input": "...", "output": "..."}]}',
    },
    "conversation": {
        "goal": "Write {n} realistic multi-turn conversations between a user and an assistant about the "
        "document, each with about {turns} user turns. Turns alternate user/assistant, start with the user "
        "and end with the assistant.",
        "schema": '{"examples": [{"messages": [{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "..."}]}]}',
    },
    "summarization": {
        "goal": "Write {n} summary of the document. Keep every key fact, drop filler, and use plain language.",
        "schema": '{"examples": [{"summary": "..."}]}',
    },
    "classification": {
        "goal": "Classify the document into exactly one of these labels: {labels}. "
        "Return {n} example with the chosen label.",
        "schema": '{"examples": [{"label": "..."}]}',
    },
    "custom": {
        "goal": "Write {n} training examples following the instructions below.",
        "schema": '{"examples": [{"messages": [{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "..."}]}]}',
    },
}


def build_generation_prompt(
    *,
    task: str,
    document: str,
    n: int,
    turns: int = 3,
    labels: list[str] | None = None,
    language: str | None = None,
    instructions: str | None = None,
) -> str:
    spec = TASKS[task]
    if task in ("summarization", "classification"):
        n = 1
    goal = spec["goal"].format(n=n, turns=turns, labels=", ".join(labels or []))
    lines = [
        f"Task: {task}",
        f"Number of examples: {n}",
        f"Language: {language or 'same as the document'}",
        "",
        goal,
    ]
    if instructions:
        lines += ["", "Additional instructions from the dataset owner:", instructions.strip()]
    lines += [
        "",
        "Output format (JSON):",
        spec["schema"],
        "",
        "<document>",
        document,
        "</document>",
    ]
    return "\n".join(lines)


def build_review_prompt(document: str, examples: list[list[dict]]) -> str:
    rendered = json.dumps(
        [{"index": i, "messages": msgs} for i, msgs in enumerate(examples)], ensure_ascii=False, indent=1
    )
    return "\n".join(
        [
            "Task: review",
            f"Number of examples: {len(examples)}",
            "",
            "Score every example below against the source document.",
            "",
            "Output format (JSON):",
            '{"scores": [{"index": 0, "score": 5, "reason": "..."}]}',
            "",
            "<document>",
            document,
            "</document>",
            "",
            "<examples>",
            rendered,
            "</examples>",
        ]
    )
