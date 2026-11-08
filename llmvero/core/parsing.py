"""Read model output: pull the JSON out and turn each item into a chat transcript."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ParseError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and text around it."""
    text = text.strip()
    if not text:
        raise ParseError("empty reply")
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ParseError("reply is not valid JSON")


def extract_items(text: str, key: str = "examples") -> list[dict]:
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get(key, data.get("data", [data]))
    if not isinstance(data, list):
        raise ParseError(f"expected a list under '{key}'")
    return [item for item in data if isinstance(item, dict)]


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def to_messages(
    task: str,
    item: dict,
    *,
    document: str,
    labels: list[str] | None = None,
) -> list[dict] | None:
    """Convert one generated item to [{role, content}, ...]. Returns None if it is malformed."""
    if task == "qa":
        q, a = _s(item.get("question")), _s(item.get("answer"))
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}] if q and a else None

    if task == "instruction":
        instruction, extra, output = (
            _s(item.get("instruction")),
            _s(item.get("input")),
            _s(item.get("output")),
        )
        if not instruction or not output:
            return None
        user = f"{instruction}\n\n{extra}" if extra else instruction
        return [{"role": "user", "content": user}, {"role": "assistant", "content": output}]

    if task == "summarization":
        summary = _s(item.get("summary"))
        if not summary:
            return None
        return [
            {"role": "user", "content": f"Summarize the following text.\n\n{document}"},
            {"role": "assistant", "content": summary},
        ]

    if task == "classification":
        label = _s(item.get("label"))
        if labels:
            match = next((lb for lb in labels if lb.lower() == label.lower()), None)
            if match is None:
                return None
            label = match
        if not label:
            return None
        return [{"role": "user", "content": document}, {"role": "assistant", "content": label}]

    # conversation / custom
    raw = item.get("messages") or item.get("conversation") or item.get("turns")
    if not isinstance(raw, list):
        return None
    messages = []
    for msg in raw:
        if not isinstance(msg, dict):
            return None
        role = _s(msg.get("role")).lower()
        role = {"human": "user", "gpt": "assistant", "model": "assistant", "bot": "assistant"}.get(role, role)
        content = _s(msg.get("content") or msg.get("text") or msg.get("value"))
        if role not in ("user", "assistant") or not content:
            return None
        messages.append({"role": role, "content": content})
    return messages or None
