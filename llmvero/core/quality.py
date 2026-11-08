"""Quality gates applied to every generated example before it is stored."""

from __future__ import annotations

import hashlib
import re

_REFUSALS = re.compile(
    r"\b(as an ai( language model)?|i (cannot|can't|am unable to) (help|assist|answer|provide)|"
    r"the (document|text|passage) does not (say|mention|contain|provide))\b",
    re.IGNORECASE,
)

_PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"(?<!\w)\+?\d{1,3}?[ .-]?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]?\d{3,4}(?!\w)"), "[PHONE]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
]


def redact_pii(text: str) -> str:
    """Mask emails, card numbers, SSNs, phone numbers and IPs before text leaves the server."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def check_example(messages: list[dict], *, min_chars: int, max_chars: int) -> str | None:
    """Return a rejection reason, or None if the example passes."""
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return "invalid_structure"
    for prev, cur in zip(messages, messages[1:], strict=False):
        if prev["role"] == cur["role"]:
            return "invalid_structure"
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    if any(len(r) < min_chars for r in replies):
        return "too_short"
    if sum(len(m["content"]) for m in messages) > max_chars:
        return "too_long"
    if any(_REFUSALS.search(r) for r in replies):
        return "refusal"
    return None


def dedupe_key(messages: list[dict]) -> str:
    """Hash of the normalized user turns, so the same question phrased with different spacing is a duplicate."""
    users = " | ".join(m["content"] for m in messages if m["role"] == "user")
    normalized = re.sub(r"[^\w]+", " ", users.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_for(key: str, validation_ratio: float) -> str:
    """Deterministic train/validation assignment from the dedupe hash."""
    if validation_ratio <= 0:
        return "train"
    return "validation" if int(key[:8], 16) / 0xFFFFFFFF < validation_ratio else "train"
