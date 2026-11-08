from __future__ import annotations

import json
import re

from .base import Completion, Provider


class MockProvider(Provider):
    """Offline provider that builds examples from the document's own sentences.

    It needs no API key, costs nothing and is deterministic, which makes it useful for
    tests, CI and wiring up your app before you plug in a real model.
    """

    name = "mock"

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> Completion:
        task = _field(prompt, "Task") or "qa"
        n = int(_field(prompt, "Number of examples") or 1)
        doc_match = re.search(r"<document>\n(.*?)\n</document>", prompt, re.DOTALL)
        document = doc_match.group(1) if doc_match else ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document) if len(s.strip()) > 3] or [
            document
        ]

        if task == "review":
            out: dict = {"scores": [{"index": i, "score": 5, "reason": "mock"} for i in range(n)]}
        else:
            out = {"examples": [_item(task, sentences, i, prompt) for i in range(n)]}
        text = json.dumps(out)
        return Completion(text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


def _field(prompt: str, name: str) -> str | None:
    m = re.search(rf"^{name}: (.+)$", prompt, re.MULTILINE)
    return m.group(1).strip() if m else None


def _topic(sentence: str) -> str:
    return " ".join(sentence.rstrip(".!?").split()[:6])


def _item(task: str, sentences: list[str], i: int, prompt: str) -> dict:
    s = sentences[i % len(sentences)]
    if task == "instruction":
        return {"instruction": f"Explain this point: {_topic(s)}", "input": "", "output": s}
    if task == "summarization":
        return {"summary": " ".join(sentences[:2])}
    if task == "classification":
        m = re.search(r"these labels: (.+?)\. Return", prompt)
        return {"label": m.group(1).split(", ")[0] if m else "other"}
    if task == "conversation":
        messages = []
        for j in range(2):
            t = sentences[(i + j) % len(sentences)]
            messages += [
                {"role": "user", "content": f"Can you tell me about {_topic(t)}?"},
                {"role": "assistant", "content": t},
            ]
        return {"messages": messages}
    if task == "custom":
        return {
            "messages": [{"role": "user", "content": f"{_topic(s)}?"}, {"role": "assistant", "content": s}]
        }
    return {"question": f"What should I know about {_topic(s)}?", "answer": s}
