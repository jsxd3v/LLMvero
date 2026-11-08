"""Convert stored examples into the JSONL formats used by fine-tuning services and tools."""

from __future__ import annotations

import json
from collections.abc import Callable

# {"messages": [...], "system": str | None, "metadata": {...}, "tools": [...] | None, "context": ... | None}
Example = dict

# Formats that can hold tool calls (style=tool_calls).
TOOL_FORMATS = ("openai", "sharegpt", "raw", "eval")


def _openai(ex: Example) -> dict:
    msgs = ([{"role": "system", "content": ex["system"]}] if ex.get("system") else []) + ex["messages"]
    out: dict = {"messages": msgs}
    if ex.get("tools"):
        out["tools"] = ex["tools"]
    return out


def _anthropic(ex: Example) -> dict:
    out: dict = {}
    if ex.get("system"):
        out["system"] = ex["system"]
    out["messages"] = ex["messages"]
    return out


def _gemini(ex: Example) -> dict:
    out: dict = {}
    if ex.get("system"):
        out["systemInstruction"] = {"role": "system", "parts": [{"text": ex["system"]}]}
    out["contents"] = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in ex["messages"]
    ]
    return out


def _sharegpt(ex: Example) -> dict:
    """ShareGPT as read by LLaMA-Factory; tool calls use its function_call / observation turns."""
    convs = [{"from": "system", "value": ex["system"]}] if ex.get("system") else []
    for m in ex["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]["function"]
            value = json.dumps({"name": call["name"], "arguments": json.loads(call["arguments"])})
            convs.append({"from": "function_call", "value": value})
        elif m["role"] == "tool":
            convs.append({"from": "observation", "value": m["content"]})
        else:
            convs.append({"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]})
    out: dict = {"conversations": convs}
    if ex.get("tools"):
        out["tools"] = json.dumps([t["function"] for t in ex["tools"]])
    return out


def _alpaca(ex: Example) -> dict:
    msgs = ex["messages"]
    pairs = [(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs) - 1, 2)]
    *history, (instruction, output) = pairs
    out: dict = {"instruction": instruction, "input": "", "output": output}
    if ex.get("system"):
        out["system"] = ex["system"]
    if history:
        out["history"] = [list(p) for p in history]
    return out


def _raw(ex: Example) -> dict:
    out = {k: ex.get(k) for k in ("id", "system", "messages", "metadata", "score", "split")}
    if ex.get("tools"):
        out["tools"] = ex["tools"]
    return out


def _eval(ex: Example) -> dict:
    """A test case: the question, the exact expected answer, and the facts it comes from."""
    msgs = ex["messages"]
    out: dict = {
        "question": next(m["content"] for m in msgs if m["role"] == "user"),
        "expected_answer": next(m["content"] for m in reversed(msgs) if m["role"] == "assistant"),
    }
    call = next((m["tool_calls"][0]["function"] for m in msgs if m.get("tool_calls")), None)
    if call:
        out["expected_tool_call"] = {"name": call["name"], "arguments": json.loads(call["arguments"])}
    if ex.get("context") is not None:
        out["facts"] = ex["context"]
    if ex.get("metadata"):
        out["metadata"] = ex["metadata"]
    return out


FORMATS: dict[str, Callable[[Example], dict]] = {
    "openai": _openai,
    "anthropic": _anthropic,
    "gemini": _gemini,
    "sharegpt": _sharegpt,
    "alpaca": _alpaca,
    "raw": _raw,
    "eval": _eval,
}

FORMAT_NOTES = {
    "openai": "OpenAI fine-tuning, Azure OpenAI, Together, Fireworks, most open-source trainers",
    "anthropic": "Claude fine-tuning on Amazon Bedrock (system + messages)",
    "gemini": "Vertex AI Gemini supervised tuning",
    "sharegpt": "Axolotl, LLaMA-Factory, Unsloth (conversations / from / value)",
    "alpaca": "Alpaca-style instruction tuning (instruction / input / output / history)",
    "raw": "Everything LLMvero stores, including metadata and review scores",
    "eval": "Test cases (question, expected answer, facts) for checking a chatbot or agent",
}


def convert(fmt: str, example: Example) -> dict:
    return FORMATS[fmt](example)
