import pytest

from llmvero.core.chunking import chunk_text
from llmvero.core.exporters import convert
from llmvero.core.parsing import ParseError, extract_items, extract_json, to_messages
from llmvero.core.quality import check_example, dedupe_key, redact_pii, split_for


def test_chunk_short_text_is_one_chunk():
    assert chunk_text("  hello world  ") == ["hello world"]
    assert chunk_text("   ") == []


def test_chunk_respects_size_and_keeps_all_text():
    paras = [f"Paragraph {i} " + "word " * 80 for i in range(20)]
    chunks = chunk_text("\n\n".join(paras), size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i} " in c for c in chunks)


def test_chunk_splits_giant_paragraph():
    chunks = chunk_text("x" * 2500, size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} Hope that helps.') == {"a": [1, 2]}
    assert extract_items('[{"question": "q"}]') == [{"question": "q"}]
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_to_messages_per_task():
    assert to_messages("qa", {"question": "Q", "answer": "A"}, document="d") == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    assert to_messages("qa", {"question": "Q"}, document="d") is None
    inst = to_messages("instruction", {"instruction": "Do", "input": "this", "output": "Done"}, document="d")
    assert inst[0]["content"] == "Do\n\nthis"
    cls = to_messages("classification", {"label": "BILLING"}, document="doc", labels=["billing", "tech"])
    assert cls == [{"role": "user", "content": "doc"}, {"role": "assistant", "content": "billing"}]
    assert to_messages("classification", {"label": "other"}, document="d", labels=["billing"]) is None
    conv = to_messages(
        "conversation",
        {"messages": [{"role": "human", "content": "hi"}, {"role": "gpt", "content": "hello"}]},
        document="d",
    )
    assert [m["role"] for m in conv] == ["user", "assistant"]


def test_quality_checks():
    ok = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a good answer"}]
    assert check_example(ok, min_chars=2, max_chars=1000) is None
    assert check_example(ok[:1], min_chars=2, max_chars=1000) == "invalid_structure"
    assert check_example([ok[0], ok[0], ok[1]], min_chars=2, max_chars=1000) == "invalid_structure"
    assert check_example(ok, min_chars=50, max_chars=1000) == "too_short"
    assert check_example(ok, min_chars=2, max_chars=5) == "too_long"
    refusal = [ok[0], {"role": "assistant", "content": "As an AI language model, I cannot help."}]
    assert check_example(refusal, min_chars=2, max_chars=1000) == "refusal"


def test_dedupe_ignores_case_and_punctuation():
    a = [{"role": "user", "content": "What is X?"}, {"role": "assistant", "content": "1"}]
    b = [{"role": "user", "content": "what   is x"}, {"role": "assistant", "content": "2"}]
    assert dedupe_key(a) == dedupe_key(b)


def test_split_is_deterministic():
    key = dedupe_key([{"role": "user", "content": "x"}])
    assert split_for(key, 0) == "train"
    assert split_for(key, 0.2) == split_for(key, 0.2)
    keys = [dedupe_key([{"role": "user", "content": str(i)}]) for i in range(1000)]
    held_out = sum(split_for(k, 0.2) == "validation" for k in keys)
    assert 120 < held_out < 280


def test_redact_pii():
    text = "Mail jane.doe@example.com or call +1 415-555-0132. Card 4111 1111 1111 1111, server 10.0.0.12."
    out = redact_pii(text)
    for secret in ("jane.doe@example.com", "415-555-0132", "4111 1111", "10.0.0.12"):
        assert secret not in out
    assert "[EMAIL]" in out and "[PHONE]" in out and "[CARD]" in out and "[IP]" in out
    assert redact_pii("Released on 2026-11-08 with 30 days retention") == (
        "Released on 2026-11-08 with 30 days retention"
    )


EX = {
    "system": "You are Acme's assistant.",
    "messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ],
    "metadata": {},
}


def test_exporters():
    assert convert("openai", EX)["messages"][0] == {"role": "system", "content": "You are Acme's assistant."}
    anthropic = convert("anthropic", EX)
    assert anthropic["system"] == EX["system"] and anthropic["messages"][0]["role"] == "user"
    gemini = convert("gemini", EX)
    assert gemini["contents"][1]["role"] == "model"
    assert gemini["systemInstruction"]["parts"][0]["text"] == EX["system"]
    sharegpt = convert("sharegpt", EX)["conversations"]
    assert [c["from"] for c in sharegpt] == ["system", "human", "gpt", "human", "gpt"]
    alpaca = convert("alpaca", EX)
    assert alpaca["instruction"] == "q2" and alpaca["output"] == "a2" and alpaca["history"] == [["q1", "a1"]]
