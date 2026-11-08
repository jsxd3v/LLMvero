"""Split documents into overlapping chunks that fit comfortably in one generation call."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, size: int = 4000, overlap: int = 200) -> list[str]:
    """Pack paragraphs into chunks of at most `size` characters.

    Paragraphs longer than `size` are split on sentences, then hard-split as a last resort.
    Each chunk after the first starts with the last `overlap` characters of the previous one,
    so facts that straddle a boundary are not lost.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            while len(sentence) > size:
                pieces.append(sentence[:size])
                sentence = sentence[size:]
            if sentence:
                pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= size else piece
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    # Start the overlap at a word boundary.
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
