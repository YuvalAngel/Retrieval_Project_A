"""Preprocessing: Explicit Intro-Extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

CHUNK_SIZE = 180
STRIDE = 180

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record):
    page_id = int(record["page_id"])

    title = record.get("title", "").strip()
    content = record.get("content", "").strip()

    words = content.split()

    chunks = []

    for chunk_id, start in enumerate(
        range(0, len(words), STRIDE)
    ):
        chunk_words = words[start:start + CHUNK_SIZE]

        if len(chunk_words) < 40:
            continue

        text = (
            f"{title}\n\n"
            + " ".join(chunk_words)
        )

        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=chunk_id,
                text=text,
            )
        )

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
