"""Preprocessing: Explicit Intro-Extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Extract only the title and the lead paragraph (first 150 words).
    This maximizes signal-to-noise and perfectly fits the token limit.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    content = record.get("content", "").strip()

    # Grab the first 150 words of the content
    words = content.split()
    lead_paragraph = " ".join(words[:150])

    # Prepend the title for maximum semantic context
    chunk_text = f"{title} : {lead_paragraph}" if title else lead_paragraph

    return [Chunk(page_id=page_id, chunk_id=0, text=chunk_text)]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
