"""Passage chunking: sentence-packed windows sized for MiniLM's 256-token context.

Pages in this corpus are long (median ~1.2k words) while all-MiniLM-L6-v2 reads only
~190 words, so single-chunk embedding discards most of the text. Each page is split
into ~TARGET_WORDS passages on sentence boundaries with a one-sentence overlap, and
the title is prepended to every passage so each retrieval unit stays self-identifying.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

TARGET_WORDS = 100  # default; index.py builds one vector set per granularity
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _sentences(content: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(content) if s.strip()]


def chunk_entry(record: Dict[str, Any], target_words: int = TARGET_WORDS) -> List[Chunk]:
    """Split one corpus entry into overlapping sentence-packed passages."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    prefix = f"{title}. " if title else ""

    sents = _sentences(content)
    if not sents:
        sents = [title or str(page_id)]

    texts: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for s in sents:
        cur.append(s)
        cur_words += len(s.split())
        if cur_words >= target_words:
            texts.append(prefix + " ".join(cur))
            # one-sentence overlap, unless that sentence alone is near-target size
            last = cur[-1]
            if len(last.split()) <= target_words // 2:
                cur = [last]
                cur_words = len(last.split())
            else:
                cur = []
                cur_words = 0
    if cur_words > 0:
        texts.append(prefix + " ".join(cur))
    if not texts:
        texts = [prefix.strip() or str(page_id)]
    return [Chunk(page_id=page_id, chunk_id=j, text=t) for j, t in enumerate(texts)]


def chunk_corpus(
    records: List[Dict[str, Any]], target_words: int = TARGET_WORDS
) -> List[Chunk]:
    """Chunk all records; chunks of one page stay contiguous (index relies on it)."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record, target_words))
    return chunks
