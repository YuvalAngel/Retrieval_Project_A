"""Page-level BM25 with a compact numpy postings artifact.

The synthetic corpus reuses template sentences across entity clusters; the rare
content words of a query (e.g. "thermal imaging pipelines", "patent pool") are
highly discriminative, which makes lexical scoring a strong complement to dense
passage retrieval. Postings are stored as flat numpy arrays (CSR-style) instead
of JSON: ~10x smaller on disk and loads in seconds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils import ARTIFACTS_DIR

BM25_POSTINGS_NAME = "bm25_postings.npz"
BM25_VOCAB_NAME = "bm25_vocab.json"
K1 = 2.0
B = 1.0  # full length normalization: long real-Wikipedia distractor pages
         # otherwise outscore the short synthetic template pages

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    """a an and are as at be been by for from had has have he her his i if in is it
    its of on or s t that the their them they this to was were which who will with""".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def build_bm25(records: List[Dict[str, Any]], artifacts_dir: Path | None = None) -> None:
    """Build the page-level inverted index over full text (title + content)."""
    texts = [f"{r.get('title', '')} {r.get('content', '')}" for r in records]
    _build_postings(texts, artifacts_dir, BM25_POSTINGS_NAME, BM25_VOCAB_NAME)


def _build_postings(
    texts: List[str],
    artifacts_dir: Path | None,
    postings_name: str,
    vocab_name: str,
) -> None:
    out_dir = artifacts_dir or ARTIFACTS_DIR
    n_docs = len(texts)
    doc_len = np.zeros(n_docs, dtype=np.int32)
    postings: Dict[str, Dict[int, int]] = {}
    for di, text in enumerate(texts):
        tokens = tokenize(text)
        doc_len[di] = len(tokens)
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            postings.setdefault(t, {})[di] = c

    # drop hapax terms (df == 1): they never help template queries and halve the vocab
    terms = sorted(t for t, p in postings.items() if len(p) > 1)
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    idf = np.zeros(len(terms), dtype=np.float32)
    docs_flat: List[np.ndarray] = []
    tfs_flat: List[np.ndarray] = []
    for ti, t in enumerate(terms):
        p = postings[t]
        df = len(p)
        idf[ti] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        d = np.fromiter(p.keys(), dtype=np.int32, count=df)
        c = np.fromiter(p.values(), dtype=np.int64, count=df)
        docs_flat.append(d)
        tfs_flat.append(np.minimum(c, 65535).astype(np.uint16))
        offsets[ti + 1] = offsets[ti] + df

    np.savez_compressed(
        out_dir / postings_name,
        offsets=offsets,
        doc_idx=np.concatenate(docs_flat) if docs_flat else np.zeros(0, np.int32),
        tf=np.concatenate(tfs_flat) if tfs_flat else np.zeros(0, np.uint16),
        idf=idf,
        doc_len=doc_len,
    )
    (out_dir / vocab_name).write_text(
        json.dumps({t: i for i, t in enumerate(terms)}), encoding="utf-8"
    )


def load_bm25(artifacts_dir: Path | None = None) -> Dict[str, Any]:
    root = artifacts_dir or ARTIFACTS_DIR
    data = np.load(root / BM25_POSTINGS_NAME)
    vocab = json.loads((root / BM25_VOCAB_NAME).read_text(encoding="utf-8"))
    doc_len = data["doc_len"].astype(np.float32)
    avgdl = float(doc_len.mean()) if doc_len.size else 1.0
    return {
        "vocab": vocab,
        "offsets": data["offsets"],
        "doc_idx": data["doc_idx"],
        "tf": data["tf"].astype(np.float32),
        "idf": data["idf"],
        "len_norm": K1 * (1.0 - B + B * doc_len / avgdl),  # precomputed per doc
        "n_docs": doc_len.shape[0],
    }


def score_bm25_query(query: str, bm: Dict[str, Any]) -> np.ndarray:
    """Return a dense (n_docs,) BM25 score vector for one query."""
    scores = np.zeros(bm["n_docs"], dtype=np.float32)
    vocab, offsets = bm["vocab"], bm["offsets"]
    for term in set(tokenize(query)):
        ti = vocab.get(term)
        if ti is None:
            continue
        lo, hi = offsets[ti], offsets[ti + 1]
        docs = bm["doc_idx"][lo:hi]
        tf = bm["tf"][lo:hi]
        scores[docs] += bm["idf"][ti] * tf * (K1 + 1.0) / (tf + bm["len_norm"][docs])
    return scores
