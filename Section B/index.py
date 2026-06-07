"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries
from bm25 import build_bm25
from structure import build_structure

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    
    # 1. Build Dense Index (using your Single-Chunk / Intro-Extractor)
    chunks = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    # 2. Build Sparse BM25 Index
    build_bm25(records, out_dir)

    # 3. Build structural metadata for generated entity families
    build_structure(records, out_dir)
    
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    
    # Extract only page_ids. chunk_ids has been completely removed!
    page_ids = [int(x) for x in meta["page_ids"]]
    
    return vectors, page_ids
