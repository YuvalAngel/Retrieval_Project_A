"""Offline index build and load (not timed at grading).

Two dense vector sets are built at different passage granularities (100 and 150
words) and ensembled at query time — the ensemble beats either granularity alone
on the public queries. Vector files are sharded into <95MB parts so the repo
needs no Git LFS and a fresh clone runs out of the box.

Artifacts written to artifacts/:
- vectors_w{G}_f16.partNN.npy  float16 L2-normalized passage embeddings for
                               granularity G, rows grouped by page in corpus
                               order, concatenated across parts
- page_offsets_w{G}.npy        int64 (n_pages + 1,) row ranges per page
- page_ids.npy                 int64 (n_pages,) page_id per page position
- bm25_postings.npz + bm25_vocab.json  page-level BM25 (see bm25.py)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bm25 import build_bm25
from chunk import chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

GRANULARITIES = (100, 150)
PAGE_IDS_NAME = "page_ids.npy"
PAGE_CLUSTERS_NAME = "page_clusters.npy"
SHARD_BYTES = 95 * 1024 * 1024
CLUSTER_TAU = 0.75


def _save_sharded(arr: np.ndarray, out_dir: Path, stem: str) -> None:
    rows_per = max(1, SHARD_BYTES // (arr.shape[1] * arr.itemsize))
    for k, start in enumerate(range(0, arr.shape[0], rows_per)):
        np.save(out_dir / f"{stem}.part{k:02d}.npy", arr[start:start + rows_per])


def _load_sharded(root: Path, stem: str) -> np.ndarray:
    parts = sorted(root.glob(f"{stem}.part*.npy"))
    if not parts:
        raise FileNotFoundError(f"no shards for {stem} under {root}")
    return np.concatenate([np.load(p) for p in parts], axis=0)


def _cluster_pages(leads: np.ndarray, tau: float = CLUSTER_TAU) -> np.ndarray:
    """Entity clusters via union-find over lead-passage cosine > tau.

    Cluster siblings (player page / season page / arena page of one entity)
    share lead sentences, so within-cluster lead similarity is 0.73-0.97 while
    template-mates of *different* entities stay below ~0.55 — tau separates
    them cleanly. Returns an int64 cluster label per page.
    """
    n = leads.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s0 in range(0, n, 2000):
        sims = leads[s0:s0 + 2000] @ leads.T
        ii, jj = np.nonzero(sims > tau)
        for i, j in zip(ii, jj):
            gi = s0 + int(i)
            if gi < j:
                ri, rj = find(gi), find(int(j))
                if ri != rj:
                    parent[ri] = rj
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Chunk, embed (both granularities) and persist the corpus (run once, offline)."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    page_ids = [int(r["page_id"]) for r in records]
    np.save(out_dir / PAGE_IDS_NAME, np.asarray(page_ids, dtype=np.int64))

    for g in GRANULARITIES:
        chunks = chunk_corpus(records, target_words=g)
        print(f"granularity {g}: pages={len(records)} chunks={len(chunks)}")
        vectors = embed_texts(
            [c.text for c in chunks], batch_size=128, show_progress=True
        )
        counts: Dict[int, int] = {}
        for c in chunks:
            counts[c.page_id] = counts.get(c.page_id, 0) + 1
        offsets = np.zeros(len(page_ids) + 1, dtype=np.int64)
        for i, pid in enumerate(page_ids):
            offsets[i + 1] = offsets[i] + counts[pid]
        assert offsets[-1] == len(chunks)
        _save_sharded(vectors.astype(np.float16), out_dir, f"vectors_w{g}_f16")
        np.save(out_dir / f"page_offsets_w{g}.npy", offsets)
        if g == GRANULARITIES[0]:
            print("clustering pages by lead-passage similarity...")
            leads = vectors[offsets[:-1]].astype(np.float32)
            np.save(out_dir / PAGE_CLUSTERS_NAME, _cluster_pages(leads))

    print("building bm25...")
    build_bm25(records, out_dir)
    print("done.")


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Return ([(vectors f32, page_offsets) per granularity], page_ids, clusters)."""
    root = artifacts_dir or ARTIFACTS_DIR
    sets = []
    for g in GRANULARITIES:
        vectors = _load_sharded(root, f"vectors_w{g}_f16").astype(np.float32)
        offsets = np.load(root / f"page_offsets_w{g}.npy")
        sets.append((vectors, offsets))
    page_ids = np.load(root / PAGE_IDS_NAME)
    clusters = np.load(root / PAGE_CLUSTERS_NAME)
    return sets, page_ids, clusters
