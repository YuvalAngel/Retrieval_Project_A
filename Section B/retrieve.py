"""Query-time retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from bm25 import load_bm25, score_bm25_query
from utils import K_EVAL


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:

    corpus_vectors, page_ids = load_index(artifacts_dir)
    bm25_data = load_bm25(artifacts_dir)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    dense_scores_matrix = query_vectors @ corpus_vectors.T
    ranked: List[List[int]] = []

    # RRF Constant
    RRF_K = 10

    for q_idx, query_str in enumerate(queries):
        # --- 1. Get Dense Ranking ---
        dense_scores_raw = {
            page_ids[idx]: float(score)
            for idx, score in enumerate(dense_scores_matrix[q_idx])
        }
        # Sort highest to lowest
        dense_ranked = sorted(
            dense_scores_raw.keys(), key=lambda pid: dense_scores_raw[pid], reverse=True
        )

        # --- 2. Get Sparse Ranking ---
        sparse_scores_raw = score_bm25_query(query_str, bm25_data)
        # Sort highest to lowest
        sparse_ranked = sorted(
            sparse_scores_raw.keys(),
            key=lambda pid: sparse_scores_raw[pid],
            reverse=True,
        )

        # --- 3. Compute RRF Scores ---
        rrf_scores = {}

        # Process Dense Ranks (0-indexed, so we add 1 for the formula)
        for rank, pid in enumerate(dense_ranked):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Process Sparse Ranks
        for rank, pid in enumerate(sparse_ranked):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))

        # --- 4. Final Sort and Extract Top K ---
        sorted_pages = sorted(
            rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True
        )
        ranked.append(sorted_pages[:top_k])

    return ranked
