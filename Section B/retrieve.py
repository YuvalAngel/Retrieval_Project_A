"""Query-time retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from collections import defaultdict

import numpy as np

from embed import embed_queries
from index import load_index
from bm25 import load_bm25, score_bm25_query
from utils import K_EVAL


def search_batch(
    queries,
    *,
    top_k=K_EVAL,
    artifacts_dir=None,
):

    corpus_vectors, page_ids = load_index(artifacts_dir)
    bm25_data = load_bm25(artifacts_dir)

    query_vectors = embed_queries(queries)

    if query_vectors.size == 0:
        return [[] for _ in queries]

    dense_scores_matrix = query_vectors @ corpus_vectors.T

    ranked = []

    # tuning knobs
    DENSE_TOP_K = 20
    BM25_TOP_K = 20

    DENSE_WEIGHT = 0.6
    BM25_WEIGHT = 0.4

    CHUNK_AGG_TOP = 1

    for q_idx, query in enumerate(queries):

        dense_scores = dense_scores_matrix[q_idx]

        # ----------------------------
        # Dense retrieval (top chunks)
        # ----------------------------
        dense_top = np.argpartition(
            dense_scores,
            -DENSE_TOP_K,
        )[-DENSE_TOP_K:]

        # ----------------------------
        # BM25 retrieval
        # ----------------------------
        sparse_scores = score_bm25_query(
            query,
            bm25_data,
        )

        sparse_top = sorted(
            sparse_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:BM25_TOP_K]

        # ----------------------------
        # Aggregate chunk → page
        # ----------------------------
        page_dense = defaultdict(list)
        page_bm25 = defaultdict(list)

        for chunk_idx in dense_top:
            page_id = page_ids[chunk_idx]
            page_dense[page_id].append(float(dense_scores[chunk_idx]))

        for chunk_idx, score in sparse_top:
            page_id = page_ids[chunk_idx]
            page_bm25[page_id].append(float(score))

        page_scores = {}

        all_pages = set(page_dense.keys()) | set(page_bm25.keys())

        for page_id in all_pages:

            dense_vals = page_dense.get(page_id, [])
            bm25_vals = page_bm25.get(page_id, [])

            if dense_vals:
                dense_vals.sort(reverse=True)
                dense_score = sum(dense_vals[:CHUNK_AGG_TOP])
            else:
                dense_score = 0.0

            if bm25_vals:
                bm25_vals.sort(reverse=True)
                bm25_score = sum(bm25_vals[:CHUNK_AGG_TOP])
            else:
                bm25_score = 0.0

            page_scores[page_id] = (
                DENSE_WEIGHT * dense_score
                + BM25_WEIGHT * bm25_score
            )

        ranked_pages = sorted(
            page_scores,
            key=page_scores.get,
            reverse=True,
        )

        ranked.append(ranked_pages[:top_k])

    return ranked

