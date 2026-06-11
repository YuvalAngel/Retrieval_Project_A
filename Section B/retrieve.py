"""Query-time retrieval: hybrid scoring over complementary signals.

The corpus is synthetic and template-based: entity clusters share near-identical
sentences, and a query names a few facts that identify one cluster. Signals are
fused with per-query z-score normalization:

1. dense max  — best passage similarity per page (handles paraphrase), computed
   for BOTH passage granularities (100 and 150 words) and ensembled: the
   ensemble beats either granularity alone (multi-scale matching);
2. dense lead — similarity of the page's lead passage, which carries the
   entity-defining sentence (who/what the page is about), per granularity;
3. bm25       — full-page lexical score; rare template terms discriminate, and
   full length normalization (b=1) suppresses long real-Wikipedia distractors;
4. numbers    — digit groups from the query ("1987", "1,456,779", with decade
   expansion "1820s" -> 1820..1829) matched via the BM25 postings, IDF-weighted;
   the instance-identifying token in these queries is usually a number that
   embeddings are blind to.

Connective queries ("what links X, Y and Z", "how do ... connect") ask for a
SET of pages about one entity (player + season + arena pages), so for queries
matching that linguistic pattern the top-200 candidates are regrouped by
precomputed entity cluster (see index._cluster_pages) and clusters are ranked
by their two best member scores — whole relevant clusters then fill the top-10
instead of scattered single pages. Pointwise queries are left untouched
(grouping measurably hurts them).

Negative results kept out of the pipeline (measured on public queries): query
facet decomposition, passage-level BM25, MMR diversification, cluster expansion
via title queries, pseudo-relevance feedback, RRF fusion, sum-of-top-3 passage
pooling, title-BM25, declarative query rewriting, doubled-title embeddings.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bm25 import load_bm25, score_bm25_query
from embed import embed_queries
from index import load_index
from utils import K_EVAL

DENSE_WEIGHT = 1.0
LEAD_WEIGHT = 0.8
BM25_WEIGHT = 1.0
NUMBER_WEIGHT = 2.5
CLUSTER_CANDIDATES = 200
_NUM_RE = re.compile(r"\b(\d{1,4})(s?)\b")
_CONNECTIVE_RE = re.compile(r"\b(links|connect|together|learned about)\b", re.I)
_state: Dict[str, Any] = {}


def _load_state(artifacts_dir: Optional[Path]) -> Dict[str, Any]:
    key = str(artifacts_dir)
    if _state.get("key") != key:
        vector_sets, page_ids, clusters = load_index(artifacts_dir)
        _state.update(
            key=key,
            vector_sets=vector_sets,
            page_ids=page_ids,
            clusters=clusters,
            bm25=load_bm25(artifacts_dir),
        )
    return _state


def _zscore(row: np.ndarray) -> np.ndarray:
    return (row - row.mean()) / max(float(row.std()), 1e-9)


def _number_groups(query: str) -> List[List[str]]:
    """Digit groups in the query; decades expand to their ten member years."""
    groups: List[List[str]] = []
    for m in _NUM_RE.finditer(query.lower()):
        num, decade = m.group(1), m.group(2)
        if decade and len(num) == 4 and num.endswith("0"):
            groups.append([num[:3] + str(d) for d in range(10)])
        else:
            groups.append([num])
    return groups


def _number_score(query: str, bm: Dict[str, Any]) -> Optional[np.ndarray]:
    """IDF-weighted count of matched query digit-groups per page (binary per group)."""
    groups = _number_groups(query)
    if not groups:
        return None
    scores = np.zeros(bm["n_docs"], dtype=np.float32)
    vocab, offsets = bm["vocab"], bm["offsets"]
    for alts in groups:
        present = np.zeros(bm["n_docs"], dtype=np.float32)
        idf_g = 0.0
        for term in alts:
            ti = vocab.get(term)
            if ti is None:
                continue
            lo, hi = offsets[ti], offsets[ti + 1]
            present[bm["doc_idx"][lo:hi]] = 1.0
            idf_g = max(idf_g, float(bm["idf"][ti]))
        scores += idf_g * present
    return scores if scores.any() else None


def _rerank_clusters(score: np.ndarray, clusters: np.ndarray, k: int) -> List[int]:
    """Regroup top candidates by entity cluster; rank clusters by top-2 members."""
    m = min(CLUSTER_CANDIDATES, score.shape[0])
    cand = np.argpartition(-score, m - 1)[:m]
    cand = cand[np.argsort(-score[cand])]
    members: Dict[int, List[int]] = {}
    for i in cand:
        members.setdefault(int(clusters[i]), []).append(int(i))
    cluster_score = {
        c: sum(sorted((float(score[i]) for i in ms), reverse=True)[:2])
        for c, ms in members.items()
    }
    out: List[int] = []
    for c in sorted(cluster_score, key=cluster_score.get, reverse=True):
        for i in members[c]:
            out.append(i)
            if len(out) == k:
                return out
    return out


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    dense_weight: float = DENSE_WEIGHT,
    lead_weight: float = LEAD_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    number_weight: float = NUMBER_WEIGHT,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    if not queries:
        return []
    st = _load_state(artifacts_dir)
    vector_sets, page_ids = st["vector_sets"], st["page_ids"]
    n_pages = page_ids.shape[0]

    emb = embed_queries(queries)
    per_set = []
    for vectors, page_offsets in vector_sets:
        starts = page_offsets[:-1]
        chunk_scores = emb @ vectors.T                                 # (nq, n_chunks)
        dense_max = np.maximum.reduceat(chunk_scores, starts, axis=1)  # (nq, n_pages)
        dense_lead = emb @ vectors[starts].T                           # (nq, n_pages)
        per_set.append((dense_max, dense_lead))

    ranked: List[List[int]] = []
    for qi, q in enumerate(queries):
        score = np.zeros(n_pages, dtype=np.float32)
        for dense_max, dense_lead in per_set:
            score = score + dense_weight * _zscore(dense_max[qi])
            score = score + lead_weight * _zscore(dense_lead[qi])
        if bm25_weight > 0.0:
            score = score + bm25_weight * _zscore(score_bm25_query(q, st["bm25"]))
        if number_weight > 0.0:
            ns = _number_score(q, st["bm25"])
            if ns is not None:
                score = score + number_weight * _zscore(ns)
        k = min(top_k, n_pages)
        if _CONNECTIVE_RE.search(q):
            top = _rerank_clusters(score, st["clusters"], k)
        else:
            idx = np.argpartition(-score, k - 1)[:k]
            top = idx[np.argsort(-score[idx])].tolist()
        ranked.append([int(page_ids[i]) for i in top])
    return ranked
