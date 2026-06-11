"""Development evaluation: per-query NDCG, duplicate-aware diagnostics, oracle ceiling.

Public queries repeat the same string with different (disjoint) relevant sets, so a
deterministic system cannot reach NDCG 1.0. The oracle here is the best static top-10
per unique query string given full knowledge of the ground truth — the real ceiling
to compare against. (eval.py and scripts/ are read-only; this is a separate dev tool.)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import load_query_file, mean_ndcg_at_k, ndcg_at_k
from retrieve import search_batch
from utils import K_EVAL, PUBLIC_QUERIES_PATH


def _pos_weight(r: int) -> float:  # matches eval.dcg_at_k: w1=1, wr=1/log2(r) for r>=2
    return 1.0 if r == 1 else 1.0 / math.log2(r)


def _idcg(n_rel: int) -> float:
    return sum(_pos_weight(r) for r in range(1, min(n_rel, K_EVAL) + 1))


def oracle_ceiling(rows) -> float:
    """Best static top-10 per unique query string (greedy = optimal here)."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["query"]].append(set(r["relevant_page_ids"]))
    per_instance = {}
    for q, gts in by_query.items():
        pool = []  # (value_per_slot_weight, instance_idx, pid)
        for i, gt in enumerate(gts):
            for pid in gt:
                pool.append((1.0 / _idcg(len(gt)), i, pid))
        pool.sort(key=lambda x: -x[0])
        scores = [0.0] * len(gts)
        for rank, (val, i, _pid) in enumerate(pool[:K_EVAL], start=1):
            scores[i] += val * _pos_weight(rank)
        per_instance[q] = scores
    total, n = 0.0, 0
    for q, gts in by_query.items():
        for s in per_instance[q]:
            total += s
            n += 1
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lead-weight", type=float, default=0.8)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--number-weight", type=float, default=2.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gts = [set(r["relevant_page_ids"]) for r in rows]

    t0 = time.perf_counter()
    ranked = search_batch(
        queries,
        dense_weight=args.dense_weight,
        lead_weight=args.lead_weight,
        bm25_weight=args.bm25_weight,
        number_weight=args.number_weight,
    )
    elapsed = time.perf_counter() - t0

    mean = mean_ndcg_at_k(ranked, gts, k=K_EVAL)
    oracle = oracle_ceiling(rows)
    print(f"dense_w={args.dense_weight} lead_w={args.lead_weight} "
          f"bm25_w={args.bm25_weight} num_w={args.number_weight}")
    print(f"mean_ndcg@10={mean:.4f}   oracle_ceiling={oracle:.4f}   "
          f"fraction_of_oracle={mean / oracle:.3f}")
    print(f"retrieval_time={elapsed:.1f}s (50 queries, includes artifact+model load)")

    # union-recall diagnostic: how much of each unique string's pooled GT we surface
    by_query = defaultdict(set)
    for r in rows:
        by_query[r["query"]] |= set(r["relevant_page_ids"])
    got = 0
    tot = 0
    first = {}
    for q, rk in zip(queries, ranked):
        first.setdefault(q, rk)
    for q, union in by_query.items():
        hits = len(set(first[q][:K_EVAL]) & union)
        got += hits
        tot += min(len(union), K_EVAL)
    print(f"pooled-GT coverage in top-10 (unique queries): {got}/{tot}")

    if args.verbose:
        for r, rk in zip(rows, ranked):
            s = ndcg_at_k(rk, set(r["relevant_page_ids"]), k=K_EVAL)
            print(f"  {r['query_id']} ndcg={s:.3f} gt={sorted(r['relevant_page_ids'])} "
                  f"top10={rk}")


if __name__ == "__main__":
    main()
