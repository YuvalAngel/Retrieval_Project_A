# Section B — Retrieval Pipeline

End-to-end retrieval over 27,074 Wikipedia-style pages: `run(queries)` returns a
ranked list of `page_id`s per query, scored by mean NDCG@10.

**Video:** [link TBD — added before submission]

## Quick start (grading flow)

Dependencies (`numpy`, `sentence-transformers`, `faiss-cpu`) are assumed
installed per `requirements.txt`. The prebuilt index ships in `artifacts/`,
so a fresh clone evaluates immediately — no rebuild:

```bash
python scripts/eval_public.py
```

Expected output: `mean_ndcg@10=0.4815` on the 29 public queries (~6 s query
phase on a GPU machine; the autograder calls `main.run(queries)` the same way).

## Pipeline

| Stage | File | Method |
|---|---|---|
| Chunk | `chunk.py` | Sentence-packed passages with one-sentence overlap, title prepended; built at **two granularities** (~100 and ~150 words) |
| Embed | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized, float16 storage |
| Index | `index.py` | Dense passage matrices per granularity (sharded `.npy`), page-level BM25 postings (`bm25.py`), and offline entity clusters from lead-passage similarity |
| Retrieve | `retrieve.py` | Per-query z-score fusion of: dense passage max + lead-passage similarity (both granularities, ensembled), BM25 (k1=2.0, b=1.0), and IDF-weighted number matching with decade expansion ("1820s" matches 1820–1829). Connective queries ("what links…", "how do … connect") are additionally regrouped by entity cluster so whole relevant clusters fill the top-10. |

Design choices are empirical — measured on the public queries against the
alternatives they replaced. `dev_eval.py` reproduces per-query diagnostics and
the oracle ceiling.

## Artifacts (`artifacts/`, ~872 MB)

| File(s) | Format | Contents |
|---|---|---|
| `vectors_w100_f16.partNN.npy` (6 parts) | float16 `(n_chunks, 384)` | Passage embeddings, 100-word granularity (677,423 chunks), rows grouped by page; parts concatenate in order |
| `vectors_w150_f16.partNN.npy` (4 parts) | float16 | Same, 150-word granularity (452,420 chunks) |
| `page_offsets_w100.npy`, `page_offsets_w150.npy` | int64 `(n_pages+1,)` | Row ranges: page *p* owns vector rows `[off[p], off[p+1])` |
| `page_ids.npy` | int64 `(n_pages,)` | `page_id` for each page position |
| `page_clusters.npy` | int64 `(n_pages,)` | Entity-cluster label per page (union-find over lead-passage cosine > 0.75) |
| `bm25_postings.npz` | npz of int/float arrays | CSR-style inverted index over full page text (offsets, doc ids, term frequencies, IDF, doc lengths) |
| `bm25_vocab.json` | JSON `{term: row}` | Vocabulary for the postings (hapax terms dropped) |

Vector files are sharded below GitHub's 100 MB limit on purpose — plain git,
no Git LFS required to clone.

## Rebuilding the index (optional, offline)

Place the corpus at `data/Wikipedia Entries/` (one JSON per page, as in the
course bundle — not tracked in git), then:

```bash
python scripts/build_index.py   # ~75 min on a Tesla M60 (two embedding passes)
```
