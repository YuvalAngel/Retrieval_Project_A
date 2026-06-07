# Section B - Retrieval pipeline

This submission uses the required `sentence-transformers/all-MiniLM-L6-v2`
dense embeddings, a full-page BM25 index, and an offline structural index for
the generated entity families in the corpus.

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).

## Submitted artifacts

The grader should load these files directly from `artifacts/`:

- `index_vectors.npy` - normalized MiniLM page-intro embeddings
- `index_meta.json` - page ID order for the dense matrix
- `bm25_index.json.gz` - compressed full-corpus sparse lexical index
- `structure_index.json` - page family metadata and clue flags for reranking

## Build index (offline, not timed)

Run once locally to create `artifacts/`. **Submit these files** in your repo; staff do not rebuild the index at grading time.

```bash
python scripts/build_index.py
```

Rebuilding is not required for grading as long as the four artifact files above
are present.

## Public self-test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

Current public-query result:

```text
mean_ndcg@10=0.3618
```

## Submit

Public GitHub repo with this code, **required** `artifacts/`, and a concise README documenting artifact paths. See the assignment PDF for video and grading details.
