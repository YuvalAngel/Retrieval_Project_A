# Retrieval Project A

- **[Section A](Section%20A/)** — dynamic vector database index (`vector_index.py`):
  exact search via one BLAS matmul with a block-max prescreen top-k selection,
  O(1) swap-with-last deletion, and an exact query-result cache. Scores 1.0 on
  all public scenarios with runtime multiplier 1.0. Submitted separately as a
  zip per the assignment instructions.
- **[Section B](Section%20B/)** — end-to-end retrieval pipeline over the
  Wikipedia-style corpus. See [Section B/README.md](Section%20B/README.md) for
  setup, artifact documentation, and how to run the evaluation; the prebuilt
  index ships in the repo, so `python scripts/eval_public.py` works on a fresh
  clone with dependencies installed.

**Video:** https://drive.google.com/file/d/1xrsAp4msgkaYTxJVLo1oK0Muu1PoBUYI/view?usp=sharing
