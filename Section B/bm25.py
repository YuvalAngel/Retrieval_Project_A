"""BM25 Sparse Retrieval Implementation with Tokenization."""

from __future__ import annotations

import json
import math
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from utils import ensure_artifacts_dir

BM25_INDEX_NAME = "bm25_index.json"

# English stop words filter
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now"
}


def tokenize(text: str) -> List[str]:
    """Cleans text by removing punctuation and stop words."""
    text = text.lower()
    # Replace punctuation with space so things like "Connect-4" become "connect 4"
    for p in string.punctuation:
        text = text.replace(p, " ")
    return [w for w in text.split() if w not in STOP_WORDS]


def build_bm25(
    records: List[Dict[str, Any]], artifacts_dir: Path | None = None
) -> None:
    """Builds and saves the BM25 inverted index offline."""
    out_dir = artifacts_dir or ensure_artifacts_dir()

    N = len(records)
    df: Counter = Counter()
    doc_lengths: Dict[int, int] = {}
    tf_index: Dict[str, Dict[int, int]] = defaultdict(dict)
    total_length = 0

    for record in records:
        pid = int(record["page_id"])
        text = f"{record.get('title', '')} {record.get('content', '')}"

        # Use our new clean tokenizer instead of raw splitting
        tokens = tokenize(text)
        length = len(tokens)

        doc_lengths[pid] = length
        total_length += length

        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            tf_index[term][pid] = count
            df[term] += 1

    avgdl = total_length / N if N > 0 else 0.0

    idf: Dict[str, float] = {}
    for term, doc_count in df.items():
        idf[term] = math.log(((N - doc_count + 0.5) / (doc_count + 0.5)) + 1.0)

    artifact = {
        "N": N,
        "avgdl": avgdl,
        "idf": idf,
        "doc_lengths": doc_lengths,
        "tf_index": tf_index,
    }

    out_path = out_dir / BM25_INDEX_NAME
    out_path.write_text(json.dumps(artifact), encoding="utf-8")


def load_bm25(artifacts_dir: Path | None = None) -> Dict[str, Any]:
    """Loads the precomputed BM25 index."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_path = out_dir / BM25_INDEX_NAME
    return json.loads(out_path.read_text(encoding="utf-8"))


def score_bm25_query(
    query: str, bm25_data: Dict[str, Any], k1: float = 1.4, b: float = 1.5
) -> Dict[int, float]:
    """Calculates sparse BM25 scores using clean query tokens."""
    # Apply the exact same tokenization to the user query
    tokens = tokenize(query)
    scores: Dict[int, float] = defaultdict(float)

    idf = bm25_data["idf"]
    tf_index = bm25_data["tf_index"]
    doc_lengths = bm25_data["doc_lengths"]
    avgdl = bm25_data["avgdl"]

    for term in tokens:
        if term not in tf_index:
            continue

        term_idf = idf.get(term, 0.0)

        for pid_str, tf in tf_index[term].items():
            pid = int(pid_str)
            L = doc_lengths.get(pid_str, avgdl)

            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (L / avgdl))
            scores[pid] += term_idf * (numerator / denominator)

    return dict(scores)
