"""Okapi BM25 over an inverted index.

Each term's contribution to each chunk is computed once, when the index is built. A query
then touches only the postings of its own terms, so its cost grows with the number of
matching chunks rather than the size of the corpus.
"""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import snowballstemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you",
})
_STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str) -> list[str]:
    """Lower-cased word stems without stop words: "Refunded" and "refunds" both -> "refund"."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Bm25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        docs = [Counter(tokenize(text)) for text in texts]
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avg_length = lengths.mean() if len(docs) and lengths.mean() > 0 else 1.0
        # Length normalisation per chunk: k1 * (1 - b + b * |d| / avgdl)
        norm = k1 * (1 - b + b * lengths / avg_length)

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                postings[term].append((i, tf))

        self.size = len(docs)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, entries in postings.items():
            index = np.fromiter((i for i, _ in entries), dtype=np.int32, count=len(entries))
            tf = np.fromiter((t for _, t in entries), dtype=np.float64, count=len(entries))
            # Lucene's idf: always positive, so very common terms still count a little.
            idf = math.log(1 + (self.size - len(entries) + 0.5) / (len(entries) + 0.5))
            self._postings[term] = (index, idf * tf * (k1 + 1) / (tf + norm[index]))

    def top(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        """Indices of the k best-scoring texts; `allowed` is an optional boolean mask."""
        terms = [t for t in set(tokenize(query)) if t in self._postings]
        if not terms or k <= 0:
            return []
        scores = np.zeros(self.size)
        for term in terms:
            index, weight = self._postings[term]
            scores[index] += weight
        if allowed is not None:
            scores[~allowed] = 0.0
        matched = np.flatnonzero(scores)
        if len(matched) > k:
            matched = matched[np.argpartition(scores[matched], -k)[-k:]]
        return matched[np.argsort(-scores[matched], kind="stable")].tolist()
