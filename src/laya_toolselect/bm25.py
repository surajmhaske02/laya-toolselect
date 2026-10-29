"""Small in-memory BM25 index over token lists (Okapi BM25, Lucene-style IDF)."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np


class BM25:
    def __init__(self, docs: list, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg = max(float(self.lengths.mean()) if self.n else 1.0, 1.0)
        self.postings = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)

    def scores(self, query_tokens: list) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.norm[i])
        return s
