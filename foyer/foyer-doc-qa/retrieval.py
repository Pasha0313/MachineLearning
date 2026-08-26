"""Retrieval.

Two independent scorers and a fusion step, all implemented directly:

  1. BM25  -- classic sparse keyword ranking, written from scratch.
  2. Dense -- cosine similarity over embedding vectors.
  3. Hybrid -- Reciprocal Rank Fusion (default) or normalised weighted sum.

Why hybrid: keyword search nails exact/rare tokens (a policy number, a proper
noun, a code) that a dense model can smooth away; dense search catches
paraphrase and synonymy that keyword search misses. Fusing them beats either
alone. `scripts/demo_keyword_beats_cosine.py` shows a concrete case where BM25
retrieves the correct chunk and cosine does not.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small stopword list. Kept short on purpose: over-aggressive stopping hurts
# exact-match queries (e.g. removing "no" from "no claims bonus").
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "as", "at", "by", "with", "this", "that", "it",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 (from scratch)
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25. Standard defaults k1=1.5, b=0.75."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [list(doc) for doc in corpus_tokens]
        self.N = len(self.corpus_tokens)
        self.doc_len = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(doc) for doc in self.corpus_tokens]

        # document frequency: in how many docs does each term appear
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.df = df

        # idf with the BM25 +0.5 smoothing; floored at 0 so common terms
        # can't contribute negative scores.
        self.idf = {
            term: max(0.0, math.log(1 + (self.N - n + 0.5) / (n + 0.5)))
            for term, n in df.items()
        }

    def score(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0] * self.N
        q_terms = set(query_tokens)
        for i in range(self.N):
            tf_i = self.tf[i]
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0))
            s = 0.0
            for term in q_terms:
                f = tf_i.get(term, 0)
                if f == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores


# ---------------------------------------------------------------------------
# dense cosine
# ---------------------------------------------------------------------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_scores(query_vec: Sequence[float], doc_vecs: Sequence[Sequence[float]]) -> List[float]:
    qn = _norm(query_vec) or 1e-12
    out = []
    for v in doc_vecs:
        out.append(_dot(query_vec, v) / (qn * (_norm(v) or 1e-12)))
    return out


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def _minmax(xs: Sequence[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _ranks(scores: Sequence[float]) -> List[int]:
    """Return 0-based rank of each index (0 = best)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = [0] * len(scores)
    for r, idx in enumerate(order):
        rank[idx] = r
    return rank


def weighted_sum(bm25: Sequence[float], dense: Sequence[float], alpha: float = 0.5) -> List[float]:
    """alpha weights the dense score; (1-alpha) weights BM25. Both min-max
    normalised first so the two very different scales are comparable."""
    b, d = _minmax(bm25), _minmax(dense)
    return [alpha * di + (1 - alpha) * bi for bi, di in zip(b, d)]


def _rrf_contributions(scores: Sequence[float], k: int) -> List[float]:
    """RRF contribution from one scorer. Only documents the scorer actually
    matched (score > 0) get credit; everything else contributes 0. Tied scores
    share a rank (competition ranking) so arbitrary tie-order can't leak
    ranking signal — the bug that naive all-docs RRF has when one scorer
    returns a flat list of zeros."""
    contrib = [0.0] * len(scores)
    matched = [i for i in range(len(scores)) if scores[i] > 0]
    matched.sort(key=lambda i: scores[i], reverse=True)
    rank = 0
    prev = None
    for pos, i in enumerate(matched):
        if prev is None or scores[i] < prev:
            rank = pos  # new (worse) rank only when the score strictly drops
        contrib[i] = 1.0 / (k + rank)
        prev = scores[i]
    return contrib


def reciprocal_rank_fusion(bm25: Sequence[float], dense: Sequence[float], k: int = 60) -> List[float]:
    """RRF: robust, scale-free. score_i = 1/(k+rank_bm25) + 1/(k+rank_dense),
    summed only over the scorers that actually matched document i."""
    cb = _rrf_contributions(bm25, k)
    cd = _rrf_contributions(dense, k)
    return [cb[i] + cd[i] for i in range(len(bm25))]


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    index: int
    text: str
    bm25: float
    dense: float
    hybrid: float


class HybridRetriever:
    """Indexes a list of texts and retrieves with BM25, dense, or hybrid.

    `embed_fn` maps a list of strings to a list of vectors. It is injected so
    the retriever is testable without a model and the embedding backend is
    swappable (local sentence-transformers, OpenAI, etc.).
    """

    def __init__(
        self,
        texts: List[str],
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.texts = texts
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embed_fn = embed_fn
        self.doc_vecs: Optional[List[List[float]]] = None
        if embed_fn is not None and texts:
            self.doc_vecs = embed_fn(texts)

    def search(self, query: str, top_k: int = 5, method: str = "hybrid",
               alpha: float = 0.5, bm25_query: Optional[str] = None) -> List[Hit]:
        """`bm25_query`, if given, is tokenized and scored against BM25
        instead of `query` -- letting a caller feed BM25 a keyword-extracted
        version of the question (verbatim identifiers, vocabulary mapped to
        the document's own terms) while `query` still goes to the dense
        scorer as the natural-language question it actually is."""
        bm25_scores = self.bm25.score(tokenize(bm25_query if bm25_query is not None else query))

        if self.embed_fn is not None and self.doc_vecs is not None:
            qvec = self.embed_fn([query])[0]
            dense_scores = cosine_scores(qvec, self.doc_vecs)
        else:
            dense_scores = [0.0] * len(self.texts)

        if method == "bm25":
            final = bm25_scores
        elif method == "dense":
            final = dense_scores
        elif method == "weighted":
            final = weighted_sum(bm25_scores, dense_scores, alpha=alpha)
        else:  # hybrid / rrf
            final = reciprocal_rank_fusion(bm25_scores, dense_scores)

        order = sorted(range(len(self.texts)), key=lambda i: final[i], reverse=True)
        hits = [
            Hit(
                index=i,
                text=self.texts[i],
                bm25=bm25_scores[i],
                dense=dense_scores[i],
                hybrid=final[i],
            )
            for i in order[:top_k]
        ]
        return hits