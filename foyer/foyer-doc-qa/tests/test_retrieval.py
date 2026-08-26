"""Tests for retrieval.py.

Retrieval tests inject a deterministic fake embed_fn (normalized
token-count-bucket vectors over a fixed, precomputed vocabulary) so nothing
here needs a network call or an API key.

The first 11 tests are exactly the set the reviewer's brief asked for, by
name. A few extra tests below them cover things that set (deliberately)
doesn't: `weighted_sum`, `tokenize`, and BM25 on a query with zero corpus
overlap.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import (
    BM25,
    HybridRetriever,
    _rrf_contributions,
    cosine_scores,
    reciprocal_rank_fusion,
    tokenize,
    weighted_sum,
)


def _fake_embed_fn(vocab):
    """A deterministic stand-in for a real embedding model: each text becomes
    a normalized vector of token counts over a fixed vocabulary computed
    ahead of time from every text the test will ever embed (corpus + query),
    so corpus vectors and query vectors always share the same dimensions."""
    index = {w: i for i, w in enumerate(vocab)}

    def embed_fn(texts):
        vectors = []
        for t in texts:
            counts = [0.0] * len(vocab)
            for w in tokenize(t):
                if w in index:
                    counts[index[w]] += 1.0
            norm = math.sqrt(sum(c * c for c in counts)) or 1.0
            vectors.append([c / norm for c in counts])
        return vectors

    return embed_fn


# ---------------------------------------------------------------------------
# the 11 named tests
# ---------------------------------------------------------------------------

def test_bm25_ranks_document_containing_query_term_first():
    corpus = [
        "the deductible for theft is two hundred fifty euros",
        "this document discusses general insurance concepts",
        "roadside assistance is available across the EU",
    ]
    bm25 = BM25([tokenize(d) for d in corpus])
    scores = bm25.score(tokenize("theft deductible"))
    assert scores[0] == max(scores)
    assert scores[0] > 0.0


def test_bm25_idf_is_non_negative_for_common_terms():
    # "policy" appears in every document (n == N) -- the case where the
    # classic Robertson-Sparck-Jones idf formula can go negative.
    corpus = ["policy alpha", "policy beta", "policy gamma", "policy delta"]
    bm25 = BM25([tokenize(d) for d in corpus])
    assert bm25.idf["policy"] >= 0.0


def test_bm25_rare_term_outranks_common_term():
    # "common" appears in 4/5 docs (low idf); "rare" appears in 1/5 (high
    # idf). A query for both should favour the document holding the rare
    # term over documents that only hold the common one, even though those
    # repeat the common term several times.
    corpus = [
        "rare token here",
        "common common common common common",
        "common word document",
        "common information here",
        "common example text",
    ]
    bm25 = BM25([tokenize(d) for d in corpus])
    scores = bm25.score(tokenize("common rare"))
    assert scores[0] > scores[1]


def test_cosine_identical_vectors_is_one():
    scores = cosine_scores([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]])
    assert math.isclose(scores[0], 1.0, rel_tol=1e-9)


def test_cosine_orthogonal_vectors_is_zero():
    scores = cosine_scores([1.0, 0.0], [[0.0, 1.0]])
    assert math.isclose(scores[0], 0.0, abs_tol=1e-9)


def test_rrf_only_credits_documents_a_scorer_actually_matched():
    bm25_scores = [3.0, 0.0, 1.0]   # doc 1 unmatched by bm25
    dense_scores = [0.0, 0.0, 0.5]  # docs 0 and 1 unmatched by dense
    fused = reciprocal_rank_fusion(bm25_scores, dense_scores)
    assert fused[1] == 0.0  # matched by neither -> zero credit, no rank leakage
    assert fused[0] > 0.0 and fused[2] > 0.0  # matched by at least one scorer


def test_rrf_ties_use_competition_ranking():
    # three-way tie for the top score, then one clearly lower score.
    scores = [5.0, 5.0, 5.0, 1.0]
    contrib = _rrf_contributions(scores, k=60)
    assert contrib[0] == contrib[1] == contrib[2] == 1.0 / 60
    # the 4th document's rank is its position in sorted order (3), not 1 --
    # ties don't compress the ranks that come after them.
    assert math.isclose(contrib[3], 1.0 / 63)


def test_rrf_does_not_bury_the_only_bm25_hit_when_dense_is_all_zero():
    """Regression test for a real bug: naive RRF fused over *all* documents,
    so tie-ordering in an all-zero dense list (e.g. no embed_fn configured)
    handed irrelevant documents the same rank credit as the one passage BM25
    had actually found, burying it. Correct RRF only ranks documents each
    scorer actually matched (score > 0)."""
    bm25_scores = [5.0, 0.0, 0.0, 0.0, 0.0]
    dense_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
    fused = reciprocal_rank_fusion(bm25_scores, dense_scores)
    assert fused[0] == max(fused)
    assert all(fused[i] == 0.0 for i in range(1, 5))


def test_hybrid_retriever_without_embed_fn_falls_back_to_bm25():
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    retriever = HybridRetriever(texts, embed_fn=None)
    hits = retriever.search("alpha", top_k=2, method="hybrid")
    assert hits[0].index == 0
    assert hits[0].dense == 0.0


def test_hybrid_retriever_with_embed_fn_uses_dense_scores():
    texts = ["zero indexed content", "one indexed content"]
    vocab = sorted({w for t in texts + ["zero"] for w in tokenize(t)})
    embed_fn = _fake_embed_fn(vocab)
    retriever = HybridRetriever(texts, embed_fn=embed_fn)
    hits = retriever.search("zero", top_k=2, method="dense")
    assert hits[0].index == 0
    assert hits[0].dense > 0.0
    assert hits[0].dense > hits[1].dense


def test_hybrid_retriever_bm25_query_override_changes_ranking():
    # doc 0 is only findable by BM25 via "zero"; doc 1 only via "one".
    texts = ["zero indexed content", "one indexed content"]
    vocab = sorted({w for t in texts + ["zero", "one"] for w in tokenize(t)})
    embed_fn = _fake_embed_fn(vocab)
    retriever = HybridRetriever(texts, embed_fn=embed_fn)

    raw_query = "zero"   # embeds toward doc 0
    bm25_query = "one"   # BM25-matches doc 1 only

    bm25_hits = retriever.search(raw_query, top_k=2, method="bm25", bm25_query=bm25_query)
    assert bm25_hits[0].index == 1  # used bm25_query, not the raw query

    dense_hits = retriever.search(raw_query, top_k=2, method="dense", bm25_query=bm25_query)
    assert dense_hits[0].index == 0  # dense still used the raw query, unaffected


# ---------------------------------------------------------------------------
# extra coverage beyond the reviewer's prompt-specified 11
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_strips_stopwords():
    assert tokenize("The Policy Number is FOY-2026-1") == ["policy", "number", "foy", "2026", "1"]
    assert "the" not in tokenize("the quick brown fox")


def test_bm25_zero_score_for_no_overlap():
    corpus = ["apples and oranges", "cars and trucks"]
    bm25 = BM25([tokenize(d) for d in corpus])
    assert bm25.score(tokenize("zebra")) == [0.0, 0.0]


def test_weighted_sum_normalises_before_combining():
    bm25 = [0.0, 10.0, 100.0]
    dense = [0.9, 0.5, 0.1]
    combined = weighted_sum(bm25, dense, alpha=0.5)
    assert combined[2] == max(combined)  # bm25's top pick still wins at alpha=0.5
    assert all(0.0 <= c <= 1.0 for c in combined)
