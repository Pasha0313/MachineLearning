"""Demo: a case where BM25 finds the right passage and cosine similarity
does not, using a real embedding model -- not a hand-built stand-in.

Corpus design note, because it matters and isn't obvious: a corpus of
topically-varied passages (different policy sections, one holding a rare
identifier) was tried first and did NOT reproduce the effect with
all-MiniLM-L6-v2 -- cosine already ranked the correct passage 1st, because
it shares enough ordinary vocabulary with the query ("theft", "deductible",
"policy") for the embedding to do fine without needing to understand the
identifier at all. The corpus below -- near-duplicate "twin" passages that
differ *only* in the policy number -- is what actually isolates the effect:
with every other signal held constant, the identifier is the only thing
that can distinguish the right passage from a wrong one, and a real
semantic embedding has no real way to ground it. This is a more adversarial
corpus than "one PDF's passages," but it's the honest way to demonstrate the
mechanism rather than assume it happens on every document (see README).

Embedding source, in priority order:
  1. sentence-transformers (all-MiniLM-L6-v2), if importable -- offline, no
     API key, and the model used in the reference article.
  2. embedding.make_embed_fn() (OpenAI text-embedding-3-small), if
     OPENAI_API_KEY is set.
  3. Neither available -> print why and exit 1. This demo does not fall
     back to a fake embedder: the whole point is to show what a real
     semantic embedding does with a rare exact-match token, and a
     bag-of-words vector wouldn't reproduce that honestly.

Run: python scripts/demo_keyword_beats_cosine.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from retrieval import HybridRetriever  # noqa: E402

CORPUS = [
    # [0] the one passage that actually answers the question.
    "Policy FOY-HOME-2026-004417: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 150.",

    # near-duplicate "twin" passages: same wording, different policy
    # numbers. This isolates the effect cleanly -- the *only* thing that
    # distinguishes the right passage from a wrong one is the identifier,
    # which is exactly the case a rare alphanumeric code that a semantic
    # embedding has no real way to ground.
    "Policy FOY-HOME-3391-778820: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 150.",
    "Policy FOY-HOME-5512-330198: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 150.",
    "Policy FOY-HOME-8847-192256: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 150.",
    "Policy FOY-HOME-1123-560734: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 150.",
    "Policy FOY-HOME-9004-661829: theft following forcible entry is "
    "covered up to the contents sum insured, subject to a deductible of "
    "EUR 200.",
]

QUERY = "What is the theft deductible under policy FOY-HOME-2026-004417?"


def _try_sentence_transformers():
    try:
        os.environ.setdefault("USE_TF", "0")  # avoid a TF/Keras import clash pulled in by transformers
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    print("Embedding backend: sentence-transformers (all-MiniLM-L6-v2), offline.\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_fn(texts):
        return model.encode(list(texts), convert_to_numpy=True).tolist()

    return embed_fn


def _try_openai():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from embedding import make_embed_fn
    print("Embedding backend: OpenAI text-embedding-3-small (sentence-transformers not installed).\n")
    return make_embed_fn()


def get_embed_fn():
    embed_fn = _try_sentence_transformers()
    if embed_fn is not None:
        return embed_fn
    embed_fn = _try_openai()
    if embed_fn is not None:
        return embed_fn
    print(
        "No embedding backend available: install sentence-transformers "
        "(see requirements-dev.txt) for an offline run, or set "
        "OPENAI_API_KEY in .env to use the OpenAI embedding API instead.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    embed_fn = get_embed_fn()
    retriever = HybridRetriever(CORPUS, embed_fn=embed_fn)

    print(f"Query: {QUERY}\n")
    print(f"Correct passage is index 0: {CORPUS[0][:70]}...\n")

    ranks = {}
    for method in ["dense", "bm25", "hybrid", "weighted"]:
        hits = retriever.search(QUERY, top_k=len(CORPUS), method=method, alpha=0.5)
        rank_of_correct = next(i for i, h in enumerate(hits) if h.index == 0) + 1
        ranks[method] = rank_of_correct
        print(f"[{method}] rank of correct passage: {rank_of_correct} / {len(CORPUS)}")
        for pos, h in enumerate(hits[:3], start=1):
            marker = " <-- correct" if h.index == 0 else ""
            print(f"    {pos}. idx={h.index} score={h.hybrid:.4f}{marker}")
        print()

    print(
        f"cosine: rank {ranks['dense']}/{len(CORPUS)}, "
        f"bm25: rank {ranks['bm25']}/{len(CORPUS)}, "
        f"hybrid (RRF): rank {ranks['hybrid']}/{len(CORPUS)}, "
        f"weighted (score-magnitude fusion): rank {ranks['weighted']}/{len(CORPUS)}"
    )


if __name__ == "__main__":
    main()
