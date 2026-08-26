"""Pipeline: ties the four bricks together.

parse_pdf (parsing.py) -> HybridRetriever (retrieval.py, embed_fn from
embedding.py) -> parse_query + answer (generation.py).

Kept separate from app.py so the same pipeline is used by the Streamlit UI
and by anything else (tests, a future API) without going through Streamlit.
"""

from dataclasses import dataclass
from typing import List

from openai import OpenAI

from embedding import make_embed_fn
from generation import Answer, answer as generate_answer, parse_query
from parsing import Chunk, parse_pdf_bytes
from retrieval import Hit, HybridRetriever


@dataclass
class QueryResult:
    parsed_intent: str
    answer: Answer
    hits: List[Hit]
    retrieved_pages: List[int]
    generation_calls: int = 1


class DocumentIndex:
    """Parses a PDF once, builds the hybrid index once, then answers as many
    queries as asked against it."""

    def __init__(self, pdf_bytes: bytes, client: OpenAI = None):
        self.client = client or OpenAI()
        self.chunks: List[Chunk] = parse_pdf_bytes(pdf_bytes)
        if not self.chunks:
            raise ValueError("No extractable text found in this PDF.")
        texts = [c.text for c in self.chunks]
        self.retriever = HybridRetriever(texts, embed_fn=make_embed_fn(client=self.client))

    def ask(self, query: str, method: str = "hybrid") -> QueryResult:
        """Batch: one retrieval at the LLM-decided top_k, one generation
        call over all of it."""
        parsed = parse_query(query, client=self.client)
        bm25_query = " ".join(parsed.keywords) if parsed.keywords else None
        hits = self.retriever.search(query, top_k=parsed.top_k, method=method, bm25_query=bm25_query)
        pages = [self.chunks[h.index].page for h in hits]
        ans = generate_answer(query, hits, pages, client=self.client)
        return QueryResult(parsed_intent=parsed.intent, answer=ans, hits=hits, retrieved_pages=pages,
                            generation_calls=1)

    def ask_sequential(self, query: str, method: str = "hybrid") -> QueryResult:
        """Top-1-vs-top-k as an explicit, bounded loop instead of a one-shot
        guess: for a lookup question, try the single best passage first and
        ship that answer if the model reports it as both found and
        complete; only escalate to the full top_k passages -- one further
        retrieval, one further generation call -- if it isn't. For a
        comparison/list question, a single top-1 passage is very unlikely to
        be complete by definition, so skip straight to batch and don't waste
        a call finding that out. Either way this makes at most 2 generation
        calls, never more -- the loop is bounded by construction, not by a
        counter."""
        parsed = parse_query(query, client=self.client)
        bm25_query = " ".join(parsed.keywords) if parsed.keywords else None

        if parsed.intent in ("list", "comparison"):
            return self.ask(query, method=method)

        top1_hits = self.retriever.search(query, top_k=1, method=method, bm25_query=bm25_query)
        top1_pages = [self.chunks[h.index].page for h in top1_hits]
        top1_answer = generate_answer(query, top1_hits, top1_pages, client=self.client)

        if top1_answer.answer_found and top1_answer.complete_answer_found:
            return QueryResult(parsed_intent=parsed.intent, answer=top1_answer, hits=top1_hits,
                                retrieved_pages=top1_pages, generation_calls=1)

        full_k = max(parsed.top_k, 1)
        hits = self.retriever.search(query, top_k=full_k, method=method, bm25_query=bm25_query)
        pages = [self.chunks[h.index].page for h in hits]
        ans = generate_answer(query, hits, pages, client=self.client)
        return QueryResult(parsed_intent=parsed.intent, answer=ans, hits=hits, retrieved_pages=pages,
                            generation_calls=2)
