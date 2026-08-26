"""Generation: query parsing + cited answering.

Two explicit LLM calls, both made directly against the OpenAI SDK -- no
agent framework deciding when or how to call the model.

1. parse_query   -- turns the user's question into search keywords (kept
   verbatim, not stemmed/lowercased away, because identifiers like policy
   numbers must survive into the BM25 query) and a retrieval breadth:
   a single best passage for a direct lookup, more for a comparison/list
   question. This is the "top-1 vs top-k" decision from the loop-engineering
   article -- made once per query, before retrieval runs.

2. answer         -- generates strictly from the retrieved passages, citing
   them as [n], and refuses with a fixed sentence when the passages don't
   support an answer. The refusal string is fixed and checked verbatim so
   the caller (and the UI) can detect a "not found" reliably instead of
   pattern-matching free text. It also reports, per call, whether the given
   passages contained *any* relevant answer and whether they contained a
   *complete* one -- `pipeline.DocumentIndex.ask_sequential` uses that to
   decide whether a single top-1 passage was enough or whether it needs to
   escalate to more passages (the top-1-vs-top-k idea, made an explicit,
   bounded loop instead of a one-shot guess).
"""

import json
import os
from dataclasses import dataclass, field
from typing import List

from openai import OpenAI

from retrieval import Hit, tokenize

NOT_FOUND = "NOT FOUND IN DOCUMENT."

_QUERY_PARSE_SYSTEM = """You turn a user's question about a document into a search plan.
Return strict JSON: {"keywords": [string, ...], "intent": "lookup"|"comparison"|"list"|"other", "top_k": integer}.
- keywords: the exact words/phrases/identifiers worth searching for verbatim (numbers, codes, proper nouns, key terms). Do not paraphrase them.
- intent: "lookup" for a single fact, "comparison" for comparing two or more things, "list" for enumerating multiple items, "other" otherwise.
- top_k: how many passages retrieval should return. 1-2 for "lookup", 5-8 for "comparison" or "list", 3 for "other"."""

_ANSWER_SYSTEM = f"""You answer questions using ONLY the numbered passages given to you.
Return strict JSON: {{"answer": string, "answer_found": boolean, "complete_answer_found": boolean}}.
Rules:
- Every claim in "answer" must be supported by a passage, cited inline as [n] where n is the passage number.
- Do not use outside knowledge, do not guess, do not fill gaps.
- answer_found: true if the passages contain at least a partial answer to the question, false otherwise.
- complete_answer_found: true only if the passages fully answer the question and more passages would not add anything needed. If you are missing information that more passages might contain (e.g. this looks like one item from a longer list, or only one side of a comparison), set this false even if answer_found is true.
- If answer_found is false, "answer" must be EXACTLY this sentence and nothing else: "{NOT_FOUND}", and complete_answer_found must be false.
- Be concise."""


@dataclass
class ParsedQuery:
    keywords: List[str] = field(default_factory=list)
    intent: str = "other"
    top_k: int = 3
    raw: str = ""


@dataclass
class Answer:
    text: str
    not_found: bool
    citations: List[int]
    pages: List[int]
    answer_found: bool = False
    complete_answer_found: bool = False


def _client(client: OpenAI = None) -> OpenAI:
    return client or OpenAI()


def _chat_model() -> str:
    return os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def parse_query(query: str, client: OpenAI = None) -> ParsedQuery:
    """LLM-based query parsing, with a naive-tokenizer fallback so a parse
    failure (bad JSON, API error, timeout) never sinks the whole request --
    it just falls back to plain keyword search at a safe default breadth."""
    try:
        resp = _client(client).chat.completions.create(
            model=_chat_model(),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _QUERY_PARSE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        keywords = [str(k) for k in data.get("keywords", [])] or tokenize(query)
        intent = data.get("intent") if data.get("intent") in {"lookup", "comparison", "list", "other"} else "other"
        top_k = int(data.get("top_k", 3))
        top_k = max(1, min(top_k, 10))
        return ParsedQuery(keywords=keywords, intent=intent, top_k=top_k, raw=query)
    except Exception:
        return ParsedQuery(keywords=tokenize(query), intent="other", top_k=3, raw=query)


def _format_passages(hits: List[Hit], pages: List[int]) -> str:
    lines = []
    for n, (hit, page) in enumerate(zip(hits, pages), start=1):
        lines.append(f"[{n}] (page {page})\n{hit.text}")
    return "\n\n".join(lines)


def _extract_citations(text: str, n_passages: int) -> List[int]:
    import re
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in found if 1 <= i <= n_passages)


def _not_found_answer() -> Answer:
    return Answer(
        text=NOT_FOUND, not_found=True, citations=[], pages=[],
        answer_found=False, complete_answer_found=False,
    )


def answer(query: str, hits: List[Hit], pages: List[int], client: OpenAI = None) -> Answer:
    """Generate a cited answer from retrieved hits, or the fixed not-found
    refusal if the model decides the passages don't support one. Also
    reports `answer_found` / `complete_answer_found` so a caller can decide
    whether to escalate to more passages (see `ask_sequential` in
    pipeline.py). If the model's JSON response can't be parsed, this falls
    back to the conservative not-found answer rather than guessing at
    partial JSON -- an unparseable response is not evidence the passages
    support an answer.

    `pages` gives the source page number for each hit, in the same order,
    so citations can be mapped back to a page for the UI."""
    if not hits:
        return _not_found_answer()

    passages = _format_passages(hits, pages)
    resp = _client(client).chat.completions.create(
        model=_chat_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\nPassages:\n{passages}"},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        text = str(data["answer"]).strip()
        answer_found = bool(data.get("answer_found", False))
        complete_answer_found = bool(data.get("complete_answer_found", False))
    except (json.JSONDecodeError, KeyError, TypeError):
        return _not_found_answer()

    if not answer_found or text.rstrip(".") == NOT_FOUND.rstrip("."):
        return _not_found_answer()

    citations = _extract_citations(text, len(hits))
    cited_pages = sorted({pages[i - 1] for i in citations})
    return Answer(
        text=text, not_found=False, citations=citations, pages=cited_pages,
        answer_found=True, complete_answer_found=complete_answer_found,
    )
