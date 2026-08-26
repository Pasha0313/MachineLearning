# Foyer Doc-QA

A document chatbot: upload a PDF, ask questions, get answers cited back to
the page they came from -- or an explicit "not found" when the document
doesn't support an answer.

Built for a technical assessment. Four bricks, each hand-written, no
retrieval framework in between:

| Brick | File | What it does |
|---|---|---|
| Parse | [parsing.py](parsing.py) | PyMuPDF text extraction, chunked per page, page number kept on every chunk |
| Retrieve | [retrieval.py](retrieval.py) | BM25 from scratch + cosine over embeddings + Reciprocal Rank Fusion |
| Embed | [embedding.py](embedding.py) | thin OpenAI embeddings call, injected into the retriever |
| Parse query / Generate | [generation.py](generation.py) | LLM extracts search keywords + retrieval breadth, then answers strictly from retrieved passages with `[n]` citations or a fixed refusal |
| Glue | [pipeline.py](pipeline.py) | wires the above into one `DocumentIndex` |
| UI | [app.py](app.py) | Streamlit front end |

No LangChain, no LlamaIndex, no vector DB. The slicing (parsing.py), the
indexing (retrieval.py), and the model calls (embedding.py, generation.py)
are all explicit and readable end to end.

## Why hybrid retrieval

Dense embeddings are good at "what is this passage about" and bad at "does
this passage contain this exact rare token" -- a policy number, a code, a
proper noun. BM25 is the reverse. `scripts/demo_keyword_beats_cosine.py`
tests this with a real embedding model (`sentence-transformers`, offline, or
the project's own OpenAI embedding call if that's not installed -- never a
fake/bag-of-words stand-in). Run it:

```bash
python scripts/demo_keyword_beats_cosine.py
```

**Corpus design note, because the first attempt was wrong and it's worth
saying so.** A corpus of topically-varied policy passages -- different
sections, one holding a rare identifier, the rest ordinary coverage text --
was tried first, matching the shape of a single real document. It did
**not** reproduce the effect: with `all-MiniLM-L6-v2`, cosine already ranked
the correct passage 1st, because it shares enough ordinary vocabulary with
the query ("theft", "deductible", "policy") that the embedding does fine
without needing to understand the identifier at all. That held at both 6 and
12 passages. So the demo instead uses six near-duplicate "twin" passages
that differ *only* in the policy number -- the only corpus design, of the
two actually tried with a real model, that isolates the identifier as the
sole distinguishing signal and reproduces the effect honestly. It's a more
adversarial case than "passages from one PDF," and the Scope section below
is explicit about that: this app only ever indexes one document at a time,
so the near-duplicate-policy scenario this demo is built on doesn't occur
yet in what's actually shipped -- it's a real failure mode, demonstrated
honestly, for a case the current single-PDF app doesn't happen to hit.

Measured output (sentence-transformers, `all-MiniLM-L6-v2`):

```
cosine: rank 5/6, bm25: rank 1/6, hybrid (RRF): rank 4/6, weighted (score-magnitude fusion): rank 1/6
```

Cosine alone buries the correct passage at 5th of 6 -- the six passages are
worded almost identically, so a semantic embedding, which has no real way to
ground a specific alphanumeric code, can barely tell them apart. BM25 finds
it cleanly at rank 1, because the identifier tokens are rare (document
frequency 1) and get a large IDF weight.

**Hybrid (RRF), the app's default, only gets it to rank 4 here -- and that's
a real, honest finding about RRF, not a rounding error, and not left
unresolved.** Reciprocal Rank Fusion only ever looks at *rank*, never at
score magnitude. BM25's margin is enormous (score 3.45 for the correct
passage vs. 0.37 for every wrong one -- roughly 9x), but RRF collapses that
entire margin into a single rank-position of advantage over the tied wrong
passages. Dense's much noisier, much smaller cosine differences (0.79 vs
0.77 -- under 3%) get exactly the same per-rank weight. With five
near-identical "twin" documents in play, that's enough for dense's
weak-but-nonzero ranking signal to outvote BM25's decisive one. This is a
known limitation of vanilla RRF, not a bug in `_rrf_contributions` -- it's
the tradeoff RRF makes for being scale-free and not needing the two scorers'
raw scores to be comparable.

**`retrieval.weighted_sum` -- score magnitude instead of rank -- fixes it on
this exact corpus: rank 1/6.** It's the same two BM25/dense scores, just
combined by (min-max-normalised) magnitude instead of rank, so BM25's
decisive 9x margin actually counts for something instead of being flattened
to one rank-position. It's wired into the same retriever
(`HybridRetriever.search(method="weighted")`) and selectable live in the
app's sidebar. It isn't the default -- `hybrid` (RRF) is -- because it needs
an `alpha` to tune and RRF doesn't; RRF is the safer default across cases
where the two scorers' raw scales aren't known in advance, and `weighted` is
the documented answer for the specific failure mode above, not a strictly
better replacement for it. Both are one dropdown away from each other in the
running app, which is the honest way to let a reviewer see the tradeoff
rather than just read a paragraph asserting it.

Separately, and independently of the RRF-vs-magnitude tradeoff above:
`reciprocal_rank_fusion` fuses only over the documents each scorer actually
matched (score > 0), with tie-safe competition ranking. An earlier version
fused over *all* documents including ties in an all-zero list, which let
irrelevant documents borrow rank credit and bury the one passage BM25 had
actually found -- caught by
`tests/test_retrieval.py::test_rrf_does_not_bury_the_only_bm25_hit_when_dense_is_all_zero`.
That bug is fixed; the RRF-vs-magnitude tradeoff above is not a bug and
isn't "fixable" in the same sense -- it's what you sign up for by using rank
instead of magnitude, which is why `weighted` exists as the alternative.

## Why a query-parsing step before retrieval

The LLM call in `generation.parse_query` does two things before any
retrieval happens: it pulls out the exact keywords/identifiers worth
searching for (kept verbatim, not paraphrased -- a policy number must
survive into the BM25 query untouched), and it decides a retrieval breadth
`top_k` -- small for a direct lookup question, larger for a comparison or
list question. If the LLM call fails for any reason (timeout, bad JSON, API
error), it falls back to plain keyword tokenization at a fixed top-3 -- a
parse failure degrades retrieval quality, it doesn't break the request.

Intent classification is a judgment call, not a guarantee: "What does the
policy exclude?" was classified `lookup` rather than `list` in testing, even
though it's asking for an enumeration. It still answered correctly here
because the whole exclusions section fits in one page/chunk of the sample
document -- but on a document where a list spans multiple chunks, a
misclassified `lookup` (small `top_k`) could truncate the list. This isn't
something the code corrects; it's a real limit of asking an LLM to classify
intent from one line of text.

## Top-1 vs top-k: batch and sequential generation

`parse_query`'s `top_k` guess is a one-shot bet, made before generation has
seen anything. `DocumentIndex` offers two ways to spend it, and the app's
sidebar toggle switches between them:

- **`ask()` -- batch.** One retrieval at the guessed `top_k`, one generation
  call over all of it. Simple, and correct for comparison/list questions,
  which generally do need several passages at once.
- **`ask_sequential()` -- sequential.** For a `lookup` question, try the
  single best passage first. `generation.answer` now returns, alongside the
  answer text, two booleans: `answer_found` (the passage says anything
  relevant) and `complete_answer_found` (it fully answers the question, no
  more passages needed). If both are true, that's the final answer -- one
  retrieval, one generation call, for the majority of direct-lookup
  questions where the fact lives in a single passage. If not, it escalates
  exactly once, to the full `top_k`, and re-generates. Comparison/list
  questions skip straight to batch, since a single passage is essentially
  never a complete comparison. The loop is bounded by construction -- at
  most two generation calls, never a retry counter that could spin.

`QueryResult.generation_calls` reports which happened, and the Streamlit UI
shows it next to every answer, so switching the sidebar toggle between
`batch` and `sequential` on the same question makes the real cost tradeoff
visible -- and it's a genuine tradeoff, not a strict win, which is worth
being honest about. Measured on the sample PDF:

- *"What is the claims reference?"* -- a clean single-passage lookup.
  `parse_query` itself already guesses a small `top_k` (1-2) for a lookup
  question, so both strategies land on 1 hit, 1 generation call. No
  difference here -- sequential's advantage only exists when `top_k` was
  guessed *larger* than the answer actually needed.
- *"What is the policy number and what is the claims reference to quote
  when filing a claim?"* -- a two-part lookup, whose two facts live on
  different pages. `ask()` (batch) parses this as needing 2 passages up
  front and answers correctly in **1** call. `ask_sequential()` tries the
  single best passage first, finds it incomplete (`complete_answer_found`
  false, correctly), and escalates -- landing on the same correct answer in
  **2** calls. Sequential cost *more* here, not less: the top-1 attempt was
  a wasted call.

So the honest version of this tradeoff: sequential only wins when
`parse_query`'s `top_k` guess was more generous than the question actually
needed and the answer turns out to fit in the first passage anyway (it then
sends fewer prompt tokens for the same 1 call); it loses outright -- one
extra full round trip -- whenever the top-1 guess turns out wrong and
escalation fires. Whether that trade is worth it in general depends on how
often `top_k` overshoots vs. undershoots in practice, which this one sample
document isn't large or varied enough to answer honestly either way.

If the model's JSON response for a generation call can't be parsed,
`answer()` falls back to the fixed not-found refusal rather than guessing at
partial JSON -- an unparseable response isn't evidence the passages
supported an answer, and treating it as a "found" answer risks shipping
something ungrounded.

## Not-found discipline

The generation prompt allows exactly one way to say "I don't know": the
fixed sentence `NOT FOUND IN DOCUMENT.`, checked verbatim in code
(`generation.NOT_FOUND`), not pattern-matched against free text. Every other
answer must cite `[n]` back to a retrieved passage. This is deliberate: a
wrong-but-confident answer sourced from the wrong page is worse than a
refusal, because the refusal is visibly a refusal and the wrong answer isn't.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python scripts/make_sample_pdf.py   # optional: generates a fictitious test PDF
streamlit run app.py
```

Upload a PDF (or the generated `sample_docs/sample_policy.pdf`), ask a
question that's answerable and one that isn't, confirm the citation and the
refusal both fire. Switch the sidebar retrieval method between hybrid /
bm25 / dense to see the retrieved passages -- and the answer -- change.

## Tests

```bash
pytest tests/ -v
```

20 tests: BM25 correctness, cosine correctness, the RRF fusion regression
above, `HybridRetriever` end to end with and without an embedding function,
and PDF chunking (page numbers preserved, chunks never cross a page
boundary, long pages get split). All retrieval tests inject a deterministic
fake embedding function (normalized token-count-bucket vectors) so the
suite needs no network access or API key.

## Deploy

Deployed via **Streamlit Community Cloud** (share.streamlit.io), not
Hugging Face Spaces: HF now requires a paid PRO plan to create a Docker or
Gradio Space (confirmed from HF's own docs -- Static is the only free SDK
left there, and this app needs a Python backend, so Static doesn't work for
it). Streamlit Community Cloud is free, needs no Dockerfile or SDK
selection, and runs `app.py` directly off `requirements.txt` -- including
from a subdirectory of a larger repo, which is how this one is actually
hosted: **github.com/Pasha0313/MachineLearning/tree/master/foyer/foyer-doc-qa**
(this project living alongside other, unrelated repos rather than in a
dedicated one of its own).

At share.streamlit.io: **New app** -> repository `Pasha0313/MachineLearning`,
branch `master`, main file path `foyer/foyer-doc-qa/app.py` (Streamlit Cloud
supports a subdirectory path directly) -> in **Advanced settings**, add a
secret:
```
OPENAI_API_KEY = "sk-..."
```
(pasted directly into Streamlit's secrets UI, never committed). Deploy.

**Before making it public: set a hard spend cap on the OpenAI account this
key belongs to** (OpenAI dashboard -> Settings -> Billing/Project -> Limits
-> Edit spend limit -> enable "Enforce a hard limit"). A public app calling
a paid API on your key has no rate limiting or auth in front of it (see
Scope, below) -- anyone with the link can run up usage. A spend cap is the
actual safety net, not a suggestion; without "Enforce a hard limit" turned
on, OpenAI's spend figure is only an alert, not a stop.

**Alternative: Hugging Face Spaces (Docker), if you ever have HF PRO.**
[Dockerfile](Dockerfile) and `.dockerignore` are already in this repo,
built and run locally with `docker build` / `docker run` against the real
API before being committed -- not just written and assumed to work. If you
upgrade later: create the Space with SDK **Docker**, add this YAML
front-matter back to the top of this file --
```yaml
---
title: Foyer Doc-QA
emoji: 📄
sdk: docker
app_port: 7860
pinned: false
---
```
-- then `git remote add space <space-url>` and `git push space master:main`,
same secret-setting step as above but under the Space's
Settings -> Repository secrets instead. Note this repo now lives in a
subfolder of a larger one (see above), so a straight `git push` to an HF
Space -- which expects the pushed repo root to *be* the app -- would need
the same sparse-checkout-and-copy approach used to get this onto GitHub in
the first place, not a plain push from this folder.

## What this decides

A claims handler asks the policy "is water damage from a burst pipe
covered, and what's the deductible?" -- the app answers with `[n]`
citations back to the exact page and clause, so the handler can verify the
answer against the policy wording in seconds instead of reading the whole
document. That's the decision this tool is actually for: turning "search the
PDF" into "get the answer with a page to check it against."

The fixed `NOT FOUND IN DOCUMENT.` refusal matters for the same reason in
reverse: if the tool answers a coverage question it can't actually support
from the text, that's the expensive failure mode in insurance -- a
handler acting on an invented coverage detail. Silence plus a page reference
to check manually is a better outcome than a fluent, wrong answer.

## Scope -- what's left out, and what breaks first at volume

**Left out, deliberately, for a technical assessment:**
- Single PDF per session, held in memory. No persistent index, no
  multi-document corpus, no incremental re-indexing on file change.
- No auth, no rate limiting, no request queueing -- one user, one browser
  tab, synchronous calls.
- No OCR: scanned/image-only PDFs extract no text and the index build fails
  loudly (`ValueError`) rather than silently returning an empty answer.
- No conversation memory -- every question is independent; there's no
  follow-up/coreference handling ("what about *that* clause").
- Chunking is paragraph-based with a fixed character budget, not
  layout-aware -- it doesn't special-case tables, multi-column layouts, or
  figures, which is exactly the kind of "clean structure assumed" gap
  real inputs (scans, emails, mixed files) will violate more than a
  generated PDF does.

**What breaks first if volume increased:**
- The embedding step is one OpenAI call per chunk batch made synchronously
  at upload time -- indexing a large PDF (hundreds of pages) would be slow
  and there's no caching, so re-uploading the same file re-embeds it from
  scratch. That's the first wall: no persisted vector store, no
  incremental indexing, everything rebuilt per session.
- `HybridRetriever` keeps every chunk's text and embedding vector in a
  Python list in memory and does a linear scan per query (`O(n)` cosine and
  BM25 over all chunks). Fine for one document; would need a real ANN index
  (or at least numpy-vectorized scoring) well before document count or
  corpus size got large.
- Everything is synchronous and single-process -- concurrent users would
  serialize behind the same Streamlit session state; there's no job queue
  or async worker for the embed/generate calls.
