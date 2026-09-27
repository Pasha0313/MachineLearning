# Progress — Foyer Doc-QA

Snapshot of what actually exists and has been run, as of 2026-08-25. This is a
status record, not a plan.

## Post-review fixes (after an external review of the README)

A review correctly flagged that the "Why hybrid retrieval" section showed
the app's own default (`hybrid`/RRF, rank 4/6) losing to plain BM25
(rank 1/6) with no resolution — a bad look leading the README. It also
claimed, unverified, that a "realistic" topically-varied corpus would show
cosine burying the answer and hybrid winning at rank 1/6. That claim was
**not actually run** (the reviewer had no way to execute real embeddings)
and turned out to be wrong: tested here twice (6 and 12 topically-varied
passages, real `all-MiniLM-L6-v2`), and cosine did **not** bury the correct
passage either time — it ranked it 1st. That's now documented in the demo
script and the README instead of silently discarded.

The real fix: `retrieval.weighted_sum` (score-magnitude fusion, already
implemented and tested, previously undemonstrated) recovers rank 1/6 on the
exact same near-duplicate corpus where RRF only reaches rank 4/6. Added
`weighted` as a fourth method to the demo script and to the app's sidebar,
and rewrote the README section to show the full picture: RRF's real,
honest limitation (rank-only fusion discards BM25's decisive score margin),
`weighted` as the tested resolution, and why RRF still ships as the default
anyway (scale-free, no `alpha` to tune) rather than pretending one method
is strictly better.

Two smaller real fixes from the same review: the app's UI caption claimed
"rate- and spend-capped" while the README Scope says "no rate limiting" —
contradiction, now removed (caption no longer asserts an unverified spend
cap either, just flags the absence of rate limiting/auth and points at
Deploy). The review's claim that `.gitignore`/`.env.example` were misnamed
(`_gitignore`, `_env.example`) was checked and is false — both are correctly
named on disk; likely a markdown-rendering artifact on the reviewer's side,
not a real bug.

## Status: all 7 tasks in `foyer_prompt.md` (Phase 1) implemented, including the
## optional one (Task 7).

| Task | Status | Notes |
|---|---|---|
| 1. Sample PDF generator (reportlab) | Done | `scripts/make_sample_pdf.py` → `sample_docs/sample_policy.pdf`, 4 pages, home insurance. `FOY-HOME-2026-004417` and `PRC-CLAIM-24` each appear once; cyber/identity/refund asserted at zero occurrences (script raises if not). |
| 2. Keyword-beats-cosine demo, real embeddings | Done | `scripts/demo_keyword_beats_cosine.py`. sentence-transformers (`all-MiniLM-L6-v2`) by default, OpenAI embeddings if that's not installed, hard exit if neither — no fake embedder. |
| 3. Test suite, named + counted | Done | 20 tests (`pytest tests/ -v`), not 13 — see below. |
| 4. `.env.example` / secret hygiene | Done | Matches the spec's exact template; `.env` and `Key.txt` gitignored. |
| 5. Deploy prep (HF Spaces) | Done, not deployed | YAML front-matter at the top of `README.md`, `## Deploy` section with git/HF steps, spend-cap warning, and a caption in `app.py`. Nothing was pushed anywhere. |
| 6. "What this decides" business framing | Done | New README section, insurer/claims-handler framing, factual to what the code does. |
| 7. Top-1-vs-top-k loop (optional) | Done | `generation.answer` now returns `answer_found`/`complete_answer_found`; `pipeline.DocumentIndex.ask_sequential()` tries top-1 first for lookup questions and escalates once (bounded, never more than 2 calls) if incomplete; sidebar toggle in `app.py` (`batch` vs `sequential`) shows `generation_calls` live. |

## What's been run, not just written

- `pytest tests/ -v` → **20/20 pass**. 11 are the exact names the brief specified for `test_retrieval.py`, 3 are the parsing ones it specified (page numbers, no cross-page chunk, long-page split — the brief allowed a 14th test here or folding it in; it's separate), the remaining 6 are extra coverage (`tokenize`, `weighted_sum`, BM25 zero-overlap, and the two paragraph-split-fallback tests from an earlier review pass) that the brief's list didn't ask for but that test real, otherwise-uncovered code paths. README updated from "13 tests" to the real number, 20 — per the brief's own rule ("never tune a test/demo to match a number written elsewhere; change the README to match the output").
- `scripts/demo_keyword_beats_cosine.py`, real embeddings, actually run:
  - **First attempt** (topically-similar distractors, one correct passage) did **not** reproduce the burial effect with `all-MiniLM-L6-v2` — cosine already ranked the correct passage 1st. Discarded rather than reported, per the "don't fake it" rule.
  - **Final version** (six near-duplicate "twin" passages differing only by policy number) does reproduce it cleanly: `cosine: rank 5/6, bm25: rank 1/6, hybrid: rank 4/6`. Also verified the OpenAI-embedding fallback path independently (`rank 4/6` on that backend) by simulating sentence-transformers being absent.
  - **Hybrid did not fully recover rank 1 here (landed at rank 4)** — a real, reported finding, not a bug: RRF fuses by rank, not score magnitude, so BM25's huge score margin (3.45 vs 0.37, ~9x) only bought the correct passage a one-rank-position edge, while dense's much smaller, noisier margins got equal per-rank weight. Documented in README's "Why hybrid retrieval" section along with the fix, since it's a real limitation distinct from the RRF all-zero-dense bug fixed earlier.
- Live pipeline run against the real OpenAI API on the new home-insurance sample PDF, all four required question types from the brief's acceptance checklist:
  1. Lookup ("What is the policy number?") → correct, cited page 1.
  2. List ("What does the policy exclude?") → correctly listed all 8 exclusion items. **Side finding**: `parse_query` classified this as intent `lookup`, not `list` — worked out fine here only because the whole exclusions section sits in one page/chunk; flagged honestly in the README rather than silently accepted or hidden.
  3. Comparison (buildings vs. contents sum insured) → correctly compared 480,000 vs 65,000, intent correctly classified `comparison`.
  4. Not found (cyber/identity theft) → correctly refused with the fixed `NOT FOUND IN DOCUMENT.` string, under the new JSON-mode generation call.
  - Also verified Task 7's escalation actually fires: a two-part lookup question (policy number + claims reference, on different pages) made `ask_sequential` try top-1, correctly detect it as incomplete, and escalate to a second call — 2 generation calls vs. batch's 1. This is the opposite of what the README originally (wrongly) assumed ("sequential costs the same or less") — corrected the README to report the real, mixed tradeoff instead.
- `streamlit run app.py` boots and serves after all changes (checked via HTTP against a background instance, then stopped).

## Known items still open

- **Untracked file `sample_policy.pdf` at the project root** (4 pages, home-insurance, same policy number as the new canonical one) is still there, unexplained, and now redundant with `sample_docs/sample_policy.pdf`. Not deleted — origin was never confirmed, left for you to remove if you agree it's superseded.
- Nothing has been deployed. `git init` hasn't been run in this project at all — there is no git repo yet, so nothing has been committed or pushed anywhere.
- `requirements.txt` still lists `numpy`, which nothing in the codebase actually imports (pure-Python math throughout `retrieval.py`). Harmless, left in since the brief's Task 5 explicitly listed it as part of a sufficient requirements set, but worth knowing it's dead weight if you ever trim the Space's install.

## Secrets

The OpenAI key lives in `.env` (gitignored) and originally in `Key.txt` (also gitignored) at the project root. Neither has ever been committed — there is no git repo initialized in this project yet.
