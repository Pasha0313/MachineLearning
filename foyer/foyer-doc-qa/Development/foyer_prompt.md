# Foyer Doc-QA — full prompt history (build + finish + submit)

Phases 1 and 2 are complete and were both executed by the coding agent. This
file merges what were previously two separate files (`foyer_build_prompt.md`,
`foyer_finish_prompt.md`) into one, kept as a record of exactly what was
asked for. Phase 3, added 2026-08-26, is different in kind: it's the
remaining steps that only the human (Saeed) can do — credentials, money, and
sending an actual email — checked off one at a time, not run by an agent.
For live status see [progress.md](progress.md).

---

## Phase 1 — Build

Original framing: a coding agent picks this up assuming the four bricks
(`app.py`, `parsing.py`, `retrieval.py`, `embedding.py`, `generation.py`,
`pipeline.py`, `README.md`, `requirements.txt`) already exist and are
correct, and completes everything around them.

You are completing a document-QA project for a technical assessment. The
reviewer is the author of two of the three reference articles, so accuracy
matters more than polish: every claim the repo makes about itself must be
literally true and independently checkable by running the code.

### Ground rules (do not violate)

1. **Do not invent capabilities.** The repo must not claim any feature it does
   not have. If a task can't be completed honestly, stop and say so instead of
   faking it.
2. **Never tune a demo or test to match a number written elsewhere.** Where a
   README figure and actual program output disagree, change the README to match
   the output — never the reverse.
3. **Match the existing code exactly.** Import real symbols with their real
   signatures. Do not guess names. If unsure, open the file.
4. **No new heavy frameworks.** No LangChain, no LlamaIndex, no vector DB. That
   is the whole point of the exercise. `sentence-transformers` is allowed *only*
   as an optional demo/dev dependency, never in the core pipeline.
5. When you finish, run the acceptance checklist at the end and report the
   actual output of each command.

### Task 1 — Sample PDF generator (`scripts/make_sample_pdf.py`)

Writes a fictitious home-insurance policy to `sample_docs/sample_policy.pdf`
using `reportlab`. ~4 pages, fictitious, deliberately built so the app is
demonstrable:

- A rare identifier appearing exactly once: policy number
  `FOY-HOME-2026-004417`, plus a claims reference `PRC-CLAIM-24`.
- Two comparable figures: buildings sum insured `EUR 480,000` and contents
  sum insured `EUR 65,000`.
- A numbered exclusions list of 8 items (`Section 5`) for list/top-k queries.
- Absent topics so "not found" can fire: `cyber`, `identity`, `refund` must
  appear **zero** times — asserted in code (read the PDF back with PyMuPDF,
  raise if any count is non-zero).
- A header on page 1 stating it's a fictitious sample, not a real contract.

### Task 2 — Keyword-beats-cosine demo (`scripts/demo_keyword_beats_cosine.py`)

Reproduce, on a small fixed corpus, the effect from the retrieval article: a
passage holding an exact rare token is ranked poorly by dense cosine alone
but found by BM25 and kept by the hybrid fusion. Score with the real
functions in `retrieval.py`, not a reimplementation.

Embedding choice must be real: `sentence-transformers` (`all-MiniLM-L6-v2`)
if importable, else the project's own `embedding.make_embed_fn()` if
`OPENAI_API_KEY` is set, else exit 1 — never a fake/bag-of-words stand-in.
Print the ranks, then update the README's "Why hybrid retrieval" section to
state the actual measured ranks, whatever they are. Add
`sentence-transformers` to a new `requirements-dev.txt`, not the core one.

### Task 3 — Test suite (`tests/`)

Deterministic fake `embed_fn` for retrieval tests (no network/API key
needed). PyMuPDF-synthesized in-memory PDFs for parsing tests, not
reportlab. Named tests specified for `test_retrieval.py` (BM25 ranking/IDF/
rare-vs-common-term, cosine identity/orthogonal, RRF only-credits-matched/
ties-use-competition-ranking/the all-zero-dense regression by exact name,
hybrid retriever with/without embed_fn, the `bm25_query` override) and
`test_parsing.py` (page numbers preserved and 1-based, no chunk crosses a
page boundary, long page splits). If the actual test count differs from
whatever the README claims, fix the README to match reality, not the tests.

### Task 4 — `.env.example` and secret hygiene

`.env.example` with `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`,
`OPENAI_EMBED_MODEL`. Confirm `.gitignore` ignores `.env`. The real key must
never be committed.

### Task 5 — Deployment prep (Hugging Face Spaces, Streamlit SDK)

HF Spaces YAML front-matter at the top of `README.md`. Confirm
`requirements.txt` is sufficient for the Space. Add a `## Deploy` section:
push to a new HF Space, set `OPENAI_API_KEY` as a Space secret (never in
code). A UI caption plus a README warning that a public app on a paid API
key needs a hard spend cap set in the OpenAI dashboard. Do not deploy from
here — prepare only, print the exact steps for the human to run.

### Task 6 — Business-decision framing

Kezhan asked for this twice: "think first of the business," "show what your
result can decide." A short `## What this decides` README section, in an
insurer's terms — factual to what the code does, no overclaiming.

### Task 7 — OPTIONAL, only if everything above is done and time remains

The sequential top-1-vs-top-k loop from the loop-engineering article:
`generation.answer` also returns `answer_found` / `complete_answer_found`;
a `pipeline.py` method that tries top-1 first for a lookup question and
escalates to full `top_k` only if incomplete (bounded — always terminates);
a sidebar toggle in `app.py` to compare. If implemented, describe it
accurately in the README; if not, leave the existing "one-shot version of
top-1-vs-top-k" wording as-is rather than overclaiming.

### Acceptance checklist (Phase 1)

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/make_sample_pdf.py
python scripts/demo_keyword_beats_cosine.py
pytest tests/ -v
streamlit run app.py
```

Then, against the sample PDF: lookup (policy number, cites page 1), list
(exclusions), comparison (480,000 vs 65,000), not-found (cyber/identity —
fixed refusal, not an invented answer). Any mismatch between actual output
and the README gets fixed in the README, never in the output.

---

## Phase 2 — Finish and ship

Original framing: the build is done; this phase makes it self-consistent,
verifies it end-to-end, and prepares it for a public link — reconcile, not
build.

### Ground rules (do not violate)

1. Never tune a demo or test to match a number written elsewhere — fix the
   README to the real output.
2. No new capabilities, no new frameworks, no new libraries. Match existing
   function signatures exactly.
3. **Do not remove the honest limitation notes already in the README** — the
   RRF-vs-magnitude finding, the `exclude`→`lookup` intent-misclassification
   note, and the sequential-vs-batch tradeoff are the point, not defects to
   sand off.
4. Credentials and money are off-limits: do not enter, echo, print, or commit
   the API key; do not create accounts; do not `git push`; do not
   `git remote add`; do not touch OpenAI billing/spend caps. Where a step
   needs any of those, stop and print the exact commands for the human to run
   themselves.
5. Do not delete `.venv`, `.venv-1`, `.env`, or `Key.txt`. *(Superseded
   2026-08-26: the human explicitly asked for the two venvs to be merged —
   `.venv-1`, the superset with `reportlab`/`sentence-transformers`/`torch`,
   was kept and renamed to `.venv`; the old `.venv` was deleted. This rule
   applied to that specific automated pass, not permanently.)*
6. Run the acceptance checklist and report actual output.

### Task 1 — Single source of truth for the sample PDF

Two `sample_policy.pdf` files existed (project root, and `sample_docs/`) and
differed. The canonical one is whatever `scripts/make_sample_pdf.py`
produces, because that's what a reviewer can regenerate and check. Delete
the root copy if it differs from a freshly generated one; never delete
`sample_docs/sample_policy.pdf`.

### Task 2 — Reconcile every runnable README claim to reality

Run `pytest tests/ -v`, the demo script (real embedding backend required —
`requirements-dev.txt` installed, never a fake fallback), and the four
acceptance questions against the regenerated PDF (only if `OPENAI_API_KEY`
is actually set — never add or print one). Fix the README wherever a stated
number disagrees with the real output.

### Task 3 — Pre-commit hygiene and a local commit (no remote, no push)

Confirm `.gitignore` covers `.env`, `Key.txt`, `.venv/`, `.venv-1/`,
`__pycache__/`; decide and state how `sample_docs/*.pdf` is handled. `git
init` if needed, `git add -A`, scan the staged diff for a leaked key
(`grep -nE "sk-[A-Za-z0-9_-]{20,}"`) before committing — stop if it matches
anything. No remote, no push; print the exact steps for the human instead.

### Task 4 — Deploy-readiness checks (prepare only)

Confirm the HF Spaces YAML front-matter is intact and `app_file: app.py`.
Check whether the pinned `sdk_version` is actually a supported HF Spaces
version; if it can't be confirmed, remove the line rather than guess.
Confirm `requirements.txt` is sufficient. Confirm the UI caption makes no
unverified claim (no asserting a rate limit or spend cap that isn't actually
enforced).

### Task 5 — Final report

A table: each self-claim (test count, the four demo ranks, "not-found
fires", "identifiers reach BM25", "one retrieval per question") against the
verified result, marked match/fixed. Then everything still requiring a
human, explicitly: set the OpenAI spend cap, push to the HF Space and add
the key as a Space secret, open the live URL and run the four question
types by hand, then send the reply to Kezhan (gated on the live URL
actually existing).

### Acceptance checklist (Phase 2)

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/make_sample_pdf.py
python scripts/demo_keyword_beats_cosine.py
pytest tests/ -v
streamlit run app.py
```

Then, only if `OPENAI_API_KEY` is set: the same four question types as
Phase 1, against the regenerated PDF. Any mismatch gets fixed in the
README, never in the output.

---

## Phase 3 — Submit (human only, one at a time)

Everything in Phases 1 and 2 is done: code, tests, demo, README all
reconciled and verified, repo committed locally. Nothing has been pushed
anywhere and no live URL exists yet — this phase is what actually gets a
link into Kezhan's hands. Go through it in order; each step is gated on the
one before it.

- [x] **1. Set a hard spend cap on the OpenAI key.** *(Done 2026-08-26: $3.00
  monthly project spend limit, "Enforce a hard limit" on — confirmed by the
  UI showing "Requests will start to fail when limit is reached," not the
  old soft-budget "may exceed this" wording.)*

- [x] **2a. Push to GitHub.** *(Done 2026-08-26.)*
  *(Revised twice before this: first found Streamlit is no longer a
  selectable Spaces SDK on HF — it moved to Docker. Then found Docker (and
  Gradio) Spaces now require a paid HF PRO plan for personal accounts,
  confirmed from HF's own docs, not assumed. Static Spaces are the only
  free HF SDK left, and this app needs a Python backend, so that's not
  viable either. Pivoted to **Streamlit Community Cloud** instead — free,
  no Docker needed, one of the platforms Kezhan named himself. The
  Dockerfile stays in the repo, tested and working, as a documented
  fallback if HF PRO is ever worth it later — see README's Deploy section.)*

  Not a fresh standalone repo — the human pointed at an existing repo with a
  placeholder subfolder instead:
  **https://github.com/Pasha0313/MachineLearning/tree/master/foyer/foyer-doc-qa**
  (public). Pushed by: sparse-cloning just the `foyer/` path of
  `MachineLearning` into a scratch dir, replacing the placeholder file
  there with every file tracked in this local repo (`git ls-files`, 19
  files — no `.venv`, no `.env`, no `Key.txt`, no PDFs), committing, and
  pushing a normal additive commit (`a6dcc3f..0f3745d`) that touches only
  `foyer/foyer-doc-qa/` — the rest of `MachineLearning` is untouched.
  Verified server-side after push with `git ls-remote` (not just trusting
  the push output), since a first GitHub web-fetch to eyeball the result
  returned a stale 15-minute cache still showing the old placeholder.

- [ ] **2b. Deploy on Streamlit Community Cloud.**
  At **share.streamlit.io**: sign in with GitHub, **New app**, repository
  `Pasha0313/MachineLearning`, branch `master`, **main file path**
  `foyer/foyer-doc-qa/app.py` (Streamlit Cloud supports a subdirectory path
  like this directly — no need for a dedicated repo). In **Advanced
  settings**, add a secret:
  ```
  OPENAI_API_KEY = "sk-..."
  ```
  (typed directly into Streamlit's secrets UI, never committed). Deploy —
  first build takes a few minutes. *(Done 2026-08-26. Live URL:
  **https://machinelearning-crfe8zkcb2arxd9xludczc.streamlit.app/** —
  "Indexed 4 passages" confirmed on the sample PDF. Sharing setting
  confirmed as "This app is public and searchable," and confirmed
  genuinely public by opening it in a fresh/incognito Chrome profile with
  no Streamlit login — not just by reading the settings panel. Note: an
  automated `curl`/fetch check of this URL redirects through
  `share.streamlit.io/-/auth/app` to a `/-/login` page and looks like an
  auth wall — that's a false alarm from tooling that doesn't run
  JavaScript or hold cookies, not a real access restriction. Don't
  re-alarm on that same signal later; the incognito-browser test is the
  one that actually settles it.)*

- [x] **3. Open the live URL yourself and run all four question types by hand.**
  *(Done 2026-08-26, on the actual deployed app, by the human — not a
  pipeline call against local code, which was the whole point of this step.)*
  1. Lookup — "What is the policy number?" → `FOY-HOME-2026-004417 [1]`, page 1. ✓
  2. List — "What does the policy exclude?" → 8 exclusions listed, page 3. ✓
  3. Comparison — "Is the buildings sum insured higher than the contents sum insured?" → 480,000 vs 65,000, correct. ✓
  4. Not found — "Does this policy cover cyber or identity theft?" → `NOT FOUND IN DOCUMENT.` ✓

  All four confirmed matching the README's claimed behavior.

- [ ] **4. Send the reply to Kezhan.**
  Gated on step 3 actually passing. Needs: the live link; the two sentences
  he asked for (what was left out, what breaks first at volume — both
  already written in the README's Scope section, just need trimming into
  email form); and an answer to his open question about messy/scanned
  inputs, which the README's Scope section also already speaks to
  (no OCR, no layout-aware chunking).
