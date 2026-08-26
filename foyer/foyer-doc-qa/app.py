"""Streamlit front end for the document chatbot.

Deliberately thin: this app exists to make the retrieval + generation
pipeline (parsing.py, retrieval.py, embedding.py, generation.py) usable and
inspectable from a browser, not to showcase front-end engineering. Every
decision the pipeline makes -- which passages were retrieved, from which
scorer, from which page -- is shown, not hidden, so a reviewer can see why
an answer says what it says.
"""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pipeline import DocumentIndex  # noqa: E402

st.set_page_config(page_title="Foyer Doc-QA", page_icon="📄", layout="wide")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to a .env file (see .env.example) and restart.")
    st.stop()

st.title("Document Q&A -- hybrid retrieval, cited answers")
st.caption(
    "Hand-written BM25 + dense cosine + reciprocal rank fusion (retrieval.py). "
    "No LangChain, no LlamaIndex, no vector DB."
)
st.caption(
    "No app-level rate limiting or auth. If deployed publicly, set a spend "
    "cap on the API key in use -- see README Deploy."
)

with st.sidebar:
    st.header("Settings")
    method = st.radio(
        "Retrieval method",
        options=["hybrid", "weighted", "bm25", "dense"],
        index=0,
        help="hybrid = BM25 + dense fused by rank (RRF); weighted = the same two scores fused "
             "by normalised magnitude instead of rank -- more correct when one scorer is far "
             "more confident than the other, at the cost of an alpha to tune (see README, "
             "'Why hybrid retrieval'). Switch to bm25 or dense to see either scorer alone.",
    )
    strategy = st.radio(
        "Generation strategy",
        options=["batch", "sequential"],
        index=0,
        help="batch = one retrieval at the parsed top_k, one generation call. "
             "sequential = try the single best passage first for a lookup question, and only "
             "spend a second retrieval + generation call escalating to top_k if that passage "
             "wasn't a complete answer. Comparison/list questions always go straight to batch. "
             "Watch the generation-call count below change between the two.",
    )
    st.markdown("---")
    st.markdown(
        "**Scope**: single PDF, in-memory index, synchronous per-query embedding + "
        "generation calls. See README for what this does not do."
    )

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("_file_hash") != file_hash:
        with st.spinner("Parsing and indexing (chunking + embedding every passage)..."):
            try:
                st.session_state["_index"] = DocumentIndex(pdf_bytes)
                st.session_state["_file_hash"] = file_hash
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.success(f"Indexed {len(st.session_state['_index'].chunks)} passages.")

    index: DocumentIndex = st.session_state["_index"]

    query = st.text_input("Ask a question about the document")
    ask = st.button("Ask", type="primary", disabled=not query)

    if ask and query:
        with st.spinner("Parsing query, retrieving, generating..."):
            if strategy == "sequential":
                result = index.ask_sequential(query, method=method)
            else:
                result = index.ask(query, method=method)

        st.subheader("Answer")
        if result.answer.not_found:
            st.warning(result.answer.text)
        else:
            st.markdown(result.answer.text)
            if result.answer.pages:
                st.caption(f"Cited pages: {', '.join(str(p) for p in result.answer.pages)}")
        st.caption(
            f"Strategy: {strategy} -- {result.generation_calls} generation call"
            f"{'s' if result.generation_calls != 1 else ''}, {len(result.hits)} passage"
            f"{'s' if len(result.hits) != 1 else ''} in the final prompt."
        )

        with st.expander(f"Retrieved passages ({method}, query parsed as intent={result.parsed_intent!r})"):
            st.caption("These are the exact passages sent to generation -- not a re-run.")
            for n, hit in enumerate(result.hits, start=1):
                page = index.chunks[hit.index].page
                st.markdown(
                    f"**[{n}] page {page}** -- bm25={hit.bm25:.3f}  dense={hit.dense:.3f}  "
                    f"{method}={hit.hybrid:.3f}"
                )
                st.text(hit.text[:500] + ("..." if len(hit.text) > 500 else ""))
else:
    st.info("Upload a PDF to start. No file is stored anywhere -- it's parsed and indexed in memory for this session only.")
