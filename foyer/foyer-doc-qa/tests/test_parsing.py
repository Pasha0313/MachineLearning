"""Tests for parsing.py. PDFs are synthesized in-memory with PyMuPDF
(pymupdf.open() + page.insert_textbox()) -- no reportlab dependency here, and
no file ever touches disk.
"""

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parsing import _split_paragraphs, parse_pdf_bytes


def _build_pdf(pages_text):
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(56, 56, 539, 786), text, fontsize=11, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# named tests
# ---------------------------------------------------------------------------

def test_chunk_page_numbers_are_preserved_and_one_based():
    pages = [
        "Page one content.\n\nSecond paragraph on page one.",
        "Page two content only.",
    ]
    chunks = parse_pdf_bytes(_build_pdf(pages), max_chars=1000, overlap_chars=0)
    assert chunks
    assert min(c.page for c in chunks) == 1  # 1-based, not 0-based
    assert any(c.page == 1 for c in chunks)
    assert any(c.page == 2 for c in chunks)


def test_no_chunk_crosses_a_page_boundary():
    # Both pages are short enough that, if chunking ever merged text across
    # pages, they'd land in the same chunk. They must not.
    pages = ["End of page one.", "Start of page two."]
    chunks = parse_pdf_bytes(_build_pdf(pages), max_chars=1000, overlap_chars=0)
    assert len(chunks) == 2
    assert chunks[0].page == 1 and "page one" in chunks[0].text
    assert chunks[1].page == 2 and "page two" in chunks[1].text
    assert "page two" not in chunks[0].text
    assert "page one" not in chunks[1].text

    # also holds for the general, multi-page case: page numbers across all
    # chunks are monotonically non-decreasing (no chunk ever "goes back" to
    # an earlier page, which single-page-at-a-time chunking guarantees).
    many_pages = ["Short page one.", "Short page two.", "Short page three."]
    many_chunks = parse_pdf_bytes(_build_pdf(many_pages), max_chars=1000, overlap_chars=0)
    pages_seen = [c.page for c in many_chunks]
    assert pages_seen == sorted(pages_seen)


def test_long_page_is_split_into_multiple_chunks():
    # 60 repeats (~1.7k chars) is well over max_chars=300 but still small
    # enough that insert_textbox renders the whole thing onto one A4 page
    # instead of silently dropping the overflow.
    long_text = "This is one long paragraph. " * 60
    chunks = parse_pdf_bytes(_build_pdf([long_text]), max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    assert all(c.page == 1 for c in chunks)


# ---------------------------------------------------------------------------
# extra coverage beyond the reviewer's prompt-specified set
# ---------------------------------------------------------------------------

def test_chunk_ids_are_sequential():
    pages = ["Alpha content here.", "Beta content here."]
    chunks = parse_pdf_bytes(_build_pdf(pages), max_chars=1000, overlap_chars=0)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_split_paragraphs_falls_back_to_single_newline_when_no_blank_lines():
    # No "\n\n" anywhere -- the real-PDF case that used to collapse into one
    # giant paragraph and get blindly hard-split by character count.
    text = "Policyholder: Marie Dupont\nPolicy number: FOY-2026-1\nVehicle: Golf"
    assert _split_paragraphs(text) == [
        "Policyholder: Marie Dupont",
        "Policy number: FOY-2026-1",
        "Vehicle: Golf",
    ]


def test_split_paragraphs_prefers_blank_line_boundaries_when_present():
    # Blank-line-delimited paragraphs are used as-is, internal single
    # newlines and all -- the fallback must not fire just because a
    # paragraph happens to wrap onto more than one line.
    text = "Title line\n\nBody line one\nBody line two"
    assert _split_paragraphs(text) == ["Title line", "Body line one\nBody line two"]
