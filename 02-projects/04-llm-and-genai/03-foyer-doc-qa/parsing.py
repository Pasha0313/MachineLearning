"""Parsing.

PDF -> page-numbered chunks. Uses PyMuPDF directly: no document-loader
framework in between, so the slicing decisions (where a chunk starts and
ends, how much overlap it carries, what counts as a paragraph break) are
explicit and inspectable here rather than hidden behind a library default.

Chunking strategy: split each page into paragraphs on blank lines, falling
back to single-newline splitting when a page has no blank lines at all (real
PDFs often don't), then pack paragraphs into chunks up to `max_chars`,
splitting any single paragraph that is longer than that on its own. A chunk
never spans two pages, because the citation we show the user is a page
number -- a chunk that crossed a page boundary would make that citation
ambiguous.
"""

from dataclasses import dataclass
from typing import List

import pymupdf


@dataclass
class Chunk:
    chunk_id: int
    page: int  # 1-based
    text: str


def _split_paragraphs(page_text: str) -> List[str]:
    """Split on blank lines first; if that collapses the whole page into a
    single block, fall back to splitting on single newlines instead, since
    PyMuPDF frequently emits one "\\n" per line/paragraph with no blank line
    between them at all."""
    paras = [p.strip() for p in page_text.split("\n\n")]
    paras = [p for p in paras if p]
    if len(paras) > 1:
        return paras
    lines = [p.strip() for p in page_text.split("\n")]
    return [p for p in lines if p]


def _pack(paragraphs: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """Greedily pack paragraphs into chunks <= max_chars. A paragraph longer
    than max_chars on its own is hard-split. Consecutive chunks carry a small
    trailing/leading overlap so a sentence split across a chunk boundary is
    still fully present in at least one chunk."""
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            # hard split an oversized paragraph
            start = 0
            while start < len(para):
                piece = para[start:start + max_chars]
                chunks.append(piece)
                start += max_chars
            current = ""
    if current:
        chunks.append(current)

    if overlap_chars and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(prev_tail + "\n\n" + chunks[i])
        return overlapped
    return chunks


def _chunk_doc(doc: "pymupdf.Document", max_chars: int, overlap_chars: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            continue
        for piece in _pack(paragraphs, max_chars, overlap_chars):
            chunks.append(Chunk(chunk_id=chunk_id, page=page_index + 1, text=piece.strip()))
            chunk_id += 1
    return chunks


def parse_pdf(path: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[Chunk]:
    """Extract text page by page and chunk it, preserving the page number
    each chunk came from so generation can cite it."""
    doc = pymupdf.open(path)
    try:
        return _chunk_doc(doc, max_chars, overlap_chars)
    finally:
        doc.close()


def parse_pdf_bytes(data: bytes, max_chars: int = 1200, overlap_chars: int = 150) -> List[Chunk]:
    """Same as parse_pdf but from in-memory bytes (Streamlit file uploads
    don't hand you a path)."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        return _chunk_doc(doc, max_chars, overlap_chars)
    finally:
        doc.close()
