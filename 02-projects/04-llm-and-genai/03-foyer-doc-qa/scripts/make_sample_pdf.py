"""Generates a fictitious home-insurance policy PDF for testing the pipeline
end to end. Entirely made-up data -- no real customer, policy, or claim.

Built with reportlab (not PyMuPDF) so the generator and the parser
(parsing.py, which reads with PyMuPDF) are independent -- the test isn't
"PyMuPDF reading back exactly what PyMuPDF wrote."

Deliberately constructed so the app is demonstrable:
- a rare identifier (policy number) that appears exactly once
- a claims reference that also appears exactly once
- two comparable figures (buildings vs contents sum insured) for a
  comparison-style question
- a numbered exclusions list for a list/top-k question
- three topics -- cyber, identity, refund -- that never appear anywhere,
  so a "not found" question about them has a genuine right answer

Run: python scripts/make_sample_pdf.py
"""

import os

import pymupdf  # used only to assert the absent-topics guarantee
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

POLICY_NUMBER = "FOY-HOME-2026-004417"
CLAIMS_REFERENCE = "PRC-CLAIM-24"
BUILDINGS_SUM_INSURED = "EUR 480,000"
CONTENTS_SUM_INSURED = "EUR 65,000"
ABSENT_TOPICS = ["cyber", "identity", "refund"]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "sample_policy.pdf")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 20 * mm
LINE_HEIGHT = 14


def _draw_page(c: canvas.Canvas, title: str, lines: list) -> None:
    y = PAGE_HEIGHT - MARGIN
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y, title)
    y -= LINE_HEIGHT * 1.8
    c.setFont("Helvetica", 10.5)
    for line in lines:
        if line == "":
            y -= LINE_HEIGHT * 0.6
            continue
        if line.startswith("## "):
            c.setFont("Helvetica-Bold", 11)
            y -= LINE_HEIGHT * 0.3
            c.drawString(MARGIN, y, line[3:])
            y -= LINE_HEIGHT
            c.setFont("Helvetica", 10.5)
            continue
        c.drawString(MARGIN, y, line)
        y -= LINE_HEIGHT
    c.showPage()


def build(path: str) -> None:
    c = canvas.Canvas(path, pagesize=A4)

    _draw_page(c, "Foyer Assurances -- Home Insurance Policy", [
        "This is a fictitious sample document, generated for a technical",
        "assessment. It is not a real insurance contract and describes no",
        "real person, property, or claim.",
        "",
        f"Policy number: {POLICY_NUMBER}",
        "Product: Comfort Home (buildings + contents)",
        "Policyholder: Amelie Restel",
        "Risk address: 14 Rue des Capucins, L-1313 Luxembourg",
        "Period: 1 January 2026 to 31 December 2026",
        "",
        "## Section 1 -- Sums Insured",
        f"Buildings sum insured: {BUILDINGS_SUM_INSURED}",
        f"Contents sum insured: {CONTENTS_SUM_INSURED}",
        "The buildings sum insured is higher than the contents sum insured,",
        "reflecting the higher rebuild cost of the property itself relative",
        "to its contents.",
    ])

    _draw_page(c, "Coverage", [
        "## Section 2 -- What Is Covered",
        "Fire, lightning, explosion: covered up to the buildings sum insured.",
        "Water damage from burst or leaking pipes: covered, subject to a",
        "deductible of EUR 300.",
        "Storm and hail damage: covered in full for buildings and contents.",
        "Theft following forcible entry: covered up to the contents sum",
        "insured, subject to a deductible of EUR 150.",
        "Accidental breakage of fixed glass and sanitary ware: covered.",
        "",
        "## Section 3 -- Liability",
        "Private liability arising from the insured property is covered up",
        "to EUR 6,000,000 per incident.",
        "",
        "## Section 4 -- Claims",
        f"To file a claim, contact Foyer Assurances and quote claims",
        f"reference {CLAIMS_REFERENCE}. A claims handler responds within 2",
        "working days of a complete file being received.",
    ])

    _draw_page(c, "Exclusions", [
        "## Section 5 -- Exclusions",
        "The following are not covered under this policy:",
        "",
        "1. Damage caused by normal wear and tear or gradual deterioration.",
        "2. Damage occurring while the property is unoccupied for more than",
        "   60 consecutive days without prior written agreement.",
        "3. Loss or damage caused intentionally by the policyholder.",
        "4. Damage from faulty design, workmanship, or materials.",
        "5. Loss of business income or rental income of any kind.",
        "6. Damage caused by war, invasion, or civil unrest.",
        "7. Damage caused by nuclear radiation or contamination.",
        "8. Any loss already covered under a separate, more specific policy.",
    ])

    _draw_page(c, "Premium and Contact", [
        "## Section 6 -- Premium",
        "Annual premium: EUR 512.00, payable annually by SEPA direct debit.",
        "A no-claims discount of 15% has been applied based on 3 consecutive",
        "claim-free years.",
        "",
        "## Section 7 -- Contact",
        "Foyer Assurances S.A.",
        "12 rue Leon Laval, L-3372 Leudelange, Luxembourg (fictitious",
        "address for this test document)",
        "Customer service: Monday-Friday, 8:00-18:00",
        "",
        "This is a fictitious document generated for a technical assessment",
        "and contains no real personal or policy data.",
    ])

    c.save()


def _assert_absent_topics(path: str) -> None:
    doc = pymupdf.open(path)
    try:
        full_text = "\n".join(page.get_text("text") for page in doc).lower()
    finally:
        doc.close()
    for topic in ABSENT_TOPICS:
        count = full_text.count(topic.lower())
        if count != 0:
            raise AssertionError(
                f"'{topic}' appears {count} time(s) in the generated PDF; "
                f"it must appear zero times so a question about it can "
                f"correctly trigger a not-found answer."
            )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    build(OUTPUT_PATH)
    _assert_absent_topics(OUTPUT_PATH)
    print(f"Wrote {os.path.abspath(OUTPUT_PATH)}")
