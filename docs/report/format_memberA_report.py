"""Convert Member A's report to IEEE conference format and fix two errors.

    python docs/report/format_memberA_report.py

Member A produced a complete report, but its layout does not match the
assignment's requirement: it is single-column, 11pt, with 2cm margins. The
IEEE conference format the assignment asks for is two-column, 10pt body
text, with narrower margins. That layout is what pushes the document to five
pages of content when only four are allowed.

This script edits the document in place rather than rebuilding it, so the
text, tables and figure are preserved exactly as written:

  * the default font drops from 11pt to 10pt and line spacing from 1.15 to
    single;
  * margins move to the IEEE values (top 54pt, bottom 72pt, sides 45.35pt);
  * the title and author block keep the full page width by ending with a
    continuous section break, and everything after it is set to two columns;
  * two factual errors are corrected (see FIXES below);
  * four citations and a references section are added, because the report
    makes claims about ResNet-50, transfer learning and GAP without citing
    any source, and an IEEE-format paper is expected to.

Output is COMP90086_Final_Report.docx; the uploaded original is not modified.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "COMP90086_Final_Report_MemberA.docx"
OUTPUT = HERE / "COMP90086_Final_Report.docx"

# Factual corrections. The first is a reversed comparison: E3 (99,051.2) is
# 3.60% better than E2 (102,746.2), not the other way round. The second is a
# typo in the input size.
FIXES = [
    ("E2 is 3.60% better than E3", "E3 is 3.60% better than E2"),
    ("320 × 322", "320 × 320"),
]

# In-text citations, inserted at the claim they support.
CITATIONS = [
    ("We employ an ImageNet pre-trained ResNet-50,", "We employ an ImageNet pre-trained ResNet-50 [1],"),
    ("Transfer learning provides a good initial representation",
     "Transfer learning [2] provides a good initial representation"),
    ("global average pooling (GAP)", "global average pooling (GAP) [3]"),
    ("The dataset has 8000 labeled observations", "The dataset [4] has 8000 labeled observations"),
]

REFERENCES = [
    "[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image "
    "recognition,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition "
    "(CVPR), 2016, pp. 770–778.",
    "[2] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are "
    "features in deep neural networks?” in Advances in Neural Information "
    "Processing Systems (NeurIPS), 2014, pp. 3320–3328.",
    "[3] M. Lin, Q. Chen, and S. Yan, “Network in network,” in Proc. Int. "
    "Conf. Learning Representations (ICLR), 2014.",
    "[4] T. Dogan, “House Prices and Images -- SoCal,” Kaggle, 2019. "
    "[Online]. Available: https://www.kaggle.com/datasets/ted8080/"
    "house-prices-and-images-socal",
]

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def replace_in_paragraph(paragraph, old: str, new: str) -> bool:
    """Replace text that may span several runs, preserving the first run's format."""
    runs = paragraph.runs
    if not runs:
        return False
    text = "".join(r.text for r in runs)
    if old not in text:
        return False

    start = text.index(old)
    end = start + len(old)
    pos = 0
    placed = False
    for r in runs:
        rt = r.text
        rs, re_ = pos, pos + len(rt)
        pos = re_
        if re_ <= start or rs >= end:
            continue
        prefix = rt[: start - rs] if start > rs else ""
        suffix = rt[end - rs :] if end < re_ else ""
        if not placed:
            r.text = prefix + new + suffix
            placed = True
        else:
            r.text = prefix + suffix
    return True


def set_defaults(doc) -> None:
    """10pt body text, single spacing -- IEEE conference values."""
    styles_el = doc.styles.element
    defaults = styles_el.find(f"{W}docDefaults")
    rpr = defaults.find(f"{W}rPrDefault/{W}rPr")
    for tag in ("sz", "szCs"):
        el = rpr.find(f"{W}{tag}")
        if el is not None:
            el.set(f"{W}val", "20")  # 10pt, in half-points
    ppr = defaults.find(f"{W}pPrDefault/{W}pPr")
    spacing = ppr.find(f"{W}spacing")
    if spacing is not None:
        spacing.set(f"{W}line", "240")       # single spacing
        spacing.set(f"{W}lineRule", "auto")
        spacing.set(f"{W}after", "60")       # 3pt between paragraphs


def set_margins(section) -> None:
    """IEEE conference margins, in twips (1pt = 20 twips)."""
    sect = section._sectPr
    mar = sect.find(f"{W}pgMar")
    mar.set(f"{W}top", "1080")      # 54pt
    mar.set(f"{W}bottom", "1440")   # 72pt
    mar.set(f"{W}left", "907")      # 45.35pt
    mar.set(f"{W}right", "907")
    mar.set(f"{W}header", "720")
    mar.set(f"{W}footer", "720")


def split_title_from_body(doc) -> None:
    """Keep the title and authors full-width, and set the rest to two columns.

    A sectPr inside a paragraph's properties ends a section at that paragraph,
    so putting a one-column sectPr on the affiliation line makes everything up
    to it a full-width first section. The body-level sectPr then governs
    everything after and is set to two columns.
    """
    # The affiliation line is the paragraph before the Abstract heading.
    anchor = None
    for p in doc.paragraphs[:6]:
        if p.text.strip().startswith("COMP90086 Computer Vision"):
            anchor = p
            break
    if anchor is None:
        raise RuntimeError("could not find the affiliation paragraph")

    body_sect = doc.sections[-1]._sectPr

    # Section 1 (title block): a copy of the body sectPr, forced to one column.
    title_sect = copy.deepcopy(body_sect)
    cols = title_sect.find(f"{W}cols")
    cols.set(f"{W}num", "1")
    cols.set(f"{W}space", "720")
    ppr = anchor._p.get_or_add_pPr()
    ppr.append(title_sect)

    # Section 2 (body): two columns, starting on the same page as the title.
    # w:type belongs to the section it describes and says how that section
    # begins; the default nextPage here is what was wasting a whole page on
    # the title and authors. w:type must be sectPr's first child per the schema.
    cols = body_sect.find(f"{W}cols")
    cols.set(f"{W}num", "2")
    cols.set(f"{W}space", "360")  # 18pt gutter
    for stale in body_sect.findall(f"{W}type"):
        body_sect.remove(stale)
    body_sect.insert(0, body_sect.makeelement(f"{W}type", {f"{W}val": "continuous"}))


def fit_tables(doc) -> None:
    """Scale both tables down to the width of one column.

    Member A's tables were laid out for a single-column page: each is 9576
    twips wide against a column width of about 4860, so in two-column format
    the right half falls outside the column and Word clips it -- the MSE, MAE
    and R2 headings disappeared and the numbers were truncated to six
    characters. Every grid column and cell width is scaled by the same factor
    so the proportions survive, and the cell font drops to 8pt to buy room.
    """
    column_twips = 4860
    for table in doc.tables:
        grid = table._tbl.find(f"{W}tblGrid")
        cols = grid.findall(f"{W}gridCol")
        total = sum(int(c.get(f"{W}w")) for c in cols)
        scale = column_twips / total
        for c in cols:
            c.set(f"{W}w", str(round(int(c.get(f"{W}w")) * scale)))

        tbl_pr = table._tbl.tblPr
        width_el = tbl_pr.find(f"{W}tblW")
        if width_el is not None:
            width_el.set(f"{W}w", str(column_twips))
            width_el.set(f"{W}type", "dxa")

        for row in table.rows:
            for cell, grid_col in zip(row.cells, cols):
                cell.width = int(grid_col.get(f"{W}w"))
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8)


def add_references(doc) -> None:
    head = doc.add_paragraph("References", style="Heading 1")
    for entry in REFERENCES:
        p = doc.add_paragraph(entry, style="Normal")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in p.runs:
            run.font.size = Pt(8)


def main() -> None:
    shutil.copyfile(SOURCE, OUTPUT)
    doc = Document(OUTPUT)

    applied, missed = [], []
    for old, new in FIXES + CITATIONS:
        hit = any(replace_in_paragraph(p, old, new) for p in doc.paragraphs)
        (applied if hit else missed).append(old)

    set_defaults(doc)
    split_title_from_body(doc)
    set_margins(doc.sections[-1])
    fit_tables(doc)
    add_references(doc)

    doc.save(OUTPUT)

    print(f"wrote {OUTPUT.name}")
    print(f"  corrections and citations applied: {len(applied)}")
    for a in applied:
        print(f"    + {a[:60]}")
    if missed:
        print(f"  NOT FOUND ({len(missed)}):")
        for m in missed:
            print(f"    - {m[:60]}  <- check by hand")


if __name__ == "__main__":
    main()
