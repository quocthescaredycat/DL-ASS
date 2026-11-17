"""Build the M1 PDF and assignment page from the editable proposal. No run logs."""

from html import escape
from pathlib import Path
import re

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PDF = HERE / "G09_A3_M1_Proposal.pdf"
SOURCE = HERE / "proposal.md"


def inline(text):
    """Render the small inline Markdown vocabulary used by the proposal."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  r'<a href="\2" color="#195581">\1</a>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Arial">\1</font>', text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def build_pdf(source):
    fontdir = Path("C:/Windows/Fonts")
    for name, filename in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                           ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fontdir / filename)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                  italic="Arial-Italic", boldItalic="Arial-BoldItalic")
    body = ParagraphStyle("Body", fontName="Arial", fontSize=9.1, leading=12.3,
                          textColor=colors.HexColor("#233143"), spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=8.25, leading=10.8, spaceAfter=0)
    heading = ParagraphStyle("Heading", parent=body, fontName="Arial-Bold", fontSize=11.5,
                             leading=15, textColor=colors.HexColor("#174b72"),
                             spaceBefore=9, spaceAfter=7, keepWithNext=True)
    title = ParagraphStyle("Title", parent=heading, fontSize=19, leading=23, spaceAfter=8)
    subtitle = ParagraphStyle("Subtitle", parent=heading, fontSize=13, leading=17,
                              spaceBefore=0, spaceAfter=9)
    story = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), title))
            i += 1
            continue
        if line.startswith("## "):
            style = subtitle if line[3:] == "Image-text classification with PairDX" else heading
            story.append(Paragraph(inline(line[3:]), style))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append([Paragraph(inline(c), small) for c in cells])
                i += 1
            n = len(rows[0])
            widths = ([168*mm] if n == 1 else
                      [78*mm, 90*mm] if n == 2 else [49*mm, 68*mm, 51*mm])
            if n == 3 and "Proposed class" in rows[0][0].getPlainText():
                widths = [99*mm, 36*mm, 33*mm]
            table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#b6c7d7")),
                ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#dce3ea")),
            ]))
            story.extend([table, Spacer(1, 7)])
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "<!--")):
            paragraph.append(lines[i].strip())
            i += 1
        text = " ".join(paragraph)
        if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
            text = "<i>" + inline(text[1:-1]) + "</i>"
        else:
            text = inline(text)
        story.append(Paragraph(text, body))

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bccbda"))
        canvas.line(21*mm, 18*mm, 189*mm, 18*mm)
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#536779"))
        canvas.drawString(21*mm, 13*mm, "CO3133 | Group 09 | Assignment 3 M1 | 18 November 2026")
        canvas.drawRightString(189*mm, 13*mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=(210*mm, 297*mm),
                            leftMargin=21*mm, rightMargin=21*mm,
                            topMargin=17*mm, bottomMargin=24*mm,
                            title="Group 09 - Assignment 3 M1 - PairDX Dataset Proposal",
                            author="Group 09: Lê Phước Vũ; Nguyễn Vũ Long; Nguyễn Hoàng Quốc")
    doc.build(story, onFirstPage=page, onLaterPages=page)


def build_page(source):
    rendered = MarkdownIt("commonmark").enable("table").render(source)
    rendered = rendered.replace("<!-- pagebreak -->", "")
    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Assignment 3 M1 - PairDX | CO3133</title>
  <link rel="stylesheet" href="style.css?v=2" />
  <style>
    .proposal-links { padding: 14px 18px; background: #eef3f7; border-left: 3px solid #1a4b8c; }
    .proposal table { width: 100%; border-collapse: collapse; font-size: .9rem; margin: 16px 0; }
    .proposal th, .proposal td { padding: 8px; border-bottom: 1px solid #d9d4ca; text-align: left; vertical-align: top; }
    .proposal th { background: #f1ebdf; }
    .proposal h2 { margin-top: 32px; }
    .proposal code { overflow-wrap: anywhere; font-size: .88em; }
    .proposal a { overflow-wrap: anywhere; }
    .proposal { overflow-wrap: break-word; }
    @media (max-width: 600px) {
      .page { margin: 0; padding: 24px 16px; }
      .proposal table { font-size: .75rem; }
      .proposal th, .proposal td { padding: 5px 4px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <p class="back"><a href="index.html">&larr; Back to group page</a></p>
    <p class="proposal-links"><a href="A3/G09_A3_M1_Proposal.pdf">Download M1 proposal PDF</a> &middot;
      <a href="A3/proposal.md">Editable proposal</a> &middot;
      <a href="A3/README.md">Source files and rebuild instructions</a></p>
    <article class="proposal">
CONTENT
    </article>
    <section>
      <h2>Experiment and final-deliverable status</h2>
      <p>No main experiments have been run for M1. Quantitative model results, comparison,
      error examples and checkpoints are planned for M2/M3 after dataset approval.
      The final report, slides and YouTube presentation are planned for M3.</p>
    </section>
    <footer><p>Last updated: 18 November 2026</p></footer>
  </main>
</body>
</html>
'''.replace("CONTENT", rendered)
    (ROOT / "assignment3.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    source = SOURCE.read_text(encoding="utf-8")
    build_pdf(source)
    build_page(source)
