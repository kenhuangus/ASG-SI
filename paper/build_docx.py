"""Build the paper .docx from paper.md (academic layout, editable tables, captions under objects).

Source conventions:
  % title / % subtitle / % authors / % affiliations   front matter lines
  ## N Heading, ### N.M Heading                      numbered headings (text used as written)
  | a | b |  ... then a line  Table: caption           editable table, caption under it
  ![Figure N: caption](file.png)                     figure, caption under it
  ```algorithm Algorithm N: caption ... ```          boxed monospace algorithm, caption under it
  - item                                             bullet
  > text                                             shaded note box
  **bold**, *italic*, `code`, bare https:// URLs     inline styles; URLs become hyperlinks
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

SERIF = "Times New Roman"
INK = RGBColor(17, 24, 39)


def style_run(run, size=10.5, bold=None, italic=None, font=SERIF, color=INK):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_link(paragraph, url, size):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    for tag, val in (("w:rFonts", None), ("w:color", "1D4ED8"), ("w:u", "single"), ("w:sz", str(int(size * 2)))):
        el = OxmlElement(tag)
        if tag == "w:rFonts":
            el.set(qn("w:ascii"), SERIF)
            el.set(qn("w:hAnsi"), SERIF)
        else:
            el.set(qn("w:val"), val)
        rpr.append(el)
    run.append(rpr)
    text = OxmlElement("w:t")
    text.text = url
    text.set(qn("xml:space"), "preserve")
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


TOKEN = re.compile(r"(\*\*[^*]+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|https?://[^\s)\]]+)")


def add_inline(paragraph, text, size=10.5, italic=False, bold=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            style_run(paragraph.add_run(part[2:-2]), size, bold=True, italic=italic)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            style_run(paragraph.add_run(part[1:-1]), size, italic=True, bold=bold)
        elif part.startswith("`") and part.endswith("`"):
            style_run(paragraph.add_run(part[1:-1]), size - 1, font="Consolas", bold=bold)
        elif part.startswith("http"):
            trail = ""
            while part and part[-1] in ".,;:":
                trail = part[-1] + trail
                part = part[:-1]
            add_link(paragraph, part, size)
            if trail:
                style_run(paragraph.add_run(trail), size, italic=italic, bold=bold)
        else:
            style_run(paragraph.add_run(part), size, italic=italic, bold=bold)


def para(doc, text="", size=10.5, align=None, after=5, before=0, italic=False, bold=False, indent=None):
    p = doc.add_paragraph()
    f = p.paragraph_format
    f.space_after, f.space_before, f.line_spacing = Pt(after), Pt(before), 1.12
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        f.left_indent = f.right_indent = Inches(indent)
    if text:
        add_inline(p, text, size, italic=italic, bold=bold)
    return p


def caption(doc, text):
    p = para(doc, after=10, before=3, align="center")
    add_inline(p, text, 9.5, italic=True)
    p.paragraph_format.keep_together = True


def shade(cell, fill):
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill}"/>'))


def borders(table, color="94A3B8"):
    tbl_pr = table._tbl.tblPr
    xml = "".join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="{color}"/>' for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
    tbl_pr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>{xml}</w:tblBorders>'))


def keep_rows_together(table):
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True


def table(doc, rows, widths=None):
    n = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    total = 6.4
    widths = widths or [total / n] * n
    for r, row in enumerate(rows):
        for c in range(n):
            cell = t.cell(r, c)
            cell.width = Inches(widths[c])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = p.paragraph_format.space_before = Pt(1)
            text = row[c].strip() if c < len(row) else ""
            add_inline(p, text, 9, bold=(r == 0))
            if r == 0:
                shade(cell, "E2E8F0")
    keep_rows_together(t)
    return t


def boxed(doc, lines, size=8.8):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, "64748B")
    cell = t.cell(0, 0)
    cell.width = Inches(6.2)
    shade(cell, "FFFFFF")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    for i, line in enumerate(lines):
        if i:
            p.add_run().add_break()
        style_run(p.add_run(line), size, font="Consolas")
    keep_rows_together(t)


def build(src, out):
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    doc = Document()
    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = SERIF, Pt(10.5)
    root = Path(src).parent
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("% title "):
            para(doc, line[8:], 16, "center", 4, bold=True)
        elif line.startswith("% subtitle "):
            para(doc, line[11:], 12, "center", 8)
        elif line.startswith("% authors "):
            para(doc, line[10:], 11, "center", 2)
        elif line.startswith("% affiliations "):
            para(doc, line[15:], 9.5, "center", 14, italic=True)
        elif line.startswith("## "):
            h = para(doc, line[3:], 12.5, after=5, before=12, bold=True)
            h.paragraph_format.keep_with_next = True
        elif line.startswith("### "):
            h = para(doc, line[4:], 11, after=4, before=8, bold=True)
            h.paragraph_format.keep_with_next = True
        elif line.startswith("#### "):
            h = para(doc, line[5:], 10.5, after=3, before=6, bold=True, italic=True)
            h.paragraph_format.keep_with_next = True
        elif line.startswith("ABSTRACT:"):
            para(doc, "Abstract", 11, "center", 3, bold=True)
            para(doc, line[9:].strip(), 10, "justify", 10, indent=0.45)
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-:|]+\|$", lines[i].strip()):
                    block.append([c for c in lines[i].strip().strip("|").split("|")])
                i += 1
            widths = None
            if i < len(lines) and lines[i].startswith("Widths:"):
                widths = [float(x) for x in lines[i][7:].split(",")]
                i += 1
            table(doc, block, widths)
            if i < len(lines) and lines[i].startswith("Table:"):
                caption(doc, lines[i][6:].strip())
            else:
                i -= 1
        elif line.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", line)
            p = para(doc, align="center", after=2, before=6)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(root / m.group(2)), width=Inches(6.2))
            caption(doc, m.group(1))
        elif line.startswith("```algorithm"):
            cap = line[len("```algorithm"):].strip()
            body = []
            i += 1
            while not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            boxed(doc, body)
            caption(doc, cap)
        elif line.startswith("```"):
            body = []
            i += 1
            while not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            boxed(doc, body, 8.5)
            para(doc, after=4)
        elif line.startswith("- "):
            p = para(doc, after=2, align="justify")
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            add_inline(p, "•  " + line[2:], 10.5)
        elif line.startswith("> "):
            body = []
            while i < len(lines) and lines[i].startswith("> "):
                body.append(lines[i][2:])
                i += 1
            i -= 1
            t = doc.add_table(rows=1, cols=1)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            borders(t, "1D4ED8")
            cell = t.cell(0, 0)
            cell.width = Inches(6.2)
            shade(cell, "EFF6FF")
            first = True
            for b in body:
                p = cell.paragraphs[0] if first else cell.add_paragraph()
                first = False
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, b, 10)
            para(doc, after=4)
        elif line.startswith("REF "):
            p = para(doc, after=3, align="justify")
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            add_inline(p, line[4:], 9.5)
        elif line.strip():
            para(doc, line.strip(), 10.5, "justify", 5)
        i += 1
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
