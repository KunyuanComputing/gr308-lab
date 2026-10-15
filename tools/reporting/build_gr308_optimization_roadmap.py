#!/usr/bin/env python3
"""Render the GR308 Qwen optimization roadmap Markdown as a shareable PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output" / "风华_GR308_Qwen系列性能优化思路_20260916.md"
OUTPUT = ROOT / "output" / "pdf" / "风华_GR308_Qwen系列性能优化思路_20260916.pdf"

pdfmetrics.registerFont(TTFont(
    "Heiti", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0,
))
pdfmetrics.registerFont(TTFont(
    "HeitiBold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0,
))

BLUE = HexColor("#1675C1")
NAVY = HexColor("#0D2B4D")
GRAY = HexColor("#626A73")
LIGHT = HexColor("#F3F6FA")
GRID = HexColor("#C6D2DF")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCN", fontName="Heiti", fontSize=22, leading=29,
    textColor=NAVY, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="H1CN", fontName="Heiti", fontSize=15.5, leading=21,
    textColor=BLUE, spaceBefore=9, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2CN", fontName="HeitiBold", fontSize=10.5, leading=15,
    textColor=NAVY, spaceBefore=6, spaceAfter=4, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BodyCN", fontName="Heiti", fontSize=8.6, leading=13.2,
    textColor=colors.black, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="MetaCN", fontName="Heiti", fontSize=8.2, leading=12.5,
    textColor=GRAY, spaceAfter=2,
))
styles.add(ParagraphStyle(
    name="BulletCN", fontName="Heiti", fontSize=8.4, leading=12.8,
    leftIndent=12, firstLineIndent=-7, bulletIndent=3, spaceAfter=2,
))
styles.add(ParagraphStyle(
    name="CellCN", fontName="Heiti", fontSize=6.25, leading=8.3,
    textColor=colors.black,
))
styles.add(ParagraphStyle(
    name="CellHeadCN", fontName="HeitiBold", fontSize=6.2, leading=8.1,
    textColor=NAVY, alignment=TA_CENTER,
))


def inline(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(r"`([^`]+)`", r"<font color='#0D5A91'>\1</font>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value


def table_widths(columns: int, available: float) -> list[float]:
    if columns == 8:
        ratios = [1.15, 2.35, .82, .95, .82, .95, 1.0, 1.0]
    elif columns == 6:
        ratios = [.7, 1.05, 1.0, 2.45, 1.35, 1.55]
    else:
        ratios = [1.0] * columns
    scale = available / sum(ratios)
    return [ratio * scale for ratio in ratios]


def build_table(rows: list[list[str]], available: float) -> Table:
    columns = len(rows[0])
    wrapped = []
    for row_index, row in enumerate(rows):
        wrapped.append([
            Paragraph(inline(cell), styles["CellHeadCN" if row_index == 0 else "CellCN"])
            for cell in row
        ])
    table = Table(
        wrapped,
        colWidths=table_widths(columns, available),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E9EEF4")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#9FB2C6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]))
    return table


def parse_markdown(source: str, available: float) -> list:
    lines = source.splitlines()
    story = []
    index = 0
    first_title = True
    while index < len(lines):
        raw = lines[index].rstrip()
        text = raw.strip()
        if not text:
            index += 1
            continue
        if text.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) >= 2 and all(set(cell) <= {"-", ":", " "} for cell in rows[1]):
                del rows[1]
            story.extend([build_table(rows, available), Spacer(1, 5)])
            continue
        if text.startswith("# "):
            story.append(Paragraph(inline(text[2:]), styles["TitleCN"]))
            first_title = False
        elif text.startswith("## "):
            story.append(Paragraph(inline(text[3:]), styles["H1CN"]))
        elif text.startswith("### "):
            story.append(Paragraph(inline(text[4:]), styles["H2CN"]))
        elif text.startswith("- "):
            story.append(Paragraph("• " + inline(text[2:]), styles["BulletCN"]))
        elif re.match(r"^\d+\.\s+", text):
            match = re.match(r"^(\d+)\.\s+(.*)$", text)
            story.append(Paragraph(
                f"{match.group(1)}. " + inline(match.group(2)), styles["BulletCN"],
            ))
        elif not first_title and (text.startswith("版本：") or text.startswith("适用环境：") or text.startswith("覆盖模型：")):
            story.append(Paragraph(inline(text.rstrip("  ")), styles["MetaCN"]))
        else:
            paragraph = [text]
            index += 1
            while index < len(lines):
                following = lines[index].strip()
                if (not following or following.startswith(("#", "- ", "|"))
                        or re.match(r"^\d+\.\s+", following)):
                    break
                paragraph.append(following)
                index += 1
            story.append(Paragraph(inline(" ".join(paragraph)), styles["BodyCN"]))
            continue
        index += 1
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(HexColor("#D6E0EA"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Heiti", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(18 * mm, 9 * mm, "风华 GR308 | Qwen 系列性能优化思路 | 2026-09-16")
    canvas.drawRightString(width - 18 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    left = right = 18 * mm
    top = 17 * mm
    bottom = 19 * mm
    frame = Frame(
        left, bottom, page_width - left - right, page_height - top - bottom,
        id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom,
        title="风华 GR308 Qwen 系列性能优化思路",
        author="OpenAI Codex",
        subject="GR308 上 Qwen 系列模型的性能优化路线与验证计划",
    )
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))
    doc.build(parse_markdown(SOURCE.read_text(), page_width - left - right))
    print(OUTPUT)


if __name__ == "__main__":
    main()
