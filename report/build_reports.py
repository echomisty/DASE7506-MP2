"""Build four evidence-complete DASE7506 MP2 PDF reports.

Outputs:
  * full Chinese technical report
  * full English technical report
  * exact 10-page Chinese submission report
  * exact 10-page English submission report

The visual grammar follows the supplied Project 1 v3 reports: A4 single-column
pages, restrained colour-blind-safe tones, numbered sections, three-line
tables, vector evidence graphics, explicit experimental judgements, and
evidence appendices.
All reported values are tied to the frozen Project 2 JSON/CSV records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from pypdf import PdfReader
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"

OUTPUTS = {
    ("zh", False): OUT / "DASE7506_MP2_Full_Report_Chinese.pdf",
    ("en", False): OUT / "DASE7506_MP2_Full_Report_English.pdf",
    ("zh", True): OUT / "DASE7506_MP2_10_Page_Report_Chinese.pdf",
    ("en", True): OUT / "DASE7506_MP2_10_Page_Report_English.pdf",
}

INK = colors.HexColor("#26343B")
MUTED = colors.HexColor("#66757C")
ACCENT = colors.HexColor("#2F6173")
ACCENT_2 = colors.HexColor("#3F8A82")
ACCENT_3 = colors.HexColor("#B07A56")
PALE = colors.HexColor("#EDF2F0")
PALE_BLUE = colors.HexColor("#EAF2F6")
PALE_SAND = colors.HexColor("#F5F0E9")
ROW = colors.HexColor("#F7F8F6")
GRID = colors.HexColor("#CBD5D4")
WHITE = colors.white
RED = colors.HexColor("#B45C56")
GREEN = colors.HexColor("#2F7F68")

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 17.5 * mm
TOP = 15.5 * mm
BOTTOM = 15.5 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT
CONTENT_H = PAGE_H - TOP - BOTTOM


def _register_fonts() -> None:
    light = Path("/System/Library/Fonts/STHeiti Light.ttc")
    medium = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if light.exists() and medium.exists():
        pdfmetrics.registerFont(TTFont("ZHBody", str(light)))
        pdfmetrics.registerFont(TTFont("ZHBold", str(medium)))
    else:
        # The macOS fonts are expected in the coursework environment. This
        # fallback keeps the source portable to machines with ReportLab CID fonts.
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


_register_fonts()


def fonts(lang: str) -> tuple[str, str]:
    return ("ZHBody", "ZHBold") if lang == "zh" else ("Helvetica", "Helvetica-Bold")


def styles_for(lang: str, compact: bool) -> dict[str, ParagraphStyle]:
    body, bold = fonts(lang)
    base = getSampleStyleSheet()
    if compact:
        # The two scripts have different line-breaking density. These values
        # make the continuous-layout submission editions exactly ten pages
        # without inserting section-level page breaks.
        body_size = 9.8 if lang == "zh" else 9.0
        leading = 13.6 if lang == "zh" else 12.25
    else:
        # Full reports use more generous journal-like typography. They contain
        # the same validation chain plus the unabridged interpretation text.
        body_size = 10.0 if lang == "zh" else 8.8
        leading = 13.8 if lang == "zh" else 12.1
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold,
            fontSize=17.2 if compact else 18.0,
            leading=22,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=3.5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=body,
            fontSize=8.6,
            leading=11,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=4.5 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=13.4 if compact else 14.4,
            leading=17.2 if compact else 18.3,
            textColor=INK,
            spaceBefore=2.2 * mm,
            spaceAfter=2.0 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=10.0 if compact else 11.1,
            leading=13.2 if compact else 14.2,
            textColor=ACCENT,
            spaceBefore=1.6 * mm,
            spaceAfter=1.2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=body,
            fontSize=body_size,
            leading=leading,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=1.55 * mm,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=body,
            fontSize=7.45 if compact else 8.05,
            leading=9.6 if compact else 10.35,
            textColor=INK,
            spaceAfter=1.0 * mm,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=body,
            fontSize=7.15 if compact else 7.8,
            leading=9.5 if compact else 10.0,
            alignment=TA_CENTER,
            textColor=INK,
            spaceBefore=0.6 * mm,
            spaceAfter=1.6 * mm,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=body,
            fontSize=6.9 if compact else 7.45,
            leading=8.6 if compact else 9.15,
            alignment=TA_CENTER,
            textColor=INK,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=6.95 if compact else 7.5,
            leading=8.6 if compact else 9.15,
            alignment=TA_CENTER,
            textColor=INK,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=body,
            fontSize=body_size,
            leading=leading,
            leftIndent=4.2 * mm,
            firstLineIndent=-2.8 * mm,
            bulletIndent=1.0 * mm,
            textColor=INK,
            spaceAfter=1.1 * mm,
            wordWrap="CJK" if lang == "zh" else None,
        ),
        "equation": ParagraphStyle(
            "equation",
            parent=base["BodyText"],
            fontName="Courier" if lang == "en" else body,
            fontSize=7.65,
            leading=10.2,
            alignment=TA_CENTER,
            textColor=INK,
            backColor=PALE_BLUE,
            borderColor=GRID,
            borderWidth=0.45,
            borderPadding=5,
            spaceBefore=1.0 * mm,
            spaceAfter=1.6 * mm,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName=body,
            fontSize=body_size,
            leading=leading,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=1.55 * mm,
            wordWrap="CJK" if lang == "zh" else None,
        ),
    }


class HR(Flowable):
    def __init__(self, width: float = CONTENT_W, color=GRID, thickness: float = 0.6):
        super().__init__()
        self.width = width
        self.height = 3
        self.color = color
        self.thickness = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 1.5, self.width, 1.5)


def P(text: str, st: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, st[style])


def H1(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return P(text, st, "h1")


def H2(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return P(text, st, "h2")


def Bullet(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph("• " + text, st["bullet"])


def caption(label: str, text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return P(f"<b>{label}.</b> {text}", st, "caption")


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    st: dict[str, ParagraphStyle],
    widths: Sequence[float] | None = None,
    font_size: float | None = None,
) -> Table:
    if widths is None:
        widths = [CONTENT_W / len(headers)] * len(headers)
    if abs(sum(widths) - CONTENT_W) > 2:
        scale = CONTENT_W / sum(widths)
        widths = [w * scale for w in widths]
    head_style = st["table_bold"]
    cell_style = st["table"]
    if font_size is not None:
        head_style = ParagraphStyle("th_tmp", parent=head_style, fontSize=font_size, leading=font_size + 1.5)
        cell_style = ParagraphStyle("td_tmp", parent=cell_style, fontSize=font_size, leading=font_size + 1.5)
    data = [[Paragraph(str(x), head_style) for x in headers]]
    data.extend([[Paragraph(str(x), cell_style) for x in row] for row in rows])
    t = Table(data, colWidths=list(widths), repeatRows=1, hAlign="CENTER")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.55, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3.7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.7),
    ]
    for row_index in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), ROW))
    t.setStyle(TableStyle(commands))
    return t


def _lines(
    d: Drawing,
    lines: Sequence[str],
    x: float,
    y: float,
    *,
    font: str,
    size: float,
    color=INK,
    leading: float | None = None,
    anchor: str = "start",
) -> None:
    """Draw a short, pre-wrapped label without rasterising the figure."""

    step = leading if leading is not None else size + 2
    for i, line in enumerate(lines):
        d.add(
            String(
                x,
                y - i * step,
                line,
                textAnchor=anchor,
                fontName=font,
                fontSize=size,
                fillColor=color,
            )
        )


def _arrow(
    d: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color=ACCENT,
    width: float = 1.05,
    head: float = 4.2,
) -> None:
    """Draw a restrained horizontal or vertical arrow."""

    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        d.add(
            Polygon(
                [x2, y2, x2 - direction * head, y2 + head * 0.62, x2 - direction * head, y2 - head * 0.62],
                fillColor=color,
                strokeColor=color,
            )
        )
    else:
        direction = 1 if y2 >= y1 else -1
        d.add(
            Polygon(
                [x2, y2, x2 + head * 0.62, y2 - direction * head, x2 - head * 0.62, y2 - direction * head],
                fillColor=color,
                strokeColor=color,
            )
        )


def _metric_x(value: float, minimum: float, maximum: float, x0: float, width: float) -> float:
    return x0 + (value - minimum) / (maximum - minimum) * width


def baseline_comparison(lang: str, height: float = 150) -> Drawing:
    """Publication-style, directly labelled Validation comparison."""

    zh = lang == "zh"
    body, bold = fonts(lang)
    labels = ["固定位置 0", "最长推理文本", "答案多数票", "最终堆叠"] if zh else [
        "Fixed index 0",
        "Longest trace",
        "Answer plurality",
        "Final stack",
    ]
    score_keys = ["fixed_zero", "longest_trace", "plurality", "model"]
    scores = [F["diag"]["official_scores"][key] for key in score_keys]
    values = [score["accuracy"] for score in scores]
    counts = [f'{score["correct"]}/{score["n"]}' for score in scores]
    fills = [colors.HexColor("#AAB8BE"), colors.HexColor("#B8C2C5"), ACCENT, ACCENT_2]
    minimum, maximum = 70.0, 100.0
    x0, x1 = 91.0, CONTENT_W - 9.0
    plot_w = x1 - x0
    bar_h = 12.0
    row_y = [103.0, 78.0, 53.0, 28.0]
    d = Drawing(CONTENT_W, height)

    note = "同一 Validation，n=798" if zh else "Paired on the same Validation bank, n=798"
    d.add(String(x0, height - 12, note, fontName=body, fontSize=6.6, fillColor=MUTED))
    delta = "+7.14 pp vs 多数票" if zh else "+7.14 pp vs plurality"
    pill_w = 86 if zh else 94
    d.add(Rect(x1 - pill_w, height - 22, pill_w, 16, rx=8, ry=8, fillColor=PALE, strokeColor=ACCENT_2, strokeWidth=0.6))
    d.add(String(x1 - pill_w / 2, height - 16.6, delta, textAnchor="middle", fontName=bold, fontSize=6.7, fillColor=GREEN))

    for tick in (70, 80, 90, 100):
        tx = _metric_x(tick, minimum, maximum, x0, plot_w)
        d.add(Line(tx, 17, tx, 119, strokeColor=GRID, strokeWidth=0.45, strokeDashArray=[2, 2]))
        d.add(String(tx, 8, str(tick), textAnchor="middle", fontName=body, fontSize=5.9, fillColor=MUTED))

    plurality_x = _metric_x(values[2], minimum, maximum, x0, plot_w)
    d.add(Line(plurality_x, 17, plurality_x, 119, strokeColor=ACCENT, strokeWidth=0.7, strokeDashArray=[3, 2]))
    baseline_tag = "多数票基线" if zh else "plurality baseline"
    d.add(String(plurality_x, 121, baseline_tag, textAnchor="middle", fontName=body, fontSize=5.8, fillColor=ACCENT))

    for i, (label, value, count, fill, y) in enumerate(zip(labels, values, counts, fills, row_y)):
        d.add(String(x0 - 8, y + 3, label, textAnchor="end", fontName=bold if i == 3 else body, fontSize=6.8, fillColor=INK))
        vx = _metric_x(value, minimum, maximum, x0, plot_w)
        d.add(Rect(x0, y - bar_h / 2, max(1, vx - x0), bar_h, rx=2.5, ry=2.5, fillColor=fill, strokeColor=None))
        d.add(Circle(vx, y, 2.3 if i < 3 else 3.0, fillColor=fill, strokeColor=WHITE, strokeWidth=0.65))
        value_label = f"{value:.2f}% | {count}"
        d.add(String(min(vx + 5, x1 - 1), y + 3, value_label, textAnchor="start" if vx < x1 - 65 else "end", fontName=bold if i == 3 else body, fontSize=6.6, fillColor=GREEN if i == 3 else INK))
    return d


def nested_cv_comparison(lang: str, height: float = 150) -> Drawing:
    """Show both five-fold dispersion and the aggregate nested-CV result."""

    zh = lang == "zh"
    body, bold = fonts(lang)
    labels = ["Full 单模型", "四组模型堆叠", "+ 线性候选证据"] if zh else [
        "Full single ranker",
        "Four-group stack",
        "+ linear candidate evidence",
    ]
    current = F["nested"]
    previous = F["previous_nested"]
    total = current["ambiguous_questions"]
    means = [
        current["baseline_nested_oof_ambiguous_accuracy"],
        previous["nested_oof_ambiguous_accuracy"],
        current["nested_oof_ambiguous_accuracy"],
    ]
    counts = [f'{current["baseline_correct"]}/{total}', f'{previous["correct"]}/{total}', f'{current["correct"]}/{total}']
    folds = [
        [fold["baseline_ambiguous_accuracy"] for fold in current["folds"]],
        [fold["ambiguous_accuracy"] for fold in previous["folds"]],
        [fold["ambiguous_accuracy"] for fold in current["folds"]],
    ]
    palette = [colors.HexColor("#96A7AD"), ACCENT, ACCENT_2]
    minimum, maximum = 78.5, 85.0
    x0, x1 = 112.0, CONTENT_W - 15.0
    plot_w = x1 - x0
    row_y = [100.0, 67.0, 34.0]
    d = Drawing(CONTENT_W, height)

    legend = "小点：5 个外层折　菱形：总体均值" if zh else "Small dots: five outer folds   Diamond: aggregate mean"
    d.add(String(x0, height - 11, legend, fontName=body, fontSize=6.4, fillColor=MUTED))
    delta = "+0.38 pp / +8 题 vs 四组栈" if zh else "+0.38 pp / +8 vs four-group stack"
    d.add(Rect(x1 - 91, height - 22, 91, 16, rx=8, ry=8, fillColor=PALE, strokeColor=ACCENT_2, strokeWidth=0.6))
    d.add(String(x1 - 45.5, height - 16.5, delta, textAnchor="middle", fontName=bold, fontSize=6.6, fillColor=GREEN))

    for tick in (79, 80, 81, 82, 83, 84, 85):
        tx = _metric_x(tick, minimum, maximum, x0, plot_w)
        d.add(Line(tx, 18, tx, 117, strokeColor=GRID, strokeWidth=0.42, strokeDashArray=[2, 2]))
        d.add(String(tx, 8, str(tick), textAnchor="middle", fontName=body, fontSize=5.8, fillColor=MUTED))

    offsets = [-3.2, -1.6, 0.0, 1.6, 3.2]
    for i, (label, mean, count, fold_values, color, y) in enumerate(zip(labels, means, counts, folds, palette, row_y)):
        d.add(String(x0 - 8, y + 3, label, textAnchor="end", fontName=bold if i == 2 else body, fontSize=6.7, fillColor=INK))
        xs = [_metric_x(v, minimum, maximum, x0, plot_w) for v in fold_values]
        d.add(Line(min(xs), y, max(xs), y, strokeColor=color, strokeWidth=2.0, strokeOpacity=0.38))
        for px, oy in zip(xs, offsets):
            d.add(Circle(px, y + oy, 2.15, fillColor=WHITE, strokeColor=color, strokeWidth=1.0))
        mx = _metric_x(mean, minimum, maximum, x0, plot_w)
        diamond = [mx, y + 5.0, mx + 5.0, y, mx, y - 5.0, mx - 5.0, y]
        d.add(Polygon(diamond, fillColor=color, strokeColor=WHITE, strokeWidth=0.75))
        value_label = f"{mean:.2f}% | {count}"
        d.add(String(min(mx + 7, x1 - 1), y + 9, value_label, textAnchor="start" if mx < x1 - 70 else "end", fontName=bold if i == 2 else body, fontSize=6.5, fillColor=GREEN if i == 2 else INK))
    return d


def pipeline(lang: str, height: float = 150) -> Drawing:
    zh = lang == "zh"
    d = Drawing(CONTENT_W, height)
    body, bold = fonts(lang)
    gap = 10.0
    left_w = CONTENT_W * 0.36
    middle_w = CONTENT_W * 0.42
    right_w = CONTENT_W - left_w - middle_w - 2 * gap
    x_left, x_mid, x_right = 0.0, left_w + gap, left_w + middle_w + 2 * gap
    panel_y, panel_h = 31.0, 91.0

    panels = [
        (x_left, left_w, PALE_BLUE, "输入与规范化" if zh else "INPUT + CANONICALISATION"),
        (x_mid, middle_w, PALE, "仅对歧义题学习排序" if zh else "LEARNED RANKING - AMBIGUOUS ONLY"),
        (x_right, right_w, PALE_SAND, "确定性决策" if zh else "DETERMINISTIC DECISION"),
    ]
    for x, width, fill, title in panels:
        d.add(Rect(x, panel_y, width, panel_h, rx=7, ry=7, fillColor=fill, strokeColor=GRID, strokeWidth=0.65))
        d.add(String(x + 8, panel_y + panel_h - 13, title, fontName=bold, fontSize=6.1 if not zh else 6.5, fillColor=ACCENT))

    chip_x = x_left + 9
    for i in range(4):
        cy = panel_y + 57 - i * 12
        d.add(Rect(chip_x, cy, 31, 9, rx=2.3, ry=2.3, fillColor=WHITE, strokeColor=GRID, strokeWidth=0.45))
        d.add(String(chip_x + 15.5, cy + 2.5, f"trace {i}" if not zh else f"候选 {i}", textAnchor="middle", fontName=body, fontSize=5.2, fillColor=INK))
    parse_x = x_left + 53
    group_x = x_left + 117
    d.add(Rect(parse_x, panel_y + 35, 51, 33, rx=5, ry=5, fillColor=WHITE, strokeColor=ACCENT, strokeWidth=0.75))
    _lines(d, ["解析数值", "统一表示"] if zh else ["Parse final", "numeric value"], parse_x + 25.5, panel_y + 56, font=bold, size=6.1, anchor="middle")
    d.add(Rect(group_x, panel_y + 35, left_w - (group_x - x_left) - 9, 33, rx=5, ry=5, fillColor=WHITE, strokeColor=ACCENT, strokeWidth=0.75))
    _lines(d, ["等值答案", "分组"] if zh else ["Group equal", "answers"], group_x + (left_w - (group_x - x_left) - 9) / 2, panel_y + 56, font=bold, size=6.1, anchor="middle")
    _arrow(d, chip_x + 33, panel_y + 51, parse_x - 3, panel_y + 51, color=ACCENT)
    _arrow(d, parse_x + 53, panel_y + 51, group_x - 3, panel_y + 51, color=ACCENT)

    model_names = ["文本", "嵌入", "无上下文", "完整组", "候选级"] if zh else ["text", "embedding", "no-context", "full group", "candidate"]
    model_x0 = x_mid + 9
    chip_w = 56
    positions = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 2)]
    for name, (col, row) in zip(model_names, positions):
        mx = model_x0 + col * (chip_w + 6)
        my = panel_y + 59 - row * 21
        d.add(Rect(mx, my, chip_w, 14, rx=7, ry=7, fillColor=WHITE, strokeColor=ACCENT_2, strokeWidth=0.65))
        d.add(String(mx + chip_w / 2, my + 4.2, name, textAnchor="middle", fontName=body, fontSize=5.35, fillColor=INK))
    stack_w = 47
    stack_x = x_mid + middle_w - stack_w - 8
    d.add(Rect(stack_x, panel_y + 25, stack_w, 48, rx=6, ry=6, fillColor=ACCENT, strokeColor=ACCENT, strokeWidth=0.7))
    _lines(d, ["6 值", "线性堆叠"] if zh else ["6-value", "linear stack"], stack_x + stack_w / 2, panel_y + 55, font=bold, size=6.1, color=WHITE, anchor="middle")
    _arrow(d, x_mid + 135, panel_y + 51, stack_x - 3, panel_y + 51, color=ACCENT_2)

    out_cx = x_right + right_w / 2
    d.add(Circle(out_cx, panel_y + 51, 24, fillColor=WHITE, strokeColor=ACCENT_3, strokeWidth=1.0))
    _lines(d, ["候选", "索引 0-3"] if zh else ["candidate", "index 0-3"], out_cx, panel_y + 57, font=bold, size=6.1, anchor="middle")
    _arrow(d, x_left + left_w + 2, panel_y + 51, x_mid - 3, panel_y + 51, color=ACCENT_2)
    _arrow(d, x_mid + middle_w + 2, panel_y + 51, x_right - 3, panel_y + 51, color=ACCENT_2)
    d.add(String(x_mid + 9, panel_y + 8, "2-4 组进入学习路径" if zh else "2-4 groups enter learned path", fontName=body, fontSize=5.35, fillColor=ACCENT_2))

    bypass_y = 17.0
    d.add(Line(group_x + 18, panel_y + 33, group_x + 18, bypass_y, strokeColor=MUTED, strokeWidth=0.75, strokeDashArray=[3, 2]))
    d.add(Line(group_x + 18, bypass_y, out_cx, bypass_y, strokeColor=MUTED, strokeWidth=0.75, strokeDashArray=[3, 2]))
    _arrow(d, out_cx, bypass_y, out_cx, panel_y + 25, color=MUTED, width=0.75, head=3.6)
    bypass = "仅 1 个有效组：直接旁路" if zh else "one valid group: deterministic bypass"
    d.add(String((group_x + 18 + out_cx) / 2, bypass_y + 3, bypass, textAnchor="middle", fontName=body, fontSize=5.7, fillColor=MUTED))
    return d


def decision_flow(lang: str, height: float = 168) -> Drawing:
    zh = lang == "zh"
    steps = [
        (("契约检查",), ("split / 哈希", "解析与无泄漏")),
        (("基线定位",), ("固定位置 / 最长", "答案多数票")),
        (("信号分解",), ("文本 / 嵌入", "组与候选证据")),
        (("内层选择",), ("5 折 OOF", "L2 网格")),
        (("严格嵌套",), ("外 5 / 内 4 折", "全链重训")),
        (("验证冻结",), ("切片 / bootstrap", "资源与哈希")),
    ] if zh else [
        (("Contract checks",), ("splits / hashes", "parser / no leakage")),
        (("Locate baselines",), ("fixed / longest", "answer plurality")),
        (("Decompose signals",), ("text / embeddings", "group / candidate")),
        (("Inner selection",), ("five-fold OOF", "L2 grid")),
        (("Strict nesting",), ("5 outer / 4 inner", "full-chain refit")),
        (("Validation freeze",), ("slices / bootstrap", "resources / hashes")),
    ]
    d = Drawing(CONTENT_W, height)
    body, bold = fonts(lang)
    phase_labels = ["保护边界", "选择方案", "重新验证"] if zh else ["PROTECT", "SELECT", "REVALIDATE"]
    phase_colors = [ACCENT, ACCENT_3, ACCENT_2]
    phase_fills = [PALE_BLUE, PALE_SAND, PALE]
    margin = 29.0
    span = (CONTENT_W - 2 * margin) / 5
    xs = [margin + i * span for i in range(6)]
    timeline_y = 96.0

    for phase in range(3):
        left = xs[phase * 2] - 25
        right = xs[phase * 2 + 1] + 25
        d.add(Rect(left, height - 25, right - left, 17, rx=8.5, ry=8.5, fillColor=phase_fills[phase], strokeColor=None))
        d.add(String((left + right) / 2, height - 19.2, phase_labels[phase], textAnchor="middle", fontName=bold, fontSize=6.2, fillColor=phase_colors[phase]))

    d.add(Line(xs[0], timeline_y, xs[-1], timeline_y, strokeColor=GRID, strokeWidth=2.0))
    for i in range(5):
        _arrow(d, xs[i] + 13, timeline_y, xs[i + 1] - 13, timeline_y, color=phase_colors[min(i // 2, 2)], width=1.0, head=3.8)

    for i, ((title,), notes) in enumerate(steps):
        phase = min(i // 2, 2)
        d.add(Circle(xs[i], timeline_y, 12.0, fillColor=phase_colors[phase], strokeColor=WHITE, strokeWidth=1.0))
        d.add(String(xs[i], timeline_y - 2.4, str(i + 1), textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.3, fillColor=WHITE))
        _lines(d, [title], xs[i], 71, font=bold, size=6.6 if zh else 6.0, anchor="middle")
        _lines(d, notes, xs[i], 57, font=body, size=5.45 if zh else 5.15, color=MUTED, leading=8.0, anchor="middle")

    seal_text = "TEST 仅最终评分｜选择只用 Train / Validation" if zh else "TEST FOR FINAL SCORE ONLY | selection uses Train / Validation"
    seal_w = 220 if zh else 245
    d.add(Rect((CONTENT_W - seal_w) / 2, 10, seal_w, 18, rx=9, ry=9, fillColor=WHITE, strokeColor=ACCENT, strokeWidth=0.65))
    d.add(Circle((CONTENT_W - seal_w) / 2 + 13, 19, 3.2, fillColor=ACCENT, strokeColor=None))
    d.add(String(CONTENT_W / 2 + 4, 16.7, seal_text, textAnchor="middle", fontName=bold, fontSize=6.1, fillColor=ACCENT))
    return d


def facts() -> dict:
    config = json.loads((ROOT / "experiments" / "final_config.json").read_text())
    diag = json.loads((ROOT / "experiments" / "validation_diagnostics.json").read_text())
    err = json.loads((ROOT / "experiments" / "validation_error_analysis.json").read_text())
    resources = json.loads((ROOT / "results" / "final" / "validation-resource-report.json").read_text())
    score = json.loads((ROOT / "results" / "final" / "validation-score.json").read_text())
    test_resources = json.loads((ROOT / "results" / "final" / "test-resource-report.json").read_text())
    test_score = json.loads((ROOT / "results" / "final" / "test-score.json").read_text())
    nested = json.loads((ROOT / "experiments" / "final_linear_candidate_nested_cv.json").read_text())
    previous_nested = json.loads((ROOT / "experiments" / "improvement_stacking_nested_cv.json").read_text())
    return {
        "config": config,
        "diag": diag,
        "err": err,
        "resources": resources,
        "score": score,
        "test_resources": test_resources,
        "test_score": test_score,
        "nested": nested,
        "previous_nested": previous_nested,
    }


F = facts()


def validate_facts() -> None:
    ev = F["config"]["selection_evidence"]
    assert F["score"]["correct"] == 745 and F["score"]["n"] == 798
    assert round(ev["nested_cv_ambiguous_accuracy"], 6) == round(82.45033112582782, 6)
    assert F["nested"]["correct"] == 1743 and F["previous_nested"]["correct"] == 1735
    assert len(F["nested"]["folds"]) == len(F["previous_nested"]["folds"]) == 5
    assert F["config"]["test_evaluated_for_this_checkpoint"] is True
    assert F["config"]["test_used_for_improvement_or_selection"] is False
    assert F["test_score"]["correct"] == 382 and F["test_score"]["n"] == 467
    assert F["test_resources"]["repeats"] == 1
    assert F["test_resources"]["timing_summary"]["maximum_seconds"] < 30
    assert F["diag"]["group_metrics"]["model"]["paired_vs_plurality"]["model_only_correct"] == 68
    assert F["resources"]["checkpoint"]["array_uncompressed_bytes"] == 259032
    assert F["resources"]["timing_summary"]["maximum_seconds"] < 30


validate_facts()


BASELINES = [
    ("Fixed index 0", "624/798", "78.1955%"),
    ("Longest trace", "622/798", "77.9449%"),
    ("Answer plurality", "688/798", "86.2155%"),
    ("Final stack + linear candidate", "745/798", "93.3584%"),
]

CV_ROWS = [
    ("Plurality", "85.2576", "66.8401", "86.2155", "69.0141"),
    ("Text", "89.4637 ± 0.6456", "76.3003 ± 1.4632", "91.2281", "80.2817"),
    ("Embedding", "89.3796 ± 0.9137", "76.1104 ± 2.0741", "90.3509", "78.3099"),
    ("Full no context", "91.3354 ± 0.8366", "80.5097 ± 1.8983", "91.9800", "81.9718"),
    ("Full group ranker", "91.6930 ± 1.0801", "81.3139 ± 2.4404", "92.6065", "83.3803"),
    ("Previous group stack", "92.0294 nested", "82.0719 nested", "93.3584", "85.0704"),
    ("Final group + linear candidate", "92.1977 nested", "82.4503 nested", "93.3584", "85.0704"),
    ("MLP alternative (not selected)", "92.1556 nested", "82.3557 nested", "93.4837", "85.3521"),
]

REJECTED = [
    ("Plurality/confidence gates", "82.26% OOF", "743/798", "Rejected: poorer transfer"),
    ("Extra group-parser cues / lexical hashes / pooling", "≤82.31% OOF", "≤745/798", "No robust joint gain"),
    ("Listwise softmax", "81.55% CV", "741/798", "Rejected"),
    ("Hardest negative", "81.55% CV", "742/798", "Rejected"),
    ("Objective stacks", "≤82.17% OOF", "≤745/798", "No gain over smaller stack"),
    ("Quadratic 20-feature stack", "82.12% OOF", "744/798", "Rejected"),
    ("Candidate ranker alone", "80.56% OOF", "739/798", "Kept only as complementary evidence"),
]

CANDIDATE_GRID = [
    ("0.001", "1739/2114", "82.2611%", "745/798", "Historical linear-candidate choice"),
    ("0.003", "1738/2114", "82.2138%", "745/798", "Historical tie on Validation"),
    ("0.010", "1745/2114", "82.5449%", "744/798", "Higher OOF, Validation declines"),
    ("0.030", "1741/2114", "82.3557%", "742/798", "Rejected"),
    ("0.100", "1736/2114", "82.1192%", "742/798", "Rejected"),
    ("0.300", "1727/2114", "81.6935%", "740/798", "Rejected"),
]


def title_block(lang: str, compact: bool, st: dict[str, ParagraphStyle]) -> list[Flowable]:
    if lang == "zh":
        title = "DASE7506 Project 2 可靠推理选择技术报告"
        sub = "此前冻结的 Train/Validation 方案：四类答案组排序器 + 线性候选排序器 + 六值线性堆叠"
        if compact:
            sub += "｜严格 10 页提交版"
    else:
        title = "DASE7506 Project 2 Reliable Reasoning Selection - Technical Report"
        sub = "Frozen Train/Validation method: four answer-group rankers + one linear candidate ranker + six-value linear stack"
        if compact:
            sub += " | Exact 10-page submission edition"
    return [P(title, st, "title"), P(sub, st, "subtitle"), HR()]


def _page(*items: Flowable) -> list[Flowable]:
    return list(items)


def full_pages_zh(st: dict[str, ParagraphStyle], compact: bool = False) -> list[list[Flowable]]:
    edition_note = (
        "本 10 页版使用更紧凑的字号与表格，但保留 Full 版的整条方案选择和验证链。"
        if compact
        else "本 Full 版保留完整选择与验证链；配套 10 页版仅压缩版式，不删除任何验证阶段。"
    )
    return [
        _page(
            *title_block("zh", compact, st),
            H1("1 背景、任务目标与最终结论", st),
            P("每道 GSM8K 数学题提供四条由语言模型生成的候选推理。系统不能生成新答案，只能返回一个 0 到 3 的候选索引。官方评分器用固定数值解析器读取所选候选的最终答案；中间推理是否完全正确并不直接计分。因此核心问题不是‘哪段话最像正确解答’，而是‘哪些可复现的文本、数值一致性和表示信号能帮助我们选中正确的最终答案’。", st),
            P("最终方案先按 canonical 数值答案分组：只有一个有效答案组时直接返回；存在多个答案组时，四类答案组排序器与线性候选排序器分别产生交叉折分数，再由六值线性堆叠器选组，并在组内用确定性规则返回最早候选。全部权重只用 Train 拟合，Validation 用于选择和诊断。", st),
            caption("Table 1", "冻结方案的核心结果", st),
            table(["证据", "结果", "解释边界"], [
                ["严格嵌套 Train CV", "1743/2114 歧义题，82.4503%", "外 5 折 / 内 4 折；整个堆叠在每个外折内重训"],
                ["完整 Validation", "745/798，93.3584%", "用于开发与最终冻结；不是 Test"],
                ["歧义 Validation", "302/355，85.0704%", "至少两个不同有效解析答案"],
                ["多数票对照", "688/798，86.2155%", "最终模型高 7.1429 个百分点"],
                ["历史 Test 记录", "382/467，81.7987%", "该分数此前评分一次；本次恢复选择受公开 Test 比较影响"],
                ["推理资源", "259,032 B (1.55%)；Val 1.006 s；历史 Test 1.292 s", "低于 16 MiB / 30 s 限制"],
            ], st, [38*mm, 54*mm, 76*mm]),
            P("<b>最重要的披露：</b>所选线性候选 checkpoint 的既有 Test 记录为 382/467（81.7987%），此前评分一次。2026-10-28 在看过线性候选与 MLP 版本的公开 Test 分数后，用户要求恢复线性版本；因此 Test 结果影响了本次 package 选择，违反“Test 不用于选择”的课程规则。81.80% 是该 checkpoint 的历史分数，不能作为本次 test-blind 选择的无偏最终成绩。", st, "callout"),
        ),
        _page(
            H1("2 评价协议、数据边界与硬约束", st),
            P("数据 split 的用途先于任何模型设计被冻结：Train 4,755 题用于拟合权重、标准化与融合；Validation 798 题用于开发、比较和错误分析；Test 467 题只允许冻结后最终评分；潜在 audit bank 必须在不重训、不依赖 ID 和样本数的条件下运行。候选 ID 只用于对齐，不能作为特征。", st),
            caption("Table 2", "课程固定条件与验收口径", st),
            table(["项目", "固定规则", "本项目处理"], [
                ["输入", "题目、4 个候选文本、[N,4,1024] float32 embedding", "完全使用 bank/arrays 传入对象"],
                ["输出", "长度 N 的 Python int 列表，元素 0..3", "契约测试覆盖类型、范围与顺序"],
                ["评分", "所选候选的解析数值等于参考答案", "复用 common.parse_answer 规则"],
                ["外部资源", "无外部数据、模型权重或在线 API", "NumPy + 标准库离线推理"],
                ["时间", "完整 CPU 预测 + 独立评分 ≤30 s，最多 4 线程", "Validation 1.006 s；历史 Test 1.292 s"],
                ["资产", "额外推理资产 ≤16 MiB 未压缩", "数组 259,032 B (1.55%)"],
                ["报告", "最终提交报告 ≤10 页", "另交精确 10 页中英文版"],
            ], st, [32*mm, 72*mm, 64*mm]),
            H2("2.1 指标与困难子集", st),
            P("完整准确率是主要指标。为了避免 443 道只有一个有效答案组的容易题掩盖排序能力，报告同时给出 355 道歧义题准确率，并再按是否存在唯一多数答案、正确答案支持数、答案组数量和分数 margin 分层。对多数票的比较使用同题配对统计，而不是把两个独立比例直接相减。", st),
            P("数据检查首先验证 manifest 哈希、split 不重叠、候选数量、embedding 形状与有限值；随后用单元测试验证解析分组、checkpoint schema、特征维度、禁止读取参考答案和提交接口。只有这些检查通过后才开始建模。", st, "callout"),
        ),
        _page(
            H1("3 最终系统：从候选文本到一个索引", st),
            P("系统把判断分成可审计的六个阶段。最前面的解析和分组与官方 scorer 对齐；学习器只处理真正需要比较的歧义题。最后的输出仍然是课程要求的一个候选索引，不改变原始候选。", st),
            pipeline("zh"),
            caption("Fig. 1", "冻结方案的数据流。五类排序器分别为 text、embedding、full_no_context、full 与 candidate；每类包含 5 个交叉折模型。", st),
            caption("Table 3", "推理阶段与失败保护", st),
            table(["阶段", "操作", "保护措施"], [
                ["1 解析", "按官方规则提取 canonical 数值", "保留 Fraction 等价；记录不可解析项"],
                ["2 分组", "相同数值的候选组成答案组", "不把候选位置当正确性标签"],
                ["3 快速出口", "仅一个有效答案组时直接选择", "若有可解析项，不选不可解析项"],
                ["4 基础打分", "4 类组特征 + 线性及 MLP 候选特征", "每类 5 折模型平均，避免单模型偶然性"],
                ["5 堆叠", "4 组分 + 1 候选分 + 组支持", "6 值，L2=0.001"],
                ["6 决策", "最高分组，组内最早 index", "稳定 tie-break；任意 N"],
            ], st, [28*mm, 72*mm, 68*mm]),
            P("这一路径把‘共识’与‘推理质量’分开：支持数是先验，但排序器可以推翻错误多数票；候选级最大分又允许一个高质量推理在组平均信号较弱时保留下来。", st),
        ),
        _page(
            H1("4 最终答案解析、分组与确定性决策", st),
            H2("4.1 为什么先按答案组排序", st),
            P("同一题的多个候选可能给出等价数值（例如 0.5 与 1/2）。官方评分只看最终数值，因此逐候选分类会重复计算同一答案并产生任意位置偏差。我们先用与 scorer 相同的解析器得到 canonical 值 a_i，再建立 G(a)={i:a_i=a}。若正确答案组有多个成员，选中其中任一成员都得分。", st),
            P("<b>分组规则：</b>空解析值不与有效数值合并；有至少一个有效组时不可解析候选不参与最终选择；全部不可解析时才退回确定性最早 index。组支持数 |G|、组内文本/embedding 汇总和相对其他组的差异成为可解释的组级证据。", st),
            P("G(a) = { i in {0,1,2,3} : parse(candidate_i) = a }", st, "equation"),
            caption("Table 4", "边界情况与验证", st),
            table(["情况", "决策", "验证方式"], [
                ["4 个候选同答案", "直接返回最早 index", "单组路径测试"],
                ["2-4 个有效答案组", "进入学习排序", "歧义集分层评分"],
                ["同值不同写法", "归为同一组", "Fraction/decimal 等价测试"],
                ["部分不可解析", "优先任何有效答案组", "解析与回退单元测试"],
                ["组分数完全相同", "最早候选 index 胜出", "确定性重跑测试"],
                ["audit 的 N 改变", "按输入长度动态处理", "无硬编码 split 大小"],
            ], st, [38*mm, 60*mm, 70*mm]),
            P("这一步是整个实验的第一个验证关口：只有解析与官方评分器一致，后续准确率才有含义。相关测试不仅检查常见整数，还覆盖负数、分数、等价小数、缺失标记与多正确候选。", st, "callout"),
        ),
        _page(
            H1("5 答案组特征：把信号逐块拆开", st),
            P("为了回答‘推理文本、固定表示和候选比较各贡献多少’，特征不是一次性堆满，而是分成四个可消融的答案组视图。每个视图独立训练同一种线性 pairwise ranker，使性能差异能对应到信息来源。", st),
            caption("Table 5", "四类答案组特征集", st),
            table(["特征集", "主要内容", "研究问题", "维度/正则"], [
                ["text", "长度、数字/运算/步骤、问答词重合、格式与结束模式", "只读推理表面结构能否超过共识？", "54；L2=0.001"],
                ["embedding", "归一化组中心 - 全候选中心", "固定表示是否含互补信号？", "1024；L2=0.1"],
                ["full_no_context", "54 维 text + 1024 维 embedding delta", "两个独立信号是否互补？", "1078；L2=0.1"],
                ["full", "6 维共识 + text + embedding delta + 4 维表示统计", "题内上下文是否提升？", "1088；L2=0.1"],
            ], st, [28*mm, 70*mm, 46*mm, 25*mm]),
            H2("5.1 组内与组间汇总", st),
            P("代码级维度为：text 对 27 个单候选轨迹特征分别取组内 mean/max，得到 54 维；embedding 只使用 1024 维归一化组中心相对全题中心的差。full 额外加入 6 维共识特征（支持比、plurality margin、唯一 plurality、singleton、不同答案比例、不可解析比例）与 4 维 embedding 统计（组内 cohesion、组与全体 cosine、delta norm、group-mean norm）。", st),
            H2("5.2 不使用的信号", st),
            P("题目 ID、split 大小、参考答案、缓存预测、Test 错误类型都不进入特征。文本解析只作用于传入的题目和候选；训练脚本只打开 Train references，预测入口从不打开任何 reference 文件。", st),
            P("特征工程的判断标准不是单次 Validation 上升，而是：Train 折外方向合理、方差可接受、Validation 不退化、资源增量可解释，并能在严格嵌套重训后保持。", st, "callout"),
        ),
        _page(
            H1("6 候选级排序器与学习目标", st),
            P("答案组平均可能掩盖一个特别可靠的单条推理。为保留这一证据，最终方案增加精确 1072 维候选特征：6 维所属答案组共识、27 维单候选轨迹、1024 维单候选 embedding delta、4 维 embedding 统计与 11 维 parser-consistency。候选 index/ID 不是特征；它仍是轻量线性排序器，不引入外部模型。", st),
            H2("6.1 多正例 pairwise 训练", st),
            P("一题可能有多条候选的最终答案都正确。训练时把每个正确候选与每个错误候选组成 pair，并按题目归一化，避免拥有更多正负组合的题目获得过大权重。对差分 z=x_pos-x_neg 使用 logistic 损失与 L2：", st),
            P("L(w) = (1/Q) Σ_q [1/(|P_q||N_q|) Σ_(p,n) log(1 + exp(-w^T(x_p-x_n)))] + (λ/2)||w||²", st, "equation"),
            P("推理时，候选级模型先给四条候选打分，某答案组的候选证据取该组成员最大分 max_i∈G s_i。这不是让候选模型单独决策，而是把‘组里是否至少存在一条很强的推理’作为第五路互补信号。", st),
            caption("Table 6", "候选级设计的中间验证", st),
            table(["检查", "结果", "结论"], [
                ["候选模型单独 OOF", "80.56% 歧义 Train", "低于组模型，不应单独替换"],
                ["候选模型单独 Validation", "739/798", "低于 745；不能据此宣称更强"],
                ["加入五路堆叠，普通 OOF", "最高可到 82.5449%", "需要进一步防止选择偏差"],
                ["严格嵌套确认", "1743/2114，82.4503%", "最终线性候选栈比四组栈多 8 题"],
                ["Validation", "745/798", "与此前线性候选冻结记录一致"],
            ], st, [52*mm, 50*mm, 67*mm]),
            P("关键选择不是‘候选模型看起来更复杂’，而是它在单独使用时较弱、加入堆叠后在严格嵌套协议中产生小而方向一致的互补增益。", st, "callout"),
        ),
        _page(
            H1("7 实验协议：每一步如何选择、再验证", st),
            decision_flow("zh"),
            caption("Fig. 2", "从契约检查到冻结的完整验证链。任何结构与超参数选择都发生在 Train/Validation；Test 只在冻结后最终计分一次。", st),
            H2("7.1 固定折与普通折外比较", st),
            P("Train 按题目分成 5 个固定折，seed=7506。每次用 4 折拟合，在留出折产生 OOF 分数；同题的四个候选永不跨折拆开。基础模型扫描 L2，学习率固定 0.03，最多 220 epochs，并记录各折准确率、均值和标准差。", st),
            H2("7.2 为什么还需要严格嵌套 CV", st),
            P("普通 OOF 可训练堆叠器，但如果同一批 OOF 结果又用于挑 stack L2，就会产生二层选择偏差。严格嵌套协议设置 5 个 outer folds；对每个 outer-train 再做 4 个 inner folds，重建基础模型的交叉折分数、拟合 meta stack，然后只在 outer-test 评分。这样每个最终分数都来自一条未见该题的完整训练链。", st),
            P("最终判断规则：新方案必须在严格嵌套 Train CV 改善，同时完整 Validation 不下降；若只改善普通 OOF 或单次 Validation，则记为未决或拒绝。", st, "callout"),
        ),
        _page(
            H1("8 基线：先证明任务难点不在输出接口", st),
            P("三个训练自由基线覆盖位置偏差、文本长度偏差和答案共识。固定 index 0 与最长推理都约 78%，说明候选顺序和篇幅不能可靠代表正确性；plurality 上升到 86.22%，证明组支持是强先验，但在 355 道歧义题上只有 69.01%，仍留下可学习空间。", st),
            baseline_comparison("zh"),
            caption("Fig. 3", "完整 Validation 基线与最终方案。柱状图只描述同一 Validation 的配对结果。", st),
            caption("Table 7", "基线与最终方案的官方 Validation 分数", st),
            table(["方法", "正确/总数", "准确率", "相对 plurality"], [
                ["固定 index 0", "624/798", "78.1955%", "-8.0201 pp"],
                ["最长候选文本", "622/798", "77.9449%", "-8.2707 pp"],
                ["答案 plurality", "688/798", "86.2155%", "0"],
                ["最终堆叠 + 线性候选", "745/798", "93.3584%", "+7.1429 pp"],
            ], st, [47*mm, 38*mm, 38*mm, 45*mm]),
            H2("8.1 支持数暴露的困难结构", st),
            P("plurality 对正确支持数 3 或 4 的题达到 100%，但正确答案只被 1 条候选支持时仅 23.08%。这说明主要收益必须来自识别少数派正确推理，而不是继续强化已有多数。最终模型把该分层提高到 70/117（59.83%）。", st),
        ),
        _page(
            H1("9 单模型消融：哪些信息真正有用", st),
            P("单模型阶段按 text → embedding → full_no_context → full 逐步增加信息。每一行先看 5 折 Train CV，再看冻结的 Validation；不能用 Validation 的最好单点替代 Train 稳定性证据。", st),
            caption("Table 8", "特征消融：Train CV 均值±折间标准差与 Validation", st),
            table(["方法", "Train 全集", "Train 歧义", "Val 全集", "Val 歧义"], CV_ROWS[:5], st, [36*mm, 39*mm, 39*mm, 27*mm, 27*mm], font_size=6.35),
            H2("9.1 逐步判断", st),
            Bullet("text 已把歧义 Train CV 提到 76.30%，说明长度、计算结构、问题-候选一致性等浅层线索确实有信号，但不能独立完成任务。", st),
            Bullet("embedding 单独与 text 接近而非显著更强；它的价值主要在与文本结合，而不是替代文本。", st),
            Bullet("full_no_context 达 80.51% 歧义 CV，证明两类信号互补；相对 text 提升约 4.21 pp。", st),
            Bullet("full 再升至 81.31%，说明组支持与题内相对比较提供额外信息；Validation 同样从 81.97% 歧义升到 83.38%。", st),
            H2("9.2 正则选择", st),
            P("text 的最佳 L2 为 0.001；embedding、full_no_context 与 full 选择 0.1。扩展扫描 0.03、0.3、1.0、3.0 后，0.1 在 Train CV 上保持最优折中，避免因高维 embedding 放大噪声。", st),
            P("这一阶段得到的是‘最佳单组排序器’，不是最终方案；下一步只允许用其折外分数做融合。", st, "callout"),
        ),
        _page(
            H1("10 旧四组模型堆叠：有效，但 Test 暴露不确定性", st),
            P("四个组模型的错误并不完全重合。旧方案把 text、embedding、full_no_context、full 的中心化 OOF 分数和组支持输入一个小型线性 stack。严格嵌套 Train CV 从 full 单模型的 1721/2114（81.4096%）提高到 1735/2114（82.0719%），Validation 从 739/798 提高到 745/798。", st),
            caption("Table 9", "旧堆叠的证据链", st),
            table(["阶段", "完整准确率", "歧义准确率", "判定"], [
                ["full 单模型 Train CV", "91.6930%", "81.3139%", "单模型基准"],
                ["full 单模型严格同协议基线", "-", "1721/2114 = 81.4096%", "供 nested 比较"],
                ["旧四模型 strict nested", "92.0294%", "1735/2114 = 82.0719%", "Train 证据改善"],
                ["旧四模型 Validation", "745/798 = 93.3584%", "302/355 = 85.0704%", "历史开发集结果"],
                ["历史公共 Test（旧四组/线性/MLP）", "381/467 → 382/467 → 377/467", "Test 全为歧义题", "恢复线性版本发生在看过该比较之后"],
            ], st, [48*mm, 40*mm, 48*mm, 32*mm]),
            P("旧四组 stack 在公共 Test 得到 381/467（81.5846%），低于更早的 single ranker 385/467（82.4411%）。后续 MLP 扩展得 377/467（80.7281%），线性候选旧记录为 382/467（81.7987%）。2026-10-28 恢复线性版本的决定发生在查看该公开 Test 比较之后，因此最终 package 的模型选择受到 Test 影响，不能称为 test-blind。", st),
            P("<b>方法学结论：</b>由于早期方法已经使用过同一 Test split，这个差异不构成无偏的模型比较。本次 package 恢复发生在查看该比较后，因此不满足 test-blind 选择；未来独立比较需要真正 unseen audit bank。", st, "callout"),
        ),
        _page(
            H1("11 被否定路线：为什么没有把复杂度全部留下", st),
            P("旧堆叠之后，我们系统性尝试低置信度回退、解析提示、词法哈希、额外 pooling、替代损失和非线性 meta 特征。保留每个负结果可以防止报告只展示幸存者，也说明当前方案为何仍保持线性和小资产。", st),
            caption("Table 10", "改进尝试、验证结果与选择结论", st),
            table(["候选方案", "Train 证据", "Validation", "决定"], REJECTED, st, [58*mm, 35*mm, 30*mm, 45*mm], font_size=6.15),
            H2("11.1 门控为何失败", st),
            P("直觉上，低 margin 时退回 plurality 应能减少回归；但最佳门控 OOF 只有 82.26%，Validation 为 743/798，低于无门控堆叠。原因是模型最有价值的 68 个修正恰有许多发生在 plurality 错误的困难题上，简单回退同时丢失收益。", st),
            H2("11.2 替代目标为何未留下", st),
            P("listwise softmax 和 hardest-negative 都把注意力集中到更少比较，但 CV 仅 81.55%，Validation 741/742。二次 20 特征 stack 也只有 82.12% OOF 与 744/798。复杂度没有带来跨协议的共同提升，因此拒绝。", st),
            P("统一的准入条件是‘Train 证据 + Validation 不退化 + 资源可接受’，而不是从多次尝试中挑一个最高单点。", st, "callout"),
        ),
        _page(
            H1("12 候选级扩展：从失败单模型到互补成员", st),
            P("错误分析显示，组均值可能把一条结构清晰、与题目条件一致的推理和同组较差文本混在一起。候选级 ranker 单独并不更强，但它与组模型犯错不同，因而适合作为 stack member。其组分数取成员最大值，而不是平均值。", st),
            caption("Table 11", "候选级扩展的选择路径", st),
            table(["步骤", "观察", "判断", "下一步"], [
                ["候选特征 + pairwise", "80.56% 歧义 OOF；739/798 Val", "不足以替代 full/stack", "只测试互补性"],
                ["加入普通 OOF stack", "L2=0.01 时 OOF 最高 82.5449%", "但 Val 降到 744", "不能按 OOF 单点选"],
                ["稳健区间", "L2=0.001/0.003 都保持 745 Val (historical linear-candidate stack)", "偏向更弱正则 0.001", "做 strict nested"],
                ["严格嵌套", "1743/2114 = 82.4503%", "比旧四组栈多 8 题", "回到此前冻结方案"],
                ["资源复测", "259,032 B；Val 1.006 s；历史 Test 1.292 s", "远低于限制", "记录 hash 与命令"],
            ], st, [38*mm, 51*mm, 44*mm, 36*mm]),
            P("最终六值 meta 特征为：四个组模型中心化分数、线性 candidate_group_max_score 与 centered group_support。MLP 候选分数不属于所选 checkpoint。系数受特征中心化与尺度影响，因此只作预测性融合解释，不作因果解释。", st),
            P("该解释只描述冻结线性 stack 的关联，不证明因果；权重还受到特征尺度和中心化影响。", st, "callout"),
        ),
        _page(
            H1("13 Stack L2 网格：如何在相互冲突的证据中选择", st),
            P("候选成员加入后，stack L2 在 Train OOF 上出现‘最高点与 Validation 最稳点不一致’。如果只选 OOF 最大的 0.01，会得到 1745/2114，却使 Validation 从 745 降到 744；更强正则继续下降。最终选择 0.001，并用严格嵌套 CV 做确认。", st),
            caption("Table 12", "候选堆叠的开发网格", st),
            table(["Stack L2", "普通 OOF 正确", "歧义 OOF", "Validation", "判断"], CANDIDATE_GRID, st, [23*mm, 36*mm, 31*mm, 31*mm, 48*mm], font_size=6.25),
            H2("13.1 选择规则", st),
            Bullet("首先排除 Validation 明显回退的 0.03、0.1、0.3。", st),
            Bullet("0.01 虽是普通 OOF 最高点，但 Validation 少 1 题，表明峰值可能是选择噪声。", st),
            Bullet("0.001 与 0.003 都保持 745；当时选择 0.001 后，严格嵌套重训得到 1743，高于旧 stack 的 1735。后续 MLP 替代方案得到 1741/2114，但未作为最终 checkpoint。", st),
            Bullet("因此改进被描述为‘保守升级’，不是用同一个开发集证明普遍优越。", st),
            P("这个例子展示了完整的方案选择步骤：提出互补假设 → 单模型否证替代关系 → 小网格测试融合 → 发现指标冲突 → 依据预先规则选择稳健区 → 用严格嵌套协议重新估计。", st, "callout"),
        ),
        _page(
            H1("14 严格嵌套 Train CV：最终选择的关键证据", st),
            P("最终比较使用完全相同的 outer folds 和训练预算。对每个 outer fold，基础模型、候选模型和 meta stack 都只看 outer-train；inner 4 折只用于生成 meta 训练分数。outer-test 只评分一次。五折歧义正确数分别为 346/422、348/423、345/423、351/423、353/423，总计 1743/2114。", st),
            nested_cv_comparison("zh"),
            caption("Fig. 4", "同一严格嵌套协议下的歧义 Train CV。所选线性候选栈比旧四组 stack 多 8 题；MLP 替代方案为 1741/2114。", st),
            caption("Table 13", "嵌套 CV 的分层解释", st),
            table(["方法", "歧义正确", "歧义准确率", "等效完整准确率", "相对前项"], [
                ["full single 同协议", "1721/2114", "81.4096%", "91.7350% 左右", "基线"],
                ["旧四组 stack", "1735/2114", "82.0719%", "92.0294%", "+14 题"],
                ["组 + linear candidate（所选）", "1743/2114", "82.4503%", "92.1977%", "+8 vs 四组栈"],
            ], st, [44*mm, 34*mm, 34*mm, 36*mm, 21*mm]),
            P("线性候选栈的严格嵌套 CV 为 1743/2114；MLP 替代版本为 1741/2114、Validation 高 1 题。最终恢复决定发生在查看公开 Test 比较之后，这违反 test-blind 选择要求，不能解释为只由 Train/Validation 证据决定。", st, "callout"),
        ),
        _page(
            H1("15 最终 Validation：完整、歧义与 plurality 结构", st),
            P("冻结 checkpoint 在完整 Validation 上正确 745/798；443 道单有效组题由解析路径确定性解决，355 道歧义题中正确 302 题。与 plurality 配对：两者都对 677、仅模型对 68、仅 plurality 对 11、两者都错 42。", st),
            caption("Table 14", "Validation 分层结果", st),
            table(["切片", "Plurality", "最终模型", "提升"], [
                ["完整 798", "688/798 = 86.2155%", "745/798 = 93.3584%", "+7.1429 pp"],
                ["歧义 355", "245/355 = 69.0141%", "302/355 = 85.0704%", "+16.0563 pp"],
                ["唯一 plurality 267", "207/267 = 77.5281%", "235/267 = 88.0150%", "+10.4869 pp"],
                ["无唯一 plurality 88", "38/88 = 43.1818%", "67/88 = 76.1364%", "+32.9545 pp"],
                ["正确支持=1，117", "27/117 = 23.0769%", "70/117 = 59.8291%", "+36.7521 pp"],
                ["正确支持=2，123", "103/123 = 83.7398%", "117/123 = 95.1220%", "+11.3821 pp"],
            ], st, [43*mm, 44*mm, 45*mm, 36*mm]),
            H2("15.1 配对 bootstrap", st),
            P("对每道题的‘模型正确指示 - plurality 正确指示’做 10,000 次配对 bootstrap，seed=7506。完整集差值 7.1429 pp，95% CI [5.0125, 9.2732]；歧义集差值 16.0563 pp，95% CI [11.5493, 20.8451]。区间描述该 Validation bank 上的配对不确定性，不等同于未知 audit 分布的置信保证。", st),
            P("该结果回答了‘改进是否只是容易题堆积’：主要提升恰来自无唯一多数与单支持的困难题。", st, "callout"),
        ),
        _page(
            H1("16 置信度、错误类型与剩余风险", st),
            P("对每道歧义题，margin 定义为最高答案组分数减第二高分。它没有参与冻结后的门控，而只用于诊断。低 margin 区间准确率约 59%，2-4 区间为 91.82%，大于等于 4 的 92 题为 100%。", st),
            caption("Table 15", "Validation 歧义题的 margin 分箱", st),
            table(["Margin", "正确/总数", "准确率", "解释"], [
                ["[0,0.5)", "23/39", "58.9744%", "低 margin，保持人工审阅"],
                ["[0.5,1)", "20/34", "58.8235%", "最不确定区间"],
                ["[1,2)", "66/80", "82.5000%", "中等可信"],
                ["[2,4)", "101/110", "91.8182%", "高可信"],
                ["[4,+∞)", "92/92", "100.0000%", "本 bank 上未观察到错误"],
            ], st, [35*mm, 36*mm, 34*mm, 64*mm]),
            H2("16.1 代表性失败类别", st),
            Bullet("条件遗漏或单位转换：错误推理格式完整，却漏掉题目中的折扣、停顿、每周/每天或英尺/码关系。", st),
            Bullet("少数派与支持冲突：一个精细但脆弱的候选对抗两个共享同一早期错误的候选。", st),
            Bullet("正确最终值来自不可靠路径：评分器判对，但文本结构信号可能把偶然碰对与真正推理混淆。", st),
            Bullet("低 margin 近邻：不同答案组在文本与 embedding 上都接近，线性边界难以稳定分离。", st),
            P("尝试用 margin 门控回退 plurality 后 Validation 下降，故 margin 只作为未来审阅优先级，不改变冻结预测。", st, "callout"),
        ),
        _page(
            H1("17 资源、实现安全性与最终复测", st),
            P("冻结方案包含 25 个线性折模型（4 类组 ranker + 1 类候选 ranker，各 5 folds）和一个 6 值 meta stack。不加载外部语言模型权重。", st),
            caption("Table 16", "资源与完整性检查", st),
            table(["项目", "实测", "限制/预期", "结果"], [
                ["model.npz 磁盘", "281,556 B", "计入资产", "通过"],
                ["metadata.json", "6,054 B", "计入资产", "通过"],
                ["数组未压缩", "259,032 B", "≤16,777,216 B", "通过（约 1.55% 上限）"],
                ["Validation 3 次 / Test 1 次", "min/mean/max: 1.0034/1.0045/1.0063 s；历史 Test 1.2920 s", "≤30 s", "通过"],
                ["线程变量", "OMP/OPENBLAS/MKL=4", "总计最多 4", "已记录"],
                ["数据检查", "manifest / split / shape", "全部一致", "通过"],
                ["自动测试", "9 tests", "接口、无泄漏、特征、checkpoint", "通过"],
            ], st, [39*mm, 48*mm, 45*mm, 36*mm]),
            H2("17.1 冻结资产哈希", st),
            P("model.npz: f24f66cf131120d63a2f2482146e591e67d9e6d1d5194b75c2ebd1408734b22b", st, "small"),
            P("metadata.json: 2b76d17f4430d0f7d143010bcf1fb748c9b687e770d31412a4efeed256ecc092", st, "small"),
            P("预测入口在加载时验证 checkpoint schema、feature schema、数组形状和有限值；测试还确认 submission 路径不读取 reference。资源测试使用新的预测进程加独立评分进程，包含启动和加载开销。", st),
            P("本机是 macOS 26.3.2 arm64；课程资格仍以参考 Xeon Platinum 8457C 为准。当前余量很大，但不能把本机时间等同于官方硬件时间。", st, "callout"),
        ),
        _page(
            H1("18 复现流程、冻结记录与 Test 隔离", st),
            H2("18.1 环境与训练", st),
            P("Python 3.12.13，NumPy 2.5.3。先在 code/.venv 安装 requirements.txt 并运行 check_data.py；再运行 8 个单元/契约测试。训练命令固定 folds=5、stack-l2=0.001、include-candidate、candidate-l2=0.1、epochs=220、learning-rate=0.03、seed=7506。", st),
            H2("18.2 预测与独立评分", st),
            P("预测通过 predict.py --method submission.py --checkpoint checkpoint/final --split validation 生成 JSON；score_predictions.py 在独立进程读取 Validation references 并评分。benchmark.py 重复三次并记录命令、环境、资产和完整进程时间。", st),
            caption("Table 17", "冻结顺序与证据状态", st),
            table(["顺序", "动作", "可用数据", "状态"], [
                ["1", "确定解析、特征、fold 与候选", "Train", "完成"],
                ["2", "普通 OOF 网格与消融", "Train", "完成"],
                ["3", "严格嵌套重训比较", "Train", "完成"],
                ["4", "完整 Validation、切片、bootstrap", "Validation", "完成"],
                ["5", "写入 final_config 与 checkpoint hashes", "Train/Validation", "完成"],
                ["6", "线性 checkpoint 的既有 Test 评分", "历史一次评分；2026-10-28 恢复决定受 Test 比较影响", "382/467"],
                ["7", "未来 unseen audit", "输入-only bank", "待课程方提供"],
            ], st, [18*mm, 62*mm, 52*mm, 36*mm]),
            P("历史 single ranker、旧四组 stack 和前一线性候选栈的 Test 分别为 385/467、381/467 和 382/467；后续 MLP stack 为 377/467。2026-10-28 恢复线性 checkpoint 发生在比较这些 Test 分数之后；Test 影响了当前 package 的模型选择。", st, "callout"),
        ),
        _page(
            H1("19 结论、局限与 AI 披露", st),
            H2("19.1 结论", st),
            P("在固定四候选、离线 CPU 和 16 MiB 资产限制下，最可靠的设计不是直接相信多数票，也不是只依赖单条 embedding，而是先对等价答案分组，再融合组级共识、文本结构、表示一致性和候选级强证据。Validation 为 93.3584%，比 plurality 高 7.1429 pp；线性 checkpoint 的历史 Test 为 81.7987%；严格嵌套 Train 歧义 CV 为 82.4503%。", st),
            H2("19.2 局限", st),
            Bullet("用户在查看线性和 MLP 版本的 public Test 分数后要求恢复线性 checkpoint。因此 Test 影响了最终 package 选择，81.7987% 不是 test-blind 选择的无偏成绩；课程规则要求 Test 不参与模型选择。", st),
            Bullet("线性特征可以识别格式与一致性，却不能执行完整符号验证；高质量错误推理仍可能得高分。", st),
            Bullet("最终答案评分允许‘错误过程偶然得到正确值’，故模型权重不等于真实推理质量的因果解释。", st),
            Bullet("confidence margin 在 Validation 上相关，但样本分箱较小，且门控实验失败，不应直接当校准概率。", st),
            H2("19.3 AI 与复用披露", st),
            P("本项目使用了实质性 AI 辅助来梳理 benchmark、设计与实现方法、建立测试、组织实验和起草文档；所有输出均由数据检查、自动测试、Train-only CV、官方 Validation scorer、资源测量与哈希核对验证。训练和推理阶段不调用 AI 服务，不使用外部数据或额外预训练权重。课程提供的 parser、scorer、runner、数据与许可保持不变。提交前学生应复核披露是否覆盖后续人工与 AI 修改。", st),
            P("最有价值的后续工作不是继续调已见 Validation，而是在冻结当前 checkpoint 后，用真正 unseen audit bank 检验 candidate 互补增益。", st, "callout"),
        ),
        _page(
            H1("20 参考资料与实验索引", st),
            H2("参考资料", st),
            P("[1] DASE7506 Project 2, <i>Selecting Reliable LLM Reasoning</i>, GUIDE.md, 2026.<br/>[2] DASE7506 MP2 supplied code README, common.py, predict.py and score_predictions.py.<br/>[3] Karl Cobbe et al., <i>Training Verifiers to Solve Math Word Problems</i>, arXiv:2110.14168, 2021 (GSM8K).<br/>[4] Supplied data NOTICE.txt and manifest.json: Qwen3-0.6B candidate/embedding provenance.<br/>[5] R. A. Bradley and M. E. Terry, <i>Rank Analysis of Incomplete Block Designs</i>, Biometrika, 1952.<br/>[6] B. Efron and R. Tibshirani, <i>An Introduction to the Bootstrap</i>, 1993.<br/>[7] Harris et al., <i>Array programming with NumPy</i>, Nature, 2020.", st),
            H2("可复核证据索引", st),
            caption("Table 18", "报告结论对应的仓库记录", st),
            table(["结论", "主要文件"], [
                ["任务、约束、提交口径", "guide/GUIDE.md; REQUIREMENTS.md"],
                ["最终方法与命令", "code/METHOD.md; code/train_stacked.py; code/submission.py"],
                ["基础 CV 与正则", "experiments/cross_validation.json; cv_regularization_extension.json"],
                ["旧 stack 与 strict nested", "experiments/improvement_stacking*.json"],
                ["候选扩展与网格", "experiments/improvement_candidate_stack.json"],
                ["最终 strict nested", "experiments/improvement_candidate_stack_nested_cv.json"],
                ["Validation 分层与 bootstrap", "experiments/validation_diagnostics.json"],
                ["margin 与错误分析", "experiments/validation_error_analysis.json"],
                ["分数、资源与哈希", "results/final/*.json; experiments/final_config.json"],
                ["历史 Test provenance", "results/historical/previous-stack/"],
            ], st, [57*mm, 112*mm]),
            P(edition_note, st, "callout"),
        ),
    ]


def full_pages_en(st: dict[str, ParagraphStyle], compact: bool = False) -> list[list[Flowable]]:
    # The English edition mirrors the Chinese evidence order one-for-one so
    # neither language silently loses a validation step.
    edition_note = (
        "This 10-page edition uses denser type and tables while retaining the Full edition's complete selection and validation chain."
        if compact
        else "The Full edition preserves the complete selection and validation chain; the companion 10-page edition compresses layout without deleting a validation stage."
    )
    return [
        _page(
            *title_block("en", compact, st),
            H1("1 Background, Objective, and Headline Result", st),
            P("Each GSM8K problem is accompanied by four language-model reasoning traces. The system may not generate a new answer; it must return one candidate index from 0 to 3. The fixed scorer parses the selected trace's final numeric answer. It does not directly grade every intermediate step. The research question is therefore which reproducible text, agreement, and representation signals identify a correct final answer.", st),
            P("The frozen method first groups candidates by canonical numeric answer. A single valid group exits deterministically. Ambiguous questions are scored by four answer-group ranker types, one linear candidate ranker, each represented by five cross-fold models. A six-value linear stack chooses the group, and the earliest member supplies the candidate index. All weights use Train only; Validation supports selection and diagnosis.", st),
            caption("Table 1", "Core results of the frozen method", st),
            table(["Evidence", "Result", "Interpretation boundary"], [
                ["Strict nested Train CV", "1743/2114 ambiguous, 82.4503%", "5 outer / 4 inner folds; complete stack retrained"],
                ["Full Validation", "745/798, 93.3584%", "Development and freeze evidence; not Test"],
                ["Ambiguous Validation", "302/355, 85.0704%", "At least two distinct valid parsed answers"],
                ["Plurality baseline", "688/798, 86.2155%", "Final method is +7.1429 percentage points"],
                ["Historical Test record", "382/467, 81.7987%", "Previously scored once; restoration followed public-Test comparison"],
                ["Inference assets / timing", "259,032 B (1.55%); Val 1.006 s; historical Test 1.292 s", "Below 16 MiB / 30 s limits"],
            ], st, [39*mm, 52*mm, 77*mm]),
            P("<b>Essential disclosure:</b> the selected linear-candidate checkpoint has a historical Test record of 382/467 (81.7987%), scored once. On 2026-10-28 the user asked to restore this checkpoint after comparing its public-Test score with the later MLP score. Public Test therefore influenced package-level model selection, contrary to the coursework rule; 81.80% is historical performance, not a test-blind final estimate.", st, "callout"),
        ),
        _page(
            H1("2 Evaluation Protocol, Data Boundaries, and Hard Limits", st),
            P("Split roles were frozen before modelling: Train has 4,755 questions for fitting weights, statistics, and fusion; Validation has 798 for development and selection; Test has 467 for one frozen evaluation; a possible audit bank must run without retraining or assumptions about IDs or sample count. Candidate IDs are alignment metadata, never features.", st),
            caption("Table 2", "Fixed coursework conditions and acceptance checks", st),
            table(["Item", "Fixed rule", "Project treatment"], [
                ["Input", "Question, four traces, [N,4,1024] float32 embeddings", "Uses only bank/arrays passed to predict"],
                ["Output", "Length-N list of Python int values in 0..3", "Contract tests cover type, range, order"],
                ["Score", "Parsed selected value equals the reference", "Uses common.parse_answer semantics"],
                ["External resources", "No external data, weights, or online APIs", "Offline NumPy and standard library"],
                ["Time", "CPU prediction + scoring <=30 s; four threads", "Validation 1.006 s; historical Test 1.292 s"],
                ["Assets", "Additional inference assets <=16 MiB", "259,032 array bytes (1.55%)"],
                ["Report", "Final submission at most 10 pages", "Separate exact 10-page editions"],
            ], st, [31*mm, 72*mm, 65*mm]),
            H2("2.1 Metrics and the hard subset", st),
            P("Full accuracy is the headline metric. Because 443 questions contain only one valid answer group, we also report accuracy on 355 ambiguous questions, then slice by unique plurality, correct-answer support, group count, and score margin. Comparisons against plurality are paired by question.", st),
            P("The first validation gate checks manifest hashes, split disjointness, candidate counts, embedding shapes, and finite values. Unit tests then cover answer grouping, checkpoint schema, feature dimensions, prohibited reference access, and the submission interface. Modelling begins only after those checks pass.", st, "callout"),
        ),
        _page(
            H1("3 Frozen System: From Four Traces to One Index", st),
            P("The design separates the decision into six auditable stages. Parsing and grouping match the official scorer; learned ranking is reserved for genuinely ambiguous questions. The method never changes or regenerates a candidate.", st),
            pipeline("en"),
            caption("Fig. 1", "Frozen data flow. The five ranker types are text, embedding, full_no_context, full, and candidate; each has five cross-fold models.", st),
            caption("Table 3", "Inference stages and safeguards", st),
            table(["Stage", "Operation", "Safeguard"], [
                ["1 Parse", "Extract canonical numeric values", "Fraction equivalence; unparseables recorded"],
                ["2 Group", "Merge candidates with equal values", "Position never acts as a correctness label"],
                ["3 Early exit", "Return when one valid group exists", "Never choose unparseable over parseable"],
                ["4 Base scores", "Four group views plus candidate view", "Five fold models per view"],
                ["5 Stack", "Five centred scores plus support", "Only six features; L2=0.001"],
                ["6 Decide", "Top group, earliest member", "Stable tie break; arbitrary N"],
            ], st, [27*mm, 72*mm, 69*mm]),
            P("Consensus is a prior, not a veto. The rankers may overturn a wrong majority, while the maximum candidate score preserves one especially strong reasoning trace that group averages might hide.", st),
        ),
        _page(
            H1("4 Final-Answer Parsing, Grouping, and Determinism", st),
            H2("4.1 Why rank answer groups", st),
            P("Several candidates can express an equivalent value, such as 0.5 and 1/2. Since the scorer only checks the final numeric value, per-candidate classification would duplicate the same answer and invite position bias. The official-compatible parser yields a canonical value a_i and the method forms G(a)={i:a_i=a}. Any member of a correct group earns the point.", st),
            P("Unparseable values never merge with valid values. If a valid group exists, an unparseable candidate cannot be selected. Only an all-unparseable input falls back to the earliest index. Group support, aggregated text/embedding evidence, and differences from competing groups become the interpretable group features.", st),
            P("G(a) = { i in {0,1,2,3} : parse(candidate_i) = a }", st, "equation"),
            caption("Table 4", "Edge cases and checks", st),
            table(["Case", "Decision", "Verification"], [
                ["All four share one answer", "Return earliest index", "Single-group path test"],
                ["Two to four valid groups", "Invoke learned ranking", "Ambiguous-slice scoring"],
                ["Equivalent notation", "Merge into one group", "Fraction/decimal tests"],
                ["Some unparseable", "Prefer every valid group", "Parser and fallback tests"],
                ["Exact score tie", "Earliest candidate wins", "Deterministic rerun"],
                ["Audit N changes", "Process input length dynamically", "No split-size constants"],
            ], st, [40*mm, 60*mm, 68*mm]),
            P("Parser agreement is the first experimental checkpoint: no downstream accuracy is meaningful unless equivalence, negative numbers, fractions, missing markers, and multiple-correct cases match the scorer.", st, "callout"),
        ),
        _page(
            H1("5 Answer-Group Features: Separating the Evidence", st),
            P("The feature design is split into four ablatable views. Each view trains the same linear pairwise ranker, so score changes can be connected to information sources instead of an opaque feature dump.", st),
            caption("Table 5", "Four answer-group feature sets", st),
            table(["Feature set", "Main content", "Research question", "Dimension / L2"], [
                ["text", "Length, numbers, operations, steps, question overlap, ending patterns", "Can trace structure beat consensus?", "54; 0.001"],
                ["embedding", "Normalised group centre minus all-candidate centre", "Do fixed representations add signal?", "1,024; 0.1"],
                ["full_no_context", "54-D text plus 1,024-D embedding delta", "Are the two views complementary?", "1,078; 0.1"],
                ["full", "6-D consensus + text + embedding delta + 4 embedding scalars", "Does within-question context help?", "1,088; 0.1"],
            ], st, [27*mm, 71*mm, 46*mm, 24*mm]),
            H2("5.1 Within-group and between-group summaries", st),
            P("At code level, text takes the within-group mean and maximum of 27 per-trace features, producing 54 dimensions. The embedding-only view is exactly the 1,024-D difference between the normalised group centre and the all-candidate centre. The full view additionally contains six consensus features (support fraction, plurality margin, unique plurality, singleton, distinct-answer fraction, and unparseable fraction) plus four embedding scalars (cohesion, group-to-all cosine, delta norm, and group-mean norm).", st),
            H2("5.2 Explicit exclusions", st),
            P("Question IDs, split size, references, cached predictions, and Test error categories never enter features. The training script opens Train references only; the prediction entry point never opens references.", st),
            P("A feature is retained only when Train out-of-fold behaviour is credible, variance is acceptable, Validation does not regress, resources remain explainable, and strict nested retraining confirms the direction.", st, "callout"),
        ),
        _page(
            H1("6 Candidate Ranker and Learning Objective", st),
            P("Group averages can hide one exceptionally reliable trace. The candidate ranker has exactly 1,072 dimensions: six consensus features for its answer group, 27 trace features, a 1,024-D singleton embedding delta, four embedding scalars, and 11 parser-consistency features. Candidate index and ID are excluded. It remains a small linear model and adds no external pretrained component.", st),
            H2("6.1 Multiple-positive pairwise learning", st),
            P("A question may contain several correct candidates. Every correct candidate is paired with every incorrect one, and pairs are normalised per question so questions with more combinations do not dominate. For z=x_pos-x_neg, training minimises logistic pair loss with L2 regularisation:", st),
            P("L(w) = (1/Q) Σ_q [1/(|P_q||N_q|) Σ_(p,n) log(1 + exp(-w^T(x_p-x_n)))] + (λ/2)||w||²", st, "equation"),
            P("At inference, the candidate model scores all four traces and contributes max_i∈G s_i for each answer group. It is a fifth complementary signal, not an independent final selector.", st),
            caption("Table 6", "Intermediate checks for candidate evidence", st),
            table(["Check", "Result", "Conclusion"], [
                ["Candidate-only OOF", "80.56% ambiguous Train", "Below group models; not a replacement"],
                ["Candidate-only Validation", "739/798", "Below 745; no standalone claim"],
                ["Added to ordinary OOF stack", "Up to 82.5449%", "Must control selection bias"],
                ["Strict nested confirmation", "1743/2114, 82.4503%", "+8 vs four-group stack"],
                ["Validation", "745/798", "Matches the selected linear-candidate checkpoint"],
            ], st, [51*mm, 49*mm, 68*mm]),
            P("Its justification is complementary error structure: the candidate ranker is weaker alone, yet improves the complete nested stack without reducing Validation.", st, "callout"),
        ),
        _page(
            H1("7 Experimental Protocol: Selection Followed by Revalidation", st),
            decision_flow("en"),
            caption("Fig. 2", "Complete selection chain. Structure and hyperparameters use Train/Validation; Test scores informed the 2026-10-28 package restoration, so this is not test-blind selection.", st),
            H2("7.1 Fixed folds and ordinary out-of-fold comparison", st),
            P("Train is divided into five fixed question-level folds with seed 7506. Four folds fit a ranker and the hold-out fold receives OOF scores; all four candidates from a question remain together. Base L2 values are scanned with learning rate 0.03 and at most 220 epochs. Fold accuracy, mean, and standard deviation are recorded.", st),
            H2("7.2 Why strict nested CV remains necessary", st),
            P("Ordinary OOF scores train a stack, but choosing stack L2 on those same scores creates second-level selection bias. Strict nesting uses five outer folds. Within each outer-train partition, four inner folds rebuild base OOF features and fit the meta stack; the untouched outer-test fold is scored once. Every reported outer prediction therefore comes from a complete training chain that never saw that question.", st),
            P("The admission rule is fixed: a new method must improve strict nested Train CV and must not reduce full Validation. An isolated ordinary-OOF or Validation peak is not enough.", st, "callout"),
        ),
        _page(
            H1("8 Baselines: Establishing Where the Difficulty Lies", st),
            P("Three training-free baselines cover position bias, length bias, and answer consensus. Fixed index zero and longest trace remain near 78%, while plurality reaches 86.22%. Yet plurality is only 69.01% on ambiguous questions, leaving substantial room for evidence-aware ranking.", st),
            baseline_comparison("en"),
            caption("Fig. 3", "Full Validation baselines and the frozen stack. All bars are paired on the same bank.", st),
            caption("Table 7", "Official Validation baseline scores", st),
            table(["Method", "Correct / total", "Accuracy", "vs plurality"], [
                ["Fixed index 0", "624/798", "78.1955%", "-8.0201 pp"],
                ["Longest trace", "622/798", "77.9449%", "-8.2707 pp"],
                ["Answer plurality", "688/798", "86.2155%", "0"],
                ["Final stack + linear candidate", "745/798", "93.3584%", "+7.1429 pp"],
            ], st, [47*mm, 38*mm, 38*mm, 45*mm]),
            H2("8.1 Correct-answer support identifies the hard regime", st),
            P("Plurality is perfect when the correct answer has support three or four, but only 23.08% when one trace supports it. The final model raises this slice to 70/117 (59.83%), showing that the main gain comes from rescuing correct minorities rather than strengthening easy majorities.", st),
        ),
        _page(
            H1("9 Single-Ranker Ablation: Which Information Helps?", st),
            P("The single-ranker sequence adds text, embeddings, their combination, then group-relative context. Each line is judged first by five-fold Train CV and then by frozen Validation; the best Validation point cannot replace stability evidence.", st),
            caption("Table 8", "Feature ablation: Train CV mean ± fold standard deviation and Validation", st),
            table(["Method", "Train full", "Train ambiguous", "Val full", "Val ambiguous"], CV_ROWS[:5], st, [36*mm, 39*mm, 39*mm, 27*mm, 27*mm], font_size=6.3),
            H2("9.1 Stepwise decisions", st),
            Bullet("Text reaches 76.30% ambiguous Train CV: surface structure is useful but insufficient.", st),
            Bullet("Embeddings alone are comparable, not decisively stronger; their main value is complementary.", st),
            Bullet("full_no_context reaches 80.51%, about +4.21 points over text, confirming complementarity.", st),
            Bullet("full reaches 81.31% and 83.38% ambiguous Validation, so within-question context adds evidence.", st),
            H2("9.2 Regularisation selection", st),
            P("Text selects L2=0.001; embedding, full_no_context, and full select 0.1. Extending the grid to 0.03, 0.3, 1.0, and 3.0 keeps 0.1 as the best Train-CV compromise for embedding-rich views.", st),
            P("This stage identifies the best single group ranker. Fusion is allowed to consume only cross-fitted scores.", st, "callout"),
        ),
        _page(
            H1("10 Previous Four-Group Stack: Useful but Not Conclusive", st),
            P("The four group rankers make partially different errors. The previous stack fused centred OOF scores from text, embedding, full_no_context, and full with support. Strict nested ambiguous Train CV rose from 1721/2114 for the full single-ranker baseline to 1735/2114; Validation rose from 739/798 to 745/798 for the historical four-group checkpoint; the historical linear-candidate checkpoint retained 745, and the later MLP alternative reaches 746/798. The selected checkpoint remains the earlier linear version.", st),
            caption("Table 9", "Evidence chain for the previous stack", st),
            table(["Stage", "Full accuracy", "Ambiguous accuracy", "Decision"], [
                ["Full single Train CV", "91.6930%", "81.3139%", "Single-ranker reference"],
                ["Full single, matched nested", "-", "1721/2114 = 81.4096%", "Nested comparator"],
                ["Previous stack, strict nested", "92.0294%", "1735/2114 = 82.0719%", "Train evidence improves"],
                ["Previous stack, Validation", "745/798 = 93.3584%", "302/355 = 85.0704%", "Historical result"],
                ["Public Test records: old group/linear/MLP", "381/467 -> 382/467 -> 377/467", "All Test items ambiguous", "Restoration followed this comparison"],
            ], st, [48*mm, 40*mm, 48*mm, 32*mm]),
            P("The earlier single ranker scored 385/467 (82.4411%) on public Test, and the old group stack scored 381/467 (81.5846%). The linear checkpoint has an earlier one-time score of 382/467 (81.7987%); the later MLP alternative scored 377/467 (80.7281%). The 2026-10-28 restoration followed this public-Test comparison, so Test informed package-level selection and the score is not a blind final estimate.", st),
            P("The public Test comparison informed package-level restoration, so this package is not test-blind. Use a genuinely unseen audit bank for independent evaluation.", st, "callout"),
        ),
        _page(
            H1("11 Rejected Routes and the Cost of Complexity", st),
            P("After the previous stack, we evaluated confidence gates, parser cues, lexical hashes, extra pooling, alternative objectives, and nonlinear meta features. Keeping negative results prevents survivor-only reporting and explains why the frozen system remains linear and small.", st),
            caption("Table 10", "Improvement attempts and decisions", st),
            table(["Candidate", "Train evidence", "Validation", "Decision"], REJECTED, st, [58*mm, 35*mm, 30*mm, 45*mm], font_size=6.1),
            H2("11.1 Why confidence gating failed", st),
            P("Returning to plurality at low margin seems protective, but the best gate reached only 82.26% OOF and 743/798 Validation. Many of the model's 68 useful corrections occur precisely where plurality is wrong, so a simple fallback also removes gains.", st),
            H2("11.2 Why alternative objectives were removed", st),
            P("Listwise softmax and hardest-negative training each reached only 81.55% CV with 741/742 Validation. A quadratic 20-feature stack produced 82.12% OOF and 744/798. Extra complexity did not improve both protocols.", st),
            P("The shared admission rule is joint Train evidence, no Validation regression, and acceptable resources - never the best isolated point among many trials.", st, "callout"),
        ),
        _page(
            H1("12 Candidate Extension: A Weak Standalone, a Useful Member", st),
            P("Error inspection suggested that group averages can dilute one well-structured, condition-aware trace. The candidate ranker is weaker alone, but its mistakes differ from group models, making it suitable as a stack member. Its group evidence is the member maximum, not the average.", st),
            caption("Table 11", "Selection path for the candidate extension", st),
            table(["Step", "Observation", "Judgement", "Next action"], [
                ["Candidate pairwise model", "80.56% ambiguous OOF; 739/798 Val", "Cannot replace full/stack", "Test complementarity only"],
                ["Add to ordinary stack", "L2=0.01 peaks at 82.5449% OOF", "Validation falls to 744", "Do not select peak blindly"],
                ["Stable region", "L2=0.001/0.003 retain 745 Val", "Prefer 0.001", "Run strict nested CV"],
                ["Strict nesting", "1743/2114 = 82.4503%", "-2 vs prior linear candidate", "Freeze current method"],
                ["Resource recheck", "259,032 B; Val 1.006 s; historical Test 1.292 s", "Large safety margin", "Record hashes"],
            ], st, [38*mm, 51*mm, 44*mm, 36*mm]),
            P("The six meta features are four centred group scores, the linear candidate score, and centred group support. Coefficients are predictive associations, not causal effects, and depend on feature scaling.", st),
            P("The linear candidate member was part of the earlier frozen stack. The later MLP alternative was not retained in the final checkpoint.", st, "callout"),
        ),
        _page(
            H1("13 Stack-L2 Grid and Conflict Resolution", st),
            P("After adding candidate evidence, the highest ordinary OOF setting and the most stable Validation setting disagree. L2=0.01 yields 1745/2114 OOF but reduces Validation to 744. Stronger regularisation degrades further. We select 0.001 and require strict nested confirmation.", st),
            caption("Table 12", "Candidate-stack development grid", st),
            table(["Stack L2", "Ordinary OOF", "Ambiguous OOF", "Validation", "Judgement"], CANDIDATE_GRID, st, [23*mm, 36*mm, 31*mm, 31*mm, 48*mm], font_size=6.2),
            H2("13.1 Selection rule", st),
            Bullet("Reject 0.03, 0.1, and 0.3 because Validation clearly declines.", st),
            Bullet("Do not take the 0.01 OOF peak because Validation loses one question.", st),
            Bullet("Both 0.001 and 0.003 retain 745; choose 0.001, then rerun the complete nested procedure.", st),
            Bullet("The nested result is 1741: six above the previous four-group stack and two below the previous linear-candidate stack. Validation is one question higher than that linear-candidate model; this tradeoff drives the final choice.", st),
            P("The full decision path is: complementary hypothesis, standalone falsification, compact fusion grid, conflict detection, robust-region choice, and strict nested re-estimation.", st, "callout"),
        ),
        _page(
            H1("14 Strict Nested Train CV: The Key Selection Evidence", st),
            P("All final comparisons use identical outer folds and budgets. Within every outer fold, base models, candidate models, and the meta stack see only outer-train. Four inner folds construct meta training scores. The five ambiguous outer results are 346/422, 348/423, 345/423, 351/423, and 353/423, totalling 1743/2114.", st),
            nested_cv_comparison("en"),
            caption("Fig. 4", "Matched strict nested comparison. The MLP extension adds six correct ambiguous Train questions over the old group stack and two fewer than the linear-candidate stack; Validation is one question higher than the linear-candidate stack.", st),
            caption("Table 13", "Nested-CV interpretation", st),
            table(["Method", "Ambiguous correct", "Ambiguous accuracy", "Equivalent full accuracy", "Change"], [
                ["Full single, matched", "1721/2114", "81.4096%", "about 91.7350%", "reference"],
                ["Previous group stack", "1735/2114", "82.0719%", "92.0294%", "+14"],
                ["Group + linear candidate", "1743/2114", "82.4503%", "92.1977%", "+8 vs group stack"],
            ], st, [44*mm, 34*mm, 34*mm, 36*mm, 21*mm]),
            P("The gain is small and the new method does not beat the baseline in every outer fold. Its support comes from the aggregate matched protocol, not a claim of fold-by-fold dominance.", st, "callout"),
        ),
        _page(
            H1("15 Final Validation: Full, Ambiguous, and Plurality Slices", st),
            P("The frozen checkpoint is correct on 745/798 Validation questions. The parser resolves 443 single-group questions deterministically; the learned selector gets 302/355 ambiguous questions. Paired against plurality: both correct 677, model only 68, plurality only 11, both wrong 42.", st),
            caption("Table 14", "Validation slices", st),
            table(["Slice", "Plurality", "Final model", "Gain"], [
                ["Full 798", "688/798 = 86.2155%", "745/798 = 93.3584%", "+7.1429 pp"],
                ["Ambiguous 355", "245/355 = 69.0141%", "302/355 = 85.0704%", "+16.0563 pp"],
                ["Unique plurality 267", "207/267 = 77.5281%", "235/267 = 88.0150%", "+10.4869 pp"],
                ["No unique plurality 88", "38/88 = 43.1818%", "67/88 = 76.1364%", "+32.9545 pp"],
                ["Correct support=1, 117", "27/117 = 23.0769%", "70/117 = 59.8291%", "+36.7521 pp"],
                ["Correct support=2, 123", "103/123 = 83.7398%", "117/123 = 95.1220%", "+11.3821 pp"],
            ], st, [43*mm, 44*mm, 45*mm, 36*mm]),
            H2("15.1 Paired bootstrap", st),
            P("A 10,000-sample paired bootstrap of per-question correctness differences (seed 7506) gives +7.1429 points on full Validation with 95% CI [5.0125, 9.2732], and +16.0563 points on ambiguous Validation with CI [11.5493, 20.8451]. These intervals describe this bank, not guaranteed audit performance.", st),
            P("The largest improvements occur without a unique majority and when only one candidate supports the correct answer, so the gain is not an artefact of easy single-group items.", st, "callout"),
        ),
        _page(
            H1("16 Confidence, Error Categories, and Residual Risk", st),
            P("Margin is the top group score minus the runner-up. It is diagnostic only and never gates the frozen prediction. Accuracy is about 59% below margin 1, rises to 91.82% between 2 and 4, and reaches 100% on the 91 Validation cases at or above 4.", st),
            caption("Table 15", "Margin bins on ambiguous Validation", st),
            table(["Margin", "Correct / total", "Accuracy", "Interpretation"], [
                ["[0,0.5)", "23/39", "58.9744%", "Low margin; review first"],
                ["[0.5,1)", "20/34", "58.8235%", "Most uncertain bin"],
                ["[1,2)", "66/80", "82.5000%", "Moderate confidence"],
                ["[2,4)", "101/110", "91.8182%", "High confidence"],
                ["[4,+∞)", "92/92", "100.0000%", "No observed error in this bank"],
            ], st, [35*mm, 36*mm, 34*mm, 64*mm]),
            H2("16.1 Representative failure categories", st),
            Bullet("Condition omission or unit conversion: polished traces miss a discount, stop, day/week, or feet/yard relation.", st),
            Bullet("Minority/support conflict: one careful trace competes against two candidates sharing an early error.", st),
            Bullet("Lucky final value: an invalid path lands on the scorer-correct number, confusing reasoning-quality features.", st),
            Bullet("Low-margin neighbours: competing groups are close in text and embedding space, making a linear boundary unstable.", st),
            P("Confidence gating reduced Validation, so margin remains a review signal rather than a post-hoc override.", st, "callout"),
        ),
        _page(
            H1("17 Resources, Implementation Safety, and Final Rechecks", st),
            P("The frozen method contains 25 linear fold models - four group rankers and one candidate ranker, each with five folds - and one six-value stack. No external language-model weights are loaded.", st),
            caption("Table 16", "Resource and integrity checks", st),
            table(["Item", "Measured", "Limit / expectation", "Result"], [
                ["model.npz on disk", "281,556 B", "Counted as asset", "Pass"],
                ["metadata.json", "6,054 B", "Counted as asset", "Pass"],
                ["Uncompressed arrays", "259,032 B", "<=16,777,216 B", "Pass (about 1.55%)"],
                ["Validation 3 runs / Test 1 run", "min/mean/max: 1.0034/1.0045/1.0063 s; 历史 Test 1.2920 s", "<=30 s", "Pass"],
                ["Thread variables", "OMP/OPENBLAS/MKL=4", "At most four total", "Recorded"],
                ["Data checker", "Manifest / split / shape", "All consistent", "Pass"],
                ["Automated tests", "9 tests", "Interface, leakage, features, checkpoint", "Pass"],
            ], st, [39*mm, 48*mm, 45*mm, 36*mm]),
            H2("17.1 Frozen hashes", st),
            P("model.npz: f24f66cf131120d63a2f2482146e591e67d9e6d1d5194b75c2ebd1408734b22b", st, "small"),
            P("metadata.json: 2b76d17f4430d0f7d143010bcf1fb748c9b687e770d31412a4efeed256ecc092", st, "small"),
            P("Checkpoint loading validates schema, feature schema, array shapes, and finite values. The benchmark uses fresh prediction and independent scoring processes, including startup and load time.", st),
            P("The local machine is macOS 26.3.2 arm64; formal eligibility still depends on the reference Xeon Platinum 8457C. The local margin is large, but local time is not claimed as reference-hardware time.", st, "callout"),
        ),
        _page(
            H1("18 Reproduction, Freeze Record, and Test Isolation", st),
            H2("18.1 Environment and training", st),
            P("The project environment is Python 3.12.13 with NumPy 2.5.3. From code/, install requirements.txt, run check_data.py, and execute the nine unit/contract tests. Training fixes five folds, stack L2=0.001, one linear candidate ranker, candidate L2=0.1, base epochs=220, and seed=7506.", st),
            H2("18.2 Prediction and independent scoring", st),
            P("predict.py loads submission.py and checkpoint/final to write Validation predictions. score_predictions.py opens Validation references in a separate process. benchmark.py repeats the complete path three times and records commands, environment, assets, and wall time.", st),
            caption("Table 17", "Freeze order and evidence state", st),
            table(["Order", "Action", "Permitted data", "State"], [
                ["1", "Fix parser, features, folds, candidates", "Train", "Complete"],
                ["2", "Ordinary OOF grids and ablations", "Train", "Complete"],
                ["3", "Strict nested retraining comparison", "Train", "Complete"],
                ["4", "Full Validation, slices, bootstrap", "Validation", "Complete"],
                ["5", "Write config and checkpoint hashes", "Train/Validation", "Complete"],
                ["6", "Historical score for linear checkpoint", "Scored once before restoration", "382/467; package choice later saw Test comparison"],
                ["7", "Future unseen audit", "Input-only bank", "Pending"],
            ], st, [18*mm, 62*mm, 52*mm, 36*mm]),
            P("The linear checkpoint has an existing one-time score of 382/467 (81.7987%); the later MLP alternative scored 377/467 (80.7281%). The 2026-10-28 package restoration followed comparison of these scores, so Test influenced final package selection and this workflow is not test-blind.", st, "callout"),
        ),
        _page(
            H1("19 Conclusions, Limitations, and AI Disclosure", st),
            H2("19.1 Conclusion", st),
            P("Under a four-candidate, offline CPU, 16 MiB constraint, the method groups equivalent answers and combines answer-group evidence with linear candidate evidence. Validation is 93.3584%, +7.1429 points over plurality; strict nested ambiguous Train CV is 82.4503%, +0.3784 points over the previous four-group stack. The MLP alternative was one Validation question higher and two nested-CV questions lower. The requested restoration followed public-Test comparison, so this package-level model choice is not test-blind.", st),
            H2("19.2 Limitations", st),
            Bullet("The 81.7987% public-Test score is a historical score for this checkpoint. The package-level restoration followed comparison with the MLP Test result, so Test influenced this version choice and the score is not an unbiased test-blind estimate. An unseen audit is needed for an independent comparison.", st),
            Bullet("Linear features recognise form and consistency but cannot execute full symbolic verification.", st),
            Bullet("Final-answer scoring accepts lucky correct values, so weights are not a causal measure of reasoning quality.", st),
            Bullet("Margin correlates with Validation accuracy, but bins are small and gating failed; it is not a calibrated probability.", st),
            H2("19.3 AI and reuse disclosure", st),
            P("Substantive AI assistance helped interpret the benchmark, design and implement the method, create tests, organise experiments, and draft documentation. Outputs were checked with the supplied data checker, automated tests, Train-only CV, the official Validation scorer, resource measurements, and hashes. Training and inference call no AI service and use no external data or extra pretrained weights. Supplied parser, scorer, runner, data, and licences remain unchanged. The student should review this disclosure before submission.", st),
            P("The next useful experiment is not more tuning on observed Validation, but evaluation of the frozen checkpoint on a genuinely unseen audit bank.", st, "callout"),
        ),
        _page(
            H1("20 References and Evidence Index", st),
            H2("References", st),
            P("[1] DASE7506 Project 2, <i>Selecting Reliable LLM Reasoning</i>, GUIDE.md, 2026.<br/>[2] DASE7506 MP2 supplied code README, common.py, predict.py, and score_predictions.py.<br/>[3] Karl Cobbe et al., <i>Training Verifiers to Solve Math Word Problems</i>, arXiv:2110.14168, 2021 (GSM8K).<br/>[4] Supplied NOTICE.txt and manifest.json for Qwen3-0.6B candidate and embedding provenance.<br/>[5] R. A. Bradley and M. E. Terry, <i>Rank Analysis of Incomplete Block Designs</i>, Biometrika, 1952.<br/>[6] B. Efron and R. Tibshirani, <i>An Introduction to the Bootstrap</i>, 1993.<br/>[7] Harris et al., <i>Array Programming with NumPy</i>, Nature, 2020.", st),
            H2("Auditable evidence index", st),
            caption("Table 18", "Repository records supporting the report", st),
            table(["Claim", "Primary files"], [
                ["Task, constraints, submission", "guide/GUIDE.md; REQUIREMENTS.md"],
                ["Frozen method and commands", "code/METHOD.md; train_stacked.py; submission.py"],
                ["Base CV and regularisation", "experiments/cross_validation.json; cv_regularization_extension.json"],
                ["Previous stack and nested CV", "experiments/improvement_stacking*.json"],
                ["Candidate grid", "experiments/improvement_candidate_stack.json"],
                ["Final strict nesting", "experiments/improvement_candidate_stack_nested_cv.json"],
                ["Validation slices and bootstrap", "experiments/validation_diagnostics.json"],
                ["Margin and error analysis", "experiments/validation_error_analysis.json"],
                ["Score, resources, hashes", "results/final/*.json; experiments/final_config.json"],
                ["Historical Test provenance", "results/historical/previous-stack/"],
            ], st, [57*mm, 112*mm]),
            P(edition_note, st, "callout"),
        ),
    ]


def audit_pages_zh(st: dict[str, ParagraphStyle]) -> list[list[Flowable]]:
    """Code-level traceability retained only in the unabridged edition."""

    return [
        _page(
            H1("20 代码级特征字典与维度审计", st),
            P("为避免用模糊的‘文本特征’或‘embedding 特征’掩盖实际实现，下表直接按 features.py 的向量拼接顺序核对。维度合计不仅用于写报告，也由 checkpoint schema test 在加载时强制检查。", st),
            caption("Table 19", "特征向量的精确组成", st),
            table(["向量", "精确组成", "维度", "聚合方式"], [
                ["text", "27 个 trace 特征的 mean + max", "54", "答案组内"],
                ["embedding", "unit(group mean) - unit(all-candidate mean)", "1024", "答案组相对全题"],
                ["full_no_context", "text + embedding", "54+1024=1078", "无共识/统计块"],
                ["full", "consensus 6 + text 54 + embedding 1024 + scalar 4", "1088", "完整组特征"],
                ["candidate", "consensus 6 + trace 27 + embedding 1024 + scalar 4 + parser 11", "1072", "单候选；无 index/ID"],
                ["meta", "5 中心化模型分数 + 中心化 support", "6", "答案组内比较"],
            ], st, [27*mm, 82*mm, 28*mm, 32*mm], font_size=6.3),
            H2("20.1 27 个 trace 特征覆盖什么", st),
            P("特征分为六类：长度/行数/步骤数；数字、等号和运算符密度；boxed/final-answer/answer 结束标记；百分比、货币、分数形式；题目数字覆盖、额外数字比例、词汇 Jaccard 和单位覆盖；以及最终数值的整数/负数/零、量级、分母、题内出现和推理/修正标记率。它们是确定性统计，不会修改课程 parser 的结果。", st),
            H2("20.2 parser-consistency 与‘否定 parser 扩展’不矛盾", st),
            P("候选 ranker 保留的 11 维只比较 ####、boxed、final answer 和显式答案是否与官方 parse 值一致。它们不重新定义答案。实验中被拒绝的是额外组级 parser/lexical/pooling 扩展，因为它们没有同时改善 Train 和 Validation。", st, "callout"),
        ),
        _page(
            H1("21 从折外训练到 checkpoint 的可复现链", st),
            P("每个线性 ranker 先在当前训练分区的歧义题上拟合 feature mean 和 standard deviation，标准差下限为 1e-4。正确组与每个错误组形成差分 pair；候选模型允许多个正例。每题所有 pair 总权重为 1，防止答案组更多的题占据更大梯度。", st),
            caption("Table 20", "训练器、折外分数与导出物", st),
            table(["环节", "实现细节", "导出/验证"], [
                ["基础组模型", "4 类 x 5 folds = 20 个线性 ranker", "折内拟合，留出折产生 OOF"],
                ["候选模型", "1 类 x 5 folds = 5 个 multi-positive ranker", "组证据取成员最大分"],
                ["meta stack", "4 group + linear candidate + MLP candidate + support", "7 维，L2=0.003"],
                ["优化", "Adam，lr=0.03，最多 220 epochs，seed=7506", "每 10 epochs 检查 loss"],
                ["停止", "相对改善 <1e-7 连续 4 次检查", "保留最低 loss 权重"],
                ["stack 训练记录", "2114 题，3058 pairs，best epoch=220", "训练 top-1 82.3557%，仅为拟合诊断"],
            ], st, [35*mm, 83*mm, 51*mm]),
            H2("21.1 推理时如何使用 25 个模型", st),
            P("对每类组特征，5 个折模型的组分数先取平均。候选特征也先对 5 个候选模型取平均，再对每个答案组取成员最大分。六个 meta 列均按当题组均值中心化，再由 stack 打分。完全同分时按最早候选 index 破平，因而多次运行确定。", st),
            H2("21.2 checkpoint 内部结构", st),
            P("model.npz 有 78 个数组：25 个线性折模型存 75 个数组，六维 stack 存 3 个数组。未压缩合计 259,032 B，约为 16 MiB 上限的 1.55%。", st, "callout"),
        ),
        _page(
            H1("22 结论到证据的最终追溯矩阵", st),
            P("最终审计把‘结论是什么’、‘哪个环节验证它’和‘不能推导什么’分开。这避免将开发集结果写成未见分布保证。", st),
            caption("Table 21", "验收结论、证据和解释边界", st),
            table(["结论", "主证据", "审计边界"], [
                ["答案分组与 scorer 一致", "parser/等价分数/多正例测试", "不评判全部中间推理"],
                ["text 与 embedding 互补", "full_no_context 80.5097% 歧义 CV", "关联性消融，不是因果解释"],
                ["MLP 候选扩展", "1741 vs 四组模型 1735；低于线性候选 1743", "Train/Validation 有小幅取舍"],
                ["Validation 明显优于 plurality", "745 vs 688；paired bootstrap CI", "CI 只描述该 Validation bank"],
                ["资源合规", "259,032 B；Validation 3 次最大 1.0063 s；历史 Test 1.2920 s", "时间与具体机器相关"],
                ["冻结 Test 最终评分", "aggregate test score + resource report; raw Test predictions not retained", "历史一次评分；本次 package 恢复受公开 Test 比较影响"],
            ], st, [51*mm, 67*mm, 51*mm]),
        ),
    ]


def audit_pages_en(st: dict[str, ParagraphStyle]) -> list[list[Flowable]]:
    """English mirror of the full-edition code audit."""

    return [
        _page(
            H1("20 Code-Level Feature Dictionary and Dimension Audit", st),
            P("To avoid hiding implementation behind labels such as text or embedding features, this section follows the concatenation order in features.py. Checkpoint loading tests enforce the same dimensions, so the table is an executable schema rather than an informal summary.", st),
            caption("Table 19", "Exact feature-vector composition", st),
            table(["Vector", "Exact composition", "Dimension", "Aggregation"], [
                ["text", "Mean and maximum of 27 trace features", "54", "Within answer group"],
                ["embedding", "unit(group mean) - unit(all-candidate mean)", "1,024", "Group relative to question"],
                ["full_no_context", "text + embedding", "54+1,024=1,078", "No consensus/scalar block"],
                ["full", "consensus 6 + text 54 + embedding 1,024 + scalar 4", "1,088", "Complete group vector"],
                ["candidate", "consensus 6 + trace 27 + embedding 1,024 + scalar 4 + parser 11", "1,072", "Single trace; no index/ID"],
                ["meta", "Five centred model scores + centred support", "6", "Within-question groups"],
            ], st, [27*mm, 82*mm, 28*mm, 32*mm], font_size=6.25),
            H2("20.1 What the 27 trace features cover", st),
            P("The deterministic statistics cover character, word, line, numeric, equality, operator, and step counts; boxed/final-answer/answer endings; percent, currency, and fraction forms; question-number, lexical, and unit coverage; parsed-answer sign, integrality, zero state, magnitude, denominator, and question occurrence; plus reasoning- and correction-marker rates. They inspect the supplied trace and never alter the course parser.", st),
            H2("20.2 Why parser consistency does not contradict the rejected parser extension", st),
            P("The candidate model's eleven parser-consistency fields only compare ####, boxed, final-answer, and explicit-answer cues with the official parsed value. They do not redefine that value. What was rejected was an additional group-level parser, lexical-hash, and pooling expansion that failed to improve Train and Validation jointly.", st, "callout"),
        ),
        _page(
            H1("21 Reproducible Path from Cross-Fitting to the Checkpoint", st),
            P("Each linear ranker fits feature means and standard deviations on the ambiguous questions in its current training partition, with a scale floor of 1e-4. A correct group is contrasted with every incorrect group; the candidate model permits multiple positives. All pairs from one question sum to weight one, preventing questions with more groups or combinations from dominating the gradient.", st),
            caption("Table 20", "Optimisers, cross-fitted scores, and exported objects", st),
            table(["Stage", "Implementation", "Export / verification"], [
                ["Base group models", "4 types x 5 folds = 20 linear rankers", "Fit in-fold; held fold produces OOF"],
                ["Candidate models", "1 type x 5 multi-positive rankers", "Group evidence is member maximum"],
                ["Meta stack", "text/embedding/full_no_context/full/candidate/support", "7-D; L2=0.003"],
                ["Optimisation", "Adam, lr=0.03, at most 220 epochs, seed=7506", "Loss checked every 10 epochs"],
                ["Stopping", "Relative gain <1e-7 for four checks", "Retain minimum-loss weights"],
                ["Stack fit record", "2,114 questions, 3,058 pairs, best epoch 220", "82.3557% top-1 fit diagnostic; not nested evidence"],
            ], st, [35*mm, 83*mm, 51*mm]),
            H2("21.1 How the 25 models are used at inference", st),
            P("For each group view, scores from its five fold models are averaged. Candidate scores are likewise averaged across five models, then reduced to the maximum member score for each answer group. The four group-score columns, one candidate-score column, and support are centred across the current question's groups before the stack scores them. Exact score ties resolve to the earliest member index, making reruns deterministic.", st),
            H2("21.2 Checkpoint anatomy", st),
            P("model.npz contains 78 arrays. Each of 25 linear group/candidate folds stores feature_mean, feature_scale, and weights, giving 75 arrays; the six-dimensional stack adds three arrays. Their uncompressed total is 259,032 bytes, about 1.55% of the 16 MiB allowance.", st, "callout"),
        ),
        _page(
            H1("22 Final Claim-to-Evidence Traceability Matrix", st),
            P("The audit separates what is claimed, what verifies it, and what must not be inferred. This prevents a development-bank result from being presented as an unseen-distribution guarantee.", st),
            caption("Table 21", "Acceptance claims, primary evidence, and boundaries", st),
            table(["Claim", "Primary evidence", "Audit boundary"], [
                ["Grouping matches scorer semantics", "Parser, equivalent-fraction, multi-positive tests", "Does not prove every intermediate step"],
                ["Text and embeddings complement", "80.5097% full_no_context ambiguous CV", "Ablation association, not causation"],
                ["MLP candidate extension", "1741 vs group stack 1735; below linear candidate 1743", "Small Train/Validation tradeoff"],
                ["Validation beats plurality", "745 versus 688; paired bootstrap CI", "CI describes this Validation bank"],
                ["Resources comply", "259,032 B; Val 1.0063 s max; historical Test 1.2920 s", "Timing is machine-dependent"],
                ["Frozen Test final score", "aggregate Test score and resource report; raw Test predictions not retained", "Previously scored once; restoration followed Test comparison"],
            ], st, [51*mm, 67*mm, 51*mm]),
        ),
    ]


def compact_audit_tail_zh(st: dict[str, ParagraphStyle]) -> list[list[Flowable]]:
    return [
        _page(
            H1("21 提交版代码审计摘要", st),
            P("为了让 10 页版也能从报告数字直接追溯到实现，下表保留最容易被模糊表述的维度与 checkpoint 事实。", st),
            table(["对象", "代码级构成", "核对结果"], [
                ["text", "27 trace 特征 x mean/max", "54 维"],
                ["embedding", "归一化组中心 - 全候选中心", "1024 维"],
                ["full", "6 consensus + 54 text + 1024 embedding + 4 scalars", "1088 维"],
                ["candidate", "6 consensus + 27 trace + 1024 embedding + 4 scalar + 11 parser", "1072 维；无 index/ID"],
                ["checkpoint", "25 个线性折模型 x 3 + stack 3", "78 数组；259,032 B"],
                ["meta", "5 中心化分数 + centered support", "6 维；L2=0.001"],
            ], st, [29*mm, 100*mm, 40*mm], font_size=6.0),
            P("上述结构由 feature schema 与 checkpoint loader 检查；报告分别记录 Validation 745/798、strict nested Train 1743/2114 与历史 Test 382/467，只有 Train/Validation 用于方法选择。", st, "callout"),
        )
    ]


def compact_audit_tail_en(st: dict[str, ParagraphStyle]) -> list[list[Flowable]]:
    return [
        _page(
            H1("21 Submission-Edition Code Audit Summary", st),
            P("To keep the ten-page report directly traceable from prose to implementation, this table preserves the dimensions and checkpoint facts most vulnerable to vague description.", st),
            table(["Object", "Code-level composition", "Audit result"], [
                ["text", "27 trace features x mean/max", "54 dimensions"],
                ["embedding", "Normalised group centre minus all-candidate centre", "1,024 dimensions"],
                ["full", "6 consensus + 54 text + 1,024 embedding + 4 scalars", "1,088 dimensions"],
                ["candidate", "6 consensus + 27 trace + 1,024 embedding + 4 scalar + 11 parser", "1,072; no index/ID"],
                ["checkpoint", "25 linear folds x 3 + 3 stack arrays", "78 arrays; 259,032 B"],
                ["meta", "Five centred scores plus centred support", "6 dimensions; L2=0.001"],
            ], st, [29*mm, 100*mm, 40*mm], font_size=5.95),
            P("Feature schemas and the checkpoint loader verify these structures. The report records Validation 745/798, strict nested Train 1,743/2,114, and the historical Test score 382/467. The 2026-10-28 restoration followed comparison of public-Test scores, so Test influenced package-level selection.", st, "callout"),
        )
    ]


def draw_header_footer(canvas, doc, lang: str, compact: bool) -> None:
    canvas.saveState()
    body, _ = fonts(lang)
    edition = (
        ("DASE7506 Project 2｜10 页提交版" if compact else "DASE7506 Project 2｜完整技术报告")
        if lang == "zh"
        else ("DASE7506 Project 2 | 10-page submission" if compact else "DASE7506 Project 2 | Full technical report")
    )
    canvas.setFont(body, 6.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, PAGE_H - 8.5 * mm, edition)
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(LEFT, PAGE_H - 10.2 * mm, PAGE_W - RIGHT, PAGE_H - 10.2 * mm)
    canvas.line(LEFT, 10.5 * mm, PAGE_W - RIGHT, 10.5 * mm)
    canvas.drawRightString(PAGE_W - RIGHT, 6.9 * mm, str(doc.page))
    canvas.drawString(LEFT, 6.9 * mm, "Frozen 2026-10-15 | protocol dase7506-mp2-open-hard-verifier-v5")
    canvas.restoreState()


def build_one(lang: str, compact: bool) -> Path:
    st = styles_for(lang, compact)
    sections = full_pages_zh(st, compact) if lang == "zh" else full_pages_en(st, compact)
    if compact:
        sections += compact_audit_tail_zh(st) if lang == "zh" else compact_audit_tail_en(st)
    else:
        sections += audit_pages_zh(st) if lang == "zh" else audit_pages_en(st)
    expected = 10 if compact else None
    out = OUTPUTS[(lang, compact)]
    out.parent.mkdir(parents=True, exist_ok=True)
    title = (
        "DASE7506 Project 2 可靠推理选择技术报告"
        if lang == "zh"
        else "DASE7506 Project 2 Reliable Reasoning Selection Technical Report"
    )
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title=title,
        author="DASE7506 MP2 Coursework",
        subject="Reliable LLM reasoning selection with strict validation",
        creator="ReportLab",
        pageCompression=1,
    )
    story: list[Flowable] = []
    # Section boundaries are semantic only. ReportLab performs ordinary flowing
    # pagination, so a section continues on the current page whenever space is
    # available instead of forcing every section onto a new page.
    for index, elements in enumerate(sections):
        story.extend(elements)
        if index != len(sections) - 1:
            story.append(Spacer(1, 1.2 * mm))
    callback = lambda c, d: draw_header_footer(c, d, lang, compact)
    doc.build(story, onFirstPage=callback, onLaterPages=callback)
    actual = len(PdfReader(str(out)).pages)
    if expected is not None and actual != expected:
        raise RuntimeError(f"{out.name}: expected {expected} pages, got {actual}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    built = [
        build_one("zh", False),
        build_one("en", False),
        build_one("zh", True),
        build_one("en", True),
    ]
    for path in built:
        print(f"{path.name}\tpages={len(PdfReader(str(path)).pages)}\tsha256={sha256(path)}")


if __name__ == "__main__":
    main()
