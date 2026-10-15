#!/usr/bin/env python3
"""Build the GR308 Qwen concurrency benchmark report."""

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "pdf"
WORK_DIR = ROOT / "tmp" / "pdfs" / "gr308_qwen_report"
OUT_FILE = OUT_DIR / "风华_GR308_Qwen系列并发性能测试与调优报告_20260915.pdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

FONT_REG = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
pdfmetrics.registerFont(TTFont("Heiti", FONT_REG, subfontIndex=0))
pdfmetrics.registerFont(TTFont("HeitiBold", FONT_BOLD, subfontIndex=0))

BLUE = HexColor("#1675C1")
NAVY = HexColor("#0D2B4D")
GREEN = HexColor("#168A68")
ORANGE = HexColor("#D77A00")
RED = HexColor("#C74444")
GRAY = HexColor("#626A73")
LIGHT = HexColor("#F3F6FA")
GRID = HexColor("#C6D2DF")
PALE_BLUE = HexColor("#EAF3FB")
PALE_GREEN = HexColor("#E9F6F1")
PALE_ORANGE = HexColor("#FFF4E5")


reference = {
    "35B": {1: (1731.87, 52.91), 16: (3487.11, 12.74)},
    "9B": {1: (1545.07, 45.09), 16: (2487.47, 11.97)},
    "30B": {1: (1245.71, 26.85), 16: (3384.84, 11.70)},
    "80B": {1: (1072.45, 33.23), 16: (6048.71, 15.02)},
}

measured = {
    "35B": {1: (1537.19, 50.75, 47.21), 16: (3321.47, 13.93, 180.64)},
    "9B": {1: (2335.93, 54.64, 48.62), 16: (3501.62, 18.90, 247.16)},
    "30B": {1: (1054.43, 26.43, 25.76), 16: (4225.69, 15.22, 225.67)},
    "80B": {1: (595.37, 28.86, 28.41), 16: (3528.79, 16.20, 231.98)},
}


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CNBody", fontName="Heiti", fontSize=9.2, leading=14.2,
    textColor=colors.black, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="CNSmall", fontName="Heiti", fontSize=7.4, leading=10.5,
    textColor=GRAY,
))
styles.add(ParagraphStyle(
    name="CNNote", fontName="Heiti", fontSize=7.8, leading=11,
    textColor=GRAY, backColor=LIGHT, borderPadding=7, spaceBefore=4,
))
styles.add(ParagraphStyle(
    name="CNSection", fontName="Heiti", fontSize=16.5, leading=22,
    textColor=BLUE, spaceBefore=2, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="CNSub", fontName="HeitiBold", fontSize=10.5, leading=14,
    textColor=NAVY, spaceBefore=5, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CNTitle", fontName="Heiti", fontSize=23, leading=30,
    textColor=NAVY, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="CNSubtitle", fontName="Heiti", fontSize=10.2, leading=14,
    textColor=GRAY, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="Cell", fontName="Heiti", fontSize=7.2, leading=9.4,
    textColor=colors.black,
))
styles.add(ParagraphStyle(
    name="CellSmall", fontName="Heiti", fontSize=6.6, leading=8.5,
    textColor=colors.black,
))
styles.add(ParagraphStyle(
    name="CellHead", fontName="HeitiBold", fontSize=7.2, leading=9.2,
    alignment=TA_CENTER, textColor=NAVY,
))


def P(text, style="CNBody"):
    return Paragraph(text, styles[style])


def section(number, title):
    return P(f"{number}、{title}", "CNSection")


def styled_table(data, widths, *, header=True, font_size=7.2, aligns=None, row_bgs=None):
    wrapped = []
    for ri, row in enumerate(data):
        wrapped.append([
            cell if hasattr(cell, "wrap") else P(str(cell), "CellHead" if header and ri == 0 else ("CellSmall" if font_size < 7 else "Cell"))
            for cell in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E9EEF4")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#9FB2C6")),
        ]
    if aligns:
        for col, align in aligns.items():
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    if row_bgs:
        for row, bg in row_bgs.items():
            commands.append(("BACKGROUND", (0, row), (-1, row), bg))
    table.setStyle(TableStyle(commands))
    return table


def kpi_cards():
    data = [
        [P("C16 最高聚合吞吐", "CNSmall"), P("C16 最高 Decode TPS", "CNSmall"), P("验证完成度", "CNSmall")],
        [P("<font size='16' color='#0D2B4D'><b>247.16 tok/s</b></font><br/><font size='8'>Qwen3.5-9B</font>", "CNBody"),
         P("<font size='16' color='#0D2B4D'><b>18.90 tok/s</b></font><br/><font size='8'>Qwen3.5-9B</font>", "CNBody"),
         P("<font size='16' color='#0D2B4D'><b>3 正式 + 2 短测</b></font><br/><font size='8'>35B 新候选待正式复测</font>", "CNBody")],
    ]
    t = Table(data, colWidths=[169, 169, 169], rowHeights=[20, 54])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def pct_text(value, lower_is_better=False):
    good = value < 0 if lower_is_better else value > 0
    color = "#168A68" if good else "#C74444"
    sign = "+" if value >= 0 else ""
    return f"<font color='{color}'><b>{sign}{value:.1f}%</b></font>"


def page_decor(canvas, doc):
    canvas.saveState()
    w, h = LETTER
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(52, 35, w - 52, 35)
    canvas.setFont("Heiti", 7.4)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(w - 52, 20, f"风华 GR308 | Qwen 并发性能测试与调优 | 2026-09-15   {doc.page}")
    canvas.restoreState()


chart_path = WORK_DIR / "performance_comparison.png"
if not chart_path.exists():
    raise FileNotFoundError(f"chart must be generated first: {chart_path}")

doc = BaseDocTemplate(
    str(OUT_FILE), pagesize=LETTER,
    leftMargin=52, rightMargin=52, topMargin=52, bottomMargin=48,
    title="风华 GR308 Qwen 系列并发性能测试与调优报告",
    author="Codex",
    subject="Qwen3.6-35B-A3B、Qwen3.5-9B、Qwen3-30B-A3B、Qwen3-Next-80B-A3B 性能测试",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page_decor))

story = []

# Page 1
story += [
    P("性能测试数据研究报告", "CNSmall"),
    P("风华 GR308 Qwen 系列并发性能测试与调优报告", "CNTitle"),
    P("vLLM 0.18.0 | BF16 | C1 / C16 | MTP 与 CUDA Graph 参数优化 | 35B 二次短测更新", "CNSubtitle"),
    P("<b>测试对象：</b>Qwen3.6-35B-A3B、Qwen3.5-9B、Qwen3-30B-A3B、Qwen3-Next-80B-A3B-Instruct。"),
    P("<b>测试平台：</b>4 张 GR308 物理卡、8 个逻辑 GPU；每个逻辑 GPU 约 56 GiB GDDR6、1008 GB/s；驱动 1.1.0.19。"),
    P("<b>测试工具：</b>vLLM bench serve；temperature=0、ignore-eos、request-rate=inf。前三个模型输入/输出 3072/1024，80B 输入/输出 1024/1024。"),
    Spacer(1, 7), kpi_cards(), Spacer(1, 14),
    section("一", "结论摘要"),
    P("<b>结论 1｜9B 吞吐领先。</b>双 TP1 / DP2、MTP4 配置在 C16 达到 247.16 tok/s 聚合吞吐和 18.90 Decode Avg TPS，相对对照表提升 57.9%。"),
    P("<b>结论 2｜80B 高并发达标。</b>TP4、MTP1、Graph64 在 C16 达到 16.20 Decode Avg TPS，较对照提升 7.9%；TTFT 降至 3.53 s，改善 41.7%。"),
    P("<b>结论 3｜30B 偏向吞吐。</b>无 EP 的 TP2 配置使 C16 Decode Avg TPS 提升 30.1%，但 TTFT 增加 24.8%，适合吞吐优先而非低延迟场景。"),
    P("<b>结论 4｜35B 新候选显著增益。</b>MTP4、Graph64、batch4096 的 D 组短测在 C16 达到 180.64 tok/s 聚合吞吐和 13.93 Decode Avg TPS；相对对照 TPS 提升 9.3%，TTFT 改善 4.7%。该结果尚待 64+64 正式复测。"),
    P("<b>结论 5｜自定义 all-reduce 始终保持启用。</b>所有有效结果均未使用 --disable-custom-all-reduce；80B 自动 KV 组启动失败，固定 8 GiB 更稳定。"),
    section("二", "测试口径"),
]

method_data = [
    ["项目", "定义", "本次状态"],
    ["并发档位", "C1 与 C16；C1 串行、C16 最大并发 16", "四模型覆盖"],
    ["正式测试", "C1/C16 各 64 prompts", "35B 原正式基线、9B、30B"],
    ["最新短测", "C1 16 prompts；C16 32 prompts", "35B 新候选、80B；均待正式复测"],
    ["Decode Avg TPS", "1000 / mean TPOT(ms)，代表平均解码速率", "与对照表同口径"],
    ["聚合吞吐", "并发窗口内总 output token throughput", "用于高吞吐方案排序"],
]
story += [
    styled_table(method_data, [92, 300, 115]),
    P("说明：对照表数值用于描述性比较。35B 与 80B 当前展示最新短测候选，不能直接作为正式验收结果；9B 的双 TP1 / DP2 与对照 TP2、30B 的无 EP TP2 与对照 EP+TP2 并非严格同配置。", "CNNote"),
    PageBreak(),
]

# Page 2
story += [section("三", "四模型测试结果与对照")]
result_rows = [["模型", "并发", "TTFT 实测", "TTFT 对照", "TTFT 变化", "Decode 实测", "Decode 对照", "TPS 变化", "聚合吞吐"]]
for model in ["35B", "9B", "30B", "80B"]:
    for c in (1, 16):
        ttft, dec, agg = measured[model][c]
        rttft, rdec = reference[model][c]
        dt = (ttft / rttft - 1) * 100
        dd = (dec / rdec - 1) * 100
        result_rows.append([
            {"35B":"Qwen3.6-35B-A3B*", "9B":"Qwen3.5-9B", "30B":"Qwen3-30B-A3B", "80B":"Qwen3-Next-80B-A3B*"}[model] if c == 1 else "",
            f"C{c}", f"{ttft:.2f}", f"{rttft:.2f}", pct_text(dt, True),
            f"{dec:.2f}", f"{rdec:.2f}", pct_text(dd, False), f"{agg:.2f}",
        ])
result_table = styled_table(
    result_rows, [94, 28, 57, 57, 53, 57, 57, 50, 54], font_size=6.6,
    aligns={1:"CENTER",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT",6:"RIGHT",7:"RIGHT",8:"RIGHT"},
    row_bgs={2:HexColor("#F8FAFC"),4:HexColor("#F8FAFC"),6:HexColor("#F8FAFC"),8:HexColor("#F8FAFC")},
)
result_table.setStyle(TableStyle([
    ("SPAN", (0,1), (0,2)), ("SPAN", (0,3), (0,4)),
    ("SPAN", (0,5), (0,6)), ("SPAN", (0,7), (0,8)),
    ("VALIGN", (0,1), (0,-1), "MIDDLE"),
]))
story += [result_table, Spacer(1, 10)]

story += [
    P("<b>Qwen3.6-35B-A3B：</b>D 组短测 C1 Decode Avg TPS 为 50.75，距对照缩小至 4.1%，TTFT 改善 11.2%；C16 Decode Avg TPS 达到 13.93、提升 9.3%，TTFT 同时改善 4.7%。相对上一正式配置，C16 聚合吞吐提升约 8.0%，但 TTFT 增加约 24.4%。"),
    P("<b>Qwen3.5-9B：</b>C1/C16 Decode Avg TPS 分别提升 21.2% 和 57.9%，吞吐收益最大；相应 TTFT 增加 51.2% 和 40.8%，体现双 TP1 / DP2 的吞吐优先取向。"),
    P("<b>Qwen3-30B-A3B：</b>C1 TTFT 改善 15.4%，Decode Avg TPS 基本持平；C16 Decode Avg TPS 提升 30.1%，但 TTFT 增加 24.8%。"),
    P("<b>Qwen3-Next-80B-A3B：</b>C1 TTFT 改善 44.5%，但 Decode Avg TPS 低 13.1%；C16 同时实现 TPS +7.9% 和 TTFT -41.7%，是最明确的综合改善。"),
    P("方向说明：绿色表示相对对照改善，红色表示退化。TTFT 的负变化为改善，TPS 的正变化为改善。带 * 的 35B、80B 为短测候选。", "CNNote"),
    section("四", "资源与可比性")
]
resource_data = [
    ["模型", "拓扑", "输入/输出", "样本量", "与对照可比性"],
    ["35B*", "TP2 / 1 物理卡", "3072 / 1024", "16 + 32；原正式 64 + 64", "同 TP2/负载；新候选为短测"],
    ["9B", "双 TP1 / DP2 / 1 物理卡", "3072 / 1024", "64 + 64", "有限：对照为 TP2"],
    ["30B", "TP2、无 EP / 1 物理卡", "3072 / 1024", "64 + 64", "有限：对照为 EP+TP2"],
    ["80B", "TP4 / 2 物理卡", "1024 / 1024", "16 + 32", "短测：负载相同、样本量较少"],
]
story += [styled_table(resource_data, [72, 137, 83, 67, 148]), PageBreak()]

# Page 3
story += [
    section("五", "性能图表"),
    P("图 1：四个模型在 C1/C16 下的 Decode Avg TPS 与 Median TTFT。同一模型内部与对照表比较；不同模型因拓扑与输入长度不同，不按绝对数值推导模型优劣。", "CNSmall"),
    Spacer(1, 5),
    Image(str(chart_path), width=7.05*inch, height=4.60*inch),
    Spacer(1, 9),
    section("六", "图表解读"),
    P("<b>吞吐观察：</b>9B 的 C16 18.90 Decode Avg TPS 与 247.16 tok/s 聚合吞吐仍为四模型最高。35B D 组把 C16 Decode Avg TPS 提升至 13.93、聚合吞吐提升至 180.64，首次同时超过对照 TPS 与 TTFT。"),
    P("<b>延迟观察：</b>35B D 组相对对照的 C1/C16 TTFT 均改善，但相对上一正式配置，C16 TTFT 从 2671 ms 增至 3321 ms。80B 的 C1/C16 TTFT 均大幅低于对照；9B 与 30B 的吞吐配置仍存在延迟代价。"),
    P("<b>决策变化：</b>35B 的部署候选由 MTP3 / batch2048 更新为 MTP4 / batch4096，但必须先完成正式复测。生产部署仍不宜只设一个统一参数；80B 可将 batched tokens 4096 作为低 TTFT 备选。"),
    P("统计限制：当前每个最终配置只有一个固定 seed 的结果；没有重复运行、标准差或置信区间，因此报告不作显著性结论。", "CNNote"),
    PageBreak(),
]

# Page 4
story += [section("七", "35B 调优矩阵与跨卡验证")]
data35 = [
    ["候选", "关键差异", "C1 Decode", "C16 Decode", "C16 聚合", "C16 TTFT", "判断"],
    ["原正式", "MTP3 / G1-128 / b2048", "47.40", "11.78", "167.31", "2671 ms", "正式基线"],
    ["A", "MTP4 / G4-64 / b2048", "50.76", "13.65", "175.57", "5594 ms", "有效"],
    ["B", "MTP4 / G4-128 / b2048", "50.81", "13.58", "176.49", "4973 ms", "C1 最优"],
    ["C", "MTP4 / G4-64 / b3072", "50.18", "13.56", "176.41", "4819 ms", "中间档"],
    ["D", "MTP4 / G4-64 / b4096", "50.75", "13.93", "180.64", "3321 ms", "短测最优"],
    ["E", "MTP3 / G4-64 / b2048", "47.67", "12.53", "164.38", "4623 ms", "MTP3 对照"],
    ["F", "MTP3 / G4-64 / b4096", "47.14", "12.36", "140.61", "4344 ms", "排除"],
]
story += [
    styled_table(data35, [40, 137, 59, 61, 62, 61, 61], font_size=6.6,
                 row_bgs={1:PALE_BLUE,5:PALE_GREEN}),
    P("D 组为单轮短测（C1 16 prompts、C16 32 prompts），全部请求成功；相对原正式配置的吞吐提升需要通过跨卡重复和 64+64 prompts 正式测试确认。", "CNNote"),
    Spacer(1, 8),
]

story += [section("八", "80B 短测矩阵")]
data80 = [
    ["组别", "配置", "C1 Decode", "C16 Decode", "C16 聚合", "C16 TTFT", "判断"],
    ["A", "Eager / 无 MTP / 8 GiB", "10.82", "11.61", "177.71", "3515 ms", "基线"],
    ["B", "Eager / MTP1 / 8 GiB", "26.28", "16.01", "231.30", "3308 ms", "MTP 有效"],
    ["C", "MTP1 / Graph32 / 8 GiB", "28.93", "16.03", "228.68", "3205 ms", "C1 最优"],
    ["D", "MTP1 / Graph64 / 8 GiB", "28.86", "16.20", "231.98", "3529 ms", "吞吐最优"],
    ["E", "Graph64 / 自动 KV", "-", "-", "-", "-", "启动失败"],
    ["F", "Graph64 / batch4096", "28.63", "16.02", "230.99", "2975 ms", "TTFT 最优"],
    ["G", "Graph64 / batch4096 / async", "28.90", "15.84", "228.07", "4760 ms", "排除"],
]
story += [
    styled_table(data80, [34, 154, 63, 67, 67, 64, 58], font_size=6.6, row_bgs={4:PALE_GREEN,6:PALE_ORANGE}),
    P("调优判断：MTP1 是 80B 的主要增益来源；Graph32/64 进一步提升 C1。自动 KV 在本轮引擎初始化失败，async 则降低 C16 吞吐并增加 TTFT。", "CNNote"),
    PageBreak(),
]

# Page 5
story += [section("九", "最终配置与部署建议")]
config_data = [
    ["模型", "并行 / MTP", "Graph", "Batched tokens", "KV / Block", "调度", "建议"],
    ["35B*", "TP2 / MTP4", "4,8,16,32,64", "4096", "自动 / 1088", "同步", "待正式复测"],
    ["9B", "TP1 + DP2 / MTP4", "4,8,16,32", "4096", "自动 / 800", "同步", "吞吐优先"],
    ["30B", "TP2 / 无 MTP / 无 EP", "4,8,16,32", "4096", "8 GiB / 800", "async", "吞吐优先"],
    ["80B", "TP4 / MTP1", "4,8,16,32,64", "2048", "8 GiB / 1088", "同步", "待正式复测"],
]
story += [styled_table(config_data, [60, 89, 132, 65, 76, 48, 58], font_size=6.6), Spacer(1, 9)]

story += [
    P("<b>统一约束：</b>精度均为 BF16；自定义 all-reduce 保持启用，不加入 --disable-custom-all-reduce；关闭 prefix caching；max model length 4096；max num seqs 16。"),
    P("<b>35B：</b>最新高吞吐候选为 D 组（MTP4 / Graph64 / batch4096）。其 C16 Decode TPS 和聚合吞吐均高于原正式配置，但当前仅为短测；上线前必须完成跨卡重复与 64+64 prompts 正式复测。若更重视 C16 TTFT，可继续保留原 MTP3 / batch2048 配置。"),
    P("<b>9B：</b>双 TP1 / DP2 适合吞吐型服务。若业务更重视首包延迟，应保留 TP2 低延迟配置作为独立服务档位，不应以当前高吞吐配置替代所有场景。"),
    P("<b>30B：</b>无 EP 的 TP2 在当前软件栈下稳定且 C16 吞吐高。EP 未证明能稳定带来收益，不建议在正式配置中强制启用。"),
    P("<b>80B：</b>高吞吐首选 Graph64 / batch2048；如更重视 C16 TTFT，可选 Graph64 / batch4096，Decode Avg TPS 仅从 16.20 降至 16.02，而 TTFT 从 3529 ms 降至 2975 ms。正式上线前必须补跑 64+64 requests。"),
    section("十", "运行边界、异常与结论"),
    P("<b>已知异常：</b>并发启动多个 FHCCL/RCCL communicator 时，35B 初始候选出现通信初始化失败；固定 75 秒错峰仍不足，改为上一服务 API ready 后再启动下一通信域后，六组短测全部完成且请求零失败。80B 自动 KV 组在 EngineCore 初始化阶段失败。上述异常均未造成硬件健康告警。"),
    P("<b>硬件状态：</b>测试结束后 8 个逻辑设备均为 OK，显存回落至约 540 MiB；四张物理卡空闲。服务器为双路 AMD EPYC 9654、约 1 TiB 内存、Ubuntu 22.04 / Linux 5.15.0-191。"),
    P("<b>最终结论：</b>9B 仍具有最高绝对吞吐；35B D 组短测达到 180.64 tok/s 聚合吞吐，并在 C16 同时超过对照 TPS 与 TTFT，但尚不能替代原正式结论。30B 和 80B 在 C16 超过对照；35B 与 80B 的最新候选均需补跑正式测试。"),
    P("<b>数据路径：</b><br/>/data/infra/benchmarks/gr308-35b-mtp4-short-matrix-ready-20260915/<br/>/data/infra/benchmarks/gr308-35b-two-card-tps-20260915/<br/>/data/infra/benchmarks/gr308-final-high-throughput-20260915/<br/>/data/infra/benchmarks/gr308-80b-short-tuning-20260915/", "CNSmall"),
    P("报告依据：服务器原生 benchmark.json、result.json、launch.json、server.log 与 FANTASY SMI 输出；整理日期 2026-09-15。", "CNNote"),
]

doc.build(story)
print(OUT_FILE)
