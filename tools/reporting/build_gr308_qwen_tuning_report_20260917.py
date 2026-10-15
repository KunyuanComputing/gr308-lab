#!/usr/bin/env python3
"""Build the comprehensive 2026-09-17 GR308 Qwen tuning report."""

from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "output" / "风华_GR308_Qwen系列并发性能测试与调优报告_20260917"
DATA = json.loads((REPORT_DIR / "data" / "report_data.json").read_text())
FIG = REPORT_DIR / "figures"
OUT = ROOT / "output" / "pdf" / "风华_GR308_Qwen系列并发性能测试与调优报告_20260917.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

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
PALE_RED = HexColor("#FCEEEE")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CNBody", fontName="Heiti", fontSize=9.0, leading=14.0,
                          textColor=colors.black, spaceAfter=5))
styles.add(ParagraphStyle(name="CNSmall", fontName="Heiti", fontSize=7.2, leading=10.2,
                          textColor=GRAY))
styles.add(ParagraphStyle(name="CNNote", fontName="Heiti", fontSize=7.7, leading=10.8,
                          textColor=GRAY, backColor=LIGHT, borderPadding=7, spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="CNWarn", fontName="Heiti", fontSize=7.7, leading=10.8,
                          textColor=RED, backColor=PALE_RED, borderPadding=7, spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="CNSection", fontName="Heiti", fontSize=16.2, leading=21,
                          textColor=BLUE, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle(name="CNSub", fontName="HeitiBold", fontSize=10.5, leading=14,
                          textColor=NAVY, spaceBefore=5, spaceAfter=4))
styles.add(ParagraphStyle(name="CNTitle", fontName="Heiti", fontSize=23, leading=30,
                          textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle(name="CNSubtitle", fontName="Heiti", fontSize=10.0, leading=14,
                          textColor=GRAY, spaceAfter=12))
styles.add(ParagraphStyle(name="Cell", fontName="Heiti", fontSize=7.0, leading=9.2,
                          textColor=colors.black))
styles.add(ParagraphStyle(name="CellSmall", fontName="Heiti", fontSize=6.2, leading=8.1,
                          textColor=colors.black))
styles.add(ParagraphStyle(name="CellHead", fontName="HeitiBold", fontSize=6.8, leading=8.8,
                          alignment=TA_CENTER, textColor=NAVY))


def P(text, style="CNBody"):
    return Paragraph(str(text), styles[style])


def section(number, title):
    return P(f"{number}、{title}", "CNSection")


def styled_table(data, widths, *, font_size=7.0, row_bgs=None, spans=None, aligns=None, repeat=True):
    wrapped = []
    for ri, row in enumerate(data):
        wrapped.append([
            cell if hasattr(cell, "wrap") else P(cell, "CellHead" if ri == 0 else ("CellSmall" if font_size < 6.8 else "Cell"))
            for cell in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if repeat else 0, hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), .45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E9EEF4")),
        ("LINEBELOW", (0, 0), (-1, 0), .8, HexColor("#9FB2C6")),
    ]
    if row_bgs:
        for row, bg in row_bgs.items():
            cmds.append(("BACKGROUND", (0, row), (-1, row), bg))
    if spans:
        for start, end in spans:
            cmds.append(("SPAN", start, end))
    if aligns:
        for col, align in aligns.items():
            cmds.append(("ALIGN", (col, 1), (col, -1), align))
    table.setStyle(TableStyle(cmds))
    return table


def pct_text(value, lower_is_better=False):
    good = value < 0 if lower_is_better else value > 0
    color = "#168A68" if good else "#C74444"
    sign = "+" if value >= 0 else ""
    return f"<font color='{color}'><b>{sign}{value:.1f}%</b></font>"


def page_decor(canvas, doc):
    canvas.saveState()
    w, _ = LETTER
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(.45)
    canvas.line(52, 35, w - 52, 35)
    canvas.setFont("Heiti", 7.3)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(w - 52, 20, f"风华 GR308 | Qwen 并发性能测试与调优 | 2026-09-17   {doc.page}")
    canvas.restoreState()


def kpi_cards():
    values = [
        ("C32 单卡最高吞吐", "320.90 tok/s", "Qwen3-30B-A3B"),
        ("C32 最高总吞吐", "339.26 tok/s", "80B / 2 张物理卡"),
        ("最高屋顶效率", "40.1%", "30B / C32"),
    ]
    rows = [[P(x[0], "CNSmall") for x in values], [
        P(f"<font size='16' color='#0D2B4D'><b>{v}</b></font><br/><font size='8'>{s}</font>", "CNBody")
        for _, v, s in values
    ]]
    t = Table(rows, colWidths=[169, 169, 169], rowHeights=[20, 54])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), .6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), .45, GRID), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def tuning_table(model):
    rows = [["组别", "关键变量", "C1 聚合", "C16 聚合", "C32 聚合", "判断"]]
    bgs = {}
    for i, r in enumerate(DATA["tuning"][model], 1):
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        rows.append([r["group"], r["change"], fmt(r["c1"]), fmt(r["c16"]), fmt(r["c32"]), r["verdict"]])
        if r["group"] in ("Final", "MTP3", "M4-D"):
            bgs[i] = PALE_GREEN
        elif "EP" in r["group"] or "Triton" in r["group"]:
            bgs[i] = PALE_RED
    return styled_table(rows, [39, 218, 58, 60, 60, 72], font_size=6.2, row_bgs=bgs,
                        aligns={2: "RIGHT", 3: "RIGHT", 4: "RIGHT"})


doc = BaseDocTemplate(
    str(OUT), pagesize=LETTER,
    leftMargin=52, rightMargin=52, topMargin=50, bottomMargin=48,
    title="风华 GR308 Qwen 系列并发性能测试与调优报告",
    author="Codex",
    subject="Qwen3.6-35B-A3B、Qwen3.5-9B、Qwen3-30B-A3B、Qwen3-Next-80B-A3B 调优与选型",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page_decor))
story = []

# Page 1: cover and executive conclusions.
story += [
    P("性能测试数据研究报告", "CNSmall"),
    P("风华 GR308 Qwen 系列并发性能测试与调优报告", "CNTitle"),
    P("vLLM 0.18.0 | BF16 | C1 / C16 / C32 | MTP、CUDA Graph、KV、MoE backend 与 EP 消融 | 最终选定方案", "CNSubtitle"),
    P("<b>测试对象：</b>Qwen3.6-35B-A3B、Qwen3.5-9B、Qwen3-30B-A3B、Qwen3-Next-80B-A3B-Instruct。"),
    P("<b>平台：</b>4 张 GR308 物理卡 / 8 个逻辑设备；双路 AMD EPYC 9654；约 1 TiB 内存；驱动 1.1.0.19。"),
    P("<b>负载：</b>35B/9B/30B 输入输出 3072/1024；80B 为 1024/1024；temperature=0、ignore-eos、request-rate=inf。"),
    Spacer(1, 7), kpi_cards(), Spacer(1, 13),
    section("一", "结论摘要"),
    P("<b>结论 1｜30B 是当前单卡高并发首选。</b>纯 TP2、不开 EP、auto MoE、batch4096、async 在 C32 达到 <b>320.90 tok/s</b>，占理想批处理屋顶 <b>40.1%</b>。"),
    P("<b>结论 2｜9B 的双 TP1/DP2 是最大结构性收益。</b>C32 达 <b>287.02 tok/s</b>，比 C16 提高 16.1%；但 TTFT 升至 8.81 s，因此 C16 是平衡档，C32 是吞吐档。"),
    P("<b>结论 3｜35B 在当前软件栈下已接近参数调优上限。</b>MTP4 + batch4096 的 C16/C32 为 <b>180.64/208.08 tok/s</b>；Graph128、batch6144、Triton grouped GEMM 与 EP2 均未带来足够收益。"),
    P("<b>结论 4｜80B 的主要收益来自 MTP。</b>TP4 + MTP3 在 C32 达 <b>339.26 tok/s</b>，比 C16 提高 32.6%；TP4+EP4 的 C16 反而下降 8.6%。"),
    P("<b>结论 5｜当前 EP 不建议进入生产配置。</b>30B/35B 的 EP2 和 80B 的 EP4 均回归；80B TP2×DP2×EP2 在 profile 阶段遇到固件信号量超时。"),
    P("<b>统一约束：</b>所有有效结果均保持 custom all-reduce 开启，不使用 <font name='Courier'>--disable-custom-all-reduce</font>。", "CNNote"),
    PageBreak(),
]

# Page 2: methodology and external baseline.
story += [section("二", "测试口径、证据等级与可比性")]
method = [
    ["项目", "定义", "报告用法"],
    ["Decode Avg TPS", "1000 / mean TPOT(ms)，单请求平均解码速率", "与外部对照表同口径"],
    ["聚合输出吞吐", "整个服务实例的 output tokens/s", "高并发容量排序与屋顶比较"],
    ["Median TTFT", "请求到首 token 的中位时间", "评估吞吐换延迟的代价"],
    ["正式测试", "C1/C16 各 64 requests，failed=0", "9B、30B 最终 C1/C16"],
    ["短测", "C1 通常 16、C16 通常 32、C32 为 64 requests", "参数筛选；不声称统计显著性"],
]
story += [styled_table(method, [100, 255, 152]), Spacer(1, 8)]
story += [P("<b>外部对照的可比性：</b>35B 拓扑与负载一致；9B 的选定方案由 TP2 改为双 TP1/DP2；30B 由 EP+TP2 改为纯 TP2；80B 输入/输出和 TP4 一致。因此对照用于工程方案评估，不是严格的单变量实验。", "CNNote")]

rows = [["模型", "并发", "TTFT 选定", "TTFT 对照", "TTFT 变化", "Decode 选定", "Decode 对照", "TPS 变化", "聚合吞吐"]]
names = {"35B": "Qwen3.6-35B-A3B", "9B": "Qwen3.5-9B", "30B": "Qwen3-30B-A3B", "80B": "Qwen3-Next-80B-A3B"}
for m in ("35B", "9B", "30B", "80B"):
    for conc in ("c1", "c16"):
        s, r = DATA["selected"][m][conc], DATA["reference"][m][conc]
        dt = (s["ttft"] / r["ttft"] - 1) * 100
        dd = (s["decode"] / r["decode"] - 1) * 100
        rows.append([
            names[m] if conc == "c1" else "", conc.upper(), f"{s['ttft']:.2f}", f"{r['ttft']:.2f}", pct_text(dt, True),
            f"{s['decode']:.2f}", f"{r['decode']:.2f}", pct_text(dd), f"{s['aggregate']:.2f}",
        ])
result_table = styled_table(rows, [92, 28, 58, 58, 52, 58, 58, 49, 54], font_size=6.2,
                            spans=[((0,1),(0,2)),((0,3),(0,4)),((0,5),(0,6)),((0,7),(0,8))],
                            aligns={1:"CENTER",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT",6:"RIGHT",7:"RIGHT",8:"RIGHT"},
                            row_bgs={2:LIGHT,4:LIGHT,6:LIGHT,8:LIGHT})
story += [Spacer(1, 8), result_table,
          P("方向说明：TTFT 负变化为改善，Decode TPS 正变化为改善。9B 与 30B 是吞吐优先方案，因此接受了部分 TTFT 代价。", "CNNote"),
          PageBreak()]

# Page 3: charts and C32 scaling.
story += [section("三", "对照图表与 C32 扩展结果"),
          Image(str(FIG / "figure-01-reference-comparison.png"), width=7.05*inch, height=4.48*inch),
          Spacer(1, 5),
          P("图 1：同并发口径下的 Decode Avg TPS 和 Median TTFT。不同模型的输入长度与卡数不同，不以绝对数值直接推导模型优劣。", "CNSmall")]

c32_rows = [["模型", "C16 聚合", "C32 聚合", "扩展收益", "C32 TTFT", "C32 Decode", "完成/失败"]]
for m in ("35B", "9B", "30B", "80B"):
    a, b = DATA["selected"][m]["c16"], DATA["selected"][m]["c32"]
    c32_rows.append([names[m], f"{a['aggregate']:.2f}", f"{b['aggregate']:.2f}",
                     pct_text((b['aggregate']/a['aggregate']-1)*100), f"{b['ttft']:.2f}",
                     f"{b['decode']:.2f}", f"{b['completed']}/{b['failed']}"])
story += [Spacer(1, 7), styled_table(c32_rows, [112, 72, 72, 67, 72, 62, 50], font_size=6.8,
                                     aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT",6:"CENTER"},
                                     row_bgs={3:PALE_GREEN,4:PALE_GREEN}),
          P("C32 适合容量规划，但不应替代 C16 延迟评估。9B 的 C32 TTFT 增至 8.81 s，而 80B 在吞吐 +32.6% 时 TTFT 仅较 C16 增加约 17.1%，扩展更理想。", "CNNote"),
          PageBreak()]

# Page 4: 9B.
story += [section("四", "Qwen3.5-9B 调优矩阵"), tuning_table("9B"), Spacer(1, 8),
          P("<b>主要因果链：</b>MTP3 相比初始 MTP1 提升解码路径；batch2048→4096 进一步提高 C16；自动 KV 主要改善容量余量；最大收益来自将一张物理卡的两个逻辑设备变为双 TP1/DP2 副本。"),
          P("<b>失败与修正：</b>首次 C32 在 DP2 下将 max-num-seqs=32 应用到每个副本，不均匀调度触发 Fantasy SSM 的 ssm_state_indices 非连续布局错误。将每副本上限修正为 16 后，C32 64/64 请求完成。", "CNWarn"),
          P("<b>最终判断：</b>C16 是交互与吞吐的平衡档，C32 仅用于吞吐优先队列。继续扩大 Graph 的预期收益很小，后续应优先比较内置 DP2 与外部双端口轮询。"),
          Spacer(1, 8), Image(str(FIG / "figure-04-tuning-paths.png"), width=7.0*inch, height=4.55*inch),
          P("图 2：四模型主要调优组的 C16 聚合输出吞吐。绿色为最终选定，红色为 EP 或后端回归组。", "CNSmall"),
          PageBreak()]

# Page 5: 30B.
story += [section("五", "Qwen3-30B-A3B 调优矩阵"), tuning_table("30B"), Spacer(1, 8),
          P("<b>有效路径：</b>Graph 由单一 32 扩为 4/8/16/32 显著改善 C1 图命中；batch4096 提升 C16；async 在当前负载下进一步提升聚合吞吐。"),
          P("<b>未采用路径：</b>block800→1088 基本无收益；自动 KV 的 C16/C32 方向不一致；batch6144 仅小幅提吞吐但 TTFT 显著增加；Triton grouped GEMM 的 C32 仅比 auto 高 0.14%，低于 3% 采用阈值。"),
          P("<b>EP2 结论：</b>C16/C32 回落至 185.86/230.52 tok/s，比选定方案低约 17.6%/28.2%。当前专家 dispatch/combine、all-to-all 和同步开销高于专家计算节省。", "CNWarn"),
          P("<b>最终判断：</b>保留 auto MoE backend、block800、batch4096、8 GiB KV 和 async。C32 的 320.90 tok/s 是四个模型中最高的单物理卡吞吐。"),
          PageBreak()]

# Page 6: 35B.
story += [section("六", "Qwen3.6-35B-A3B 调优矩阵"), tuning_table("35B"), Spacer(1, 8),
          P("<b>关键收益：</b>MTP1→MTP3 建立基础增益，MTP3→MTP4 在同为 batch2048 时将 C16 从 164.38 提高至 175.57 tok/s；MTP4 下 batch2048→4096 再提高至 180.64 tok/s，同时降低 TTFT。"),
          P("<b>边际收益已收敛：</b>Graph128 相对 Graph64 的 C16 只有约 0.5% 差异；batch6144 在 C32 只高 0.48%，同时 C16 下降且 TTFT 增加，因此不改变决策。"),
          P("<b>MoE/EP 路径：</b>Triton grouped GEMM + 专家融合回落至 167.69/188.63 tok/s；EP2 linear 为 167.81/185.50 tok/s。round-robin 专家放置在当前插件中未得到真正支持，无法构成有效 A/B。", "CNWarn"),
          P("<b>最终判断：</b>在当前后端版本中，继续调 batch/Graph 的预期收益低于 3%。要获得两位数改善，需要原生融合 token sorting、grouped GEMM、expert combine 和 MTP verify kernel。"),
          PageBreak()]

# Page 7: 80B.
story += [section("七", "Qwen3-Next-80B-A3B-Instruct 调优矩阵"), tuning_table("80B"), Spacer(1, 8),
          P("<b>MTP 是决定性因素：</b>无 MTP→MTP1 将 C16 从 177.71 提高至 231.30 tok/s；MTP2 进一步提高至 251.49；MTP3 达 255.93 tok/s，且 C1 达 47.13 tok/s。"),
          P("<b>Graph/KV/调度：</b>Graph32/64 主要改善 C1 和小批命中；batch4096 可作 MTP1 的低 TTFT 档，但不是吞吐最优；async 会降低 C16 并明显恶化 TTFT；自动 KV 在初始化阶段失败，最终保留 8 GiB。"),
          P("<b>EP 结果：</b>TP4+EP4 可完成测试，但 C16 仅 233.93 tok/s，比 TP4/MTP3 低 8.6%，TTFT 增加约 21.2%。TP2×DP2×EP2 在修正 DP 设备类型后越过原始错误，但 profile run 仍触发 Fantasy firmware semaphore 超时。", "CNWarn"),
          P("<b>最终判断：</b>TP4/MTP3 为高吞吐首选。C32 达 339.26 tok/s，但占用两张物理卡，折合 169.63 tok/s/卡。"),
          PageBreak()]

# Page 8: roof and bottleneck synthesis.
story += [section("八", "理论屋顶、资源效率与瓶颈"), P("C1：单请求解码效率", "CNSub")]
c1_roof_rows = [["模型", "C1 理论屋顶", "C1 聚合", "C1 Decode", "系统屋顶效率", "纯 Decode 效率"]]
for m in ("35B", "9B", "30B", "80B"):
    roof = DATA["roof"][m]["c1"]
    s = DATA["selected"][m]["c1"]
    c1_roof_rows.append([names[m], f"{roof:.0f}", f"{s['aggregate']:.2f}", f"{s['decode']:.2f}",
                         f"{100*s['aggregate']/roof:.1f}%", f"{100*s['decode']/roof:.1f}%"])
story += [styled_table(c1_roof_rows, [126, 85, 71, 71, 78, 76], font_size=6.8,
                       aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT"},
                       row_bgs={2:PALE_GREEN}), Spacer(1, 4), P("C16/C32：高并发吞吐效率", "CNSub")]
roof_rows = [["模型", "理想批处理屋顶", "C16 聚合", "C16 效率", "C32 聚合", "C32 效率", "C32 每卡"]]
for m in ("35B", "9B", "30B", "80B"):
    roof = DATA["roof"][m]["ideal"]
    s = DATA["selected"][m]
    roof_rows.append([names[m], f"{roof:.0f}", f"{s['c16']['aggregate']:.2f}", f"{100*s['c16']['aggregate']/roof:.1f}%",
                      f"{s['c32']['aggregate']:.2f}", f"{100*s['c32']['aggregate']/roof:.1f}%",
                      f"{s['c32']['aggregate']/s['cards']:.2f}"])
story += [styled_table(roof_rows, [113, 94, 65, 57, 65, 57, 58], font_size=6.8,
                       aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT",6:"RIGHT"},
                       row_bgs={3:PALE_GREEN}), Spacer(1, 5),
          Image(str(FIG / "figure-03-roof-efficiency.png"), width=7.0*inch, height=2.28*inch),
          P("图 3：C1 同时展示端到端聚合输出效率和纯 Decode 效率；C16/C32 使用整卡理想批处理屋顶。屋顶值是规划估算，不是硬件计数器利用率。", "CNSmall")]
bottlenecks = [
    ["模型", "主要瓶颈", "已验证的优化", "下一阶段高收益方向"],
    ["9B", "DP 副本负载均衡、prefill/decode 队列气泡", "双 TP1/DP2、MTP4", "外部双端口轮询与 NUMA 绑定"],
    ["30B", "专家小 GEMM、路由后调度", "纯 TP2、batch4096、async", "原生 grouped GEMM + dispatch/combine 融合"],
    ["35B", "MoE kernel、MTP verify、专家 token 分布", "MTP4、batch4096", "token sorting + grouped GEMM + expert combine 后端融合"],
    ["80B", "跨卡 TP 通信、线性注意力、调度开销", "MTP3、Graph64", "DP/EP 设备适配、固件同步、MTP verify 优化"],
]
story += [Spacer(1, 8), styled_table(bottlenecks, [48, 141, 139, 179], font_size=6.2),
          P("总体判断：30B 在 C32 的理想屋顶效率最高；80B 虽有最高聚合吞吐，但每卡吞吐低于 9B/30B/35B。35B/80B 的下一轮显著提升需要后端或固件级工作，而不是继续扩大 batch/Graph。", "CNNote"),
          PageBreak()]

# Page 9: final selected solution and exact outcomes.
story += [section("九", "最终选定方案与测试结果")]
config_rows = [["模型", "物理卡", "最终配置", "证据"]]
for m in ("35B", "9B", "30B", "80B"):
    s = DATA["selected"][m]
    config_rows.append([names[m], str(s["cards"]), s["config"], s["evidence"]])
story += [styled_table(config_rows, [112, 46, 284, 65], font_size=6.2, row_bgs={2:PALE_GREEN,3:PALE_GREEN}), Spacer(1, 9)]

final_rows = [["模型", "并发", "Median TTFT(ms)", "Decode Avg TPS", "聚合输出 TPS", "完成/失败", "定位"]]
loc = {"c1":"单请求", "c16":"平衡吞吐", "c32":"峰值吞吐"}
for m in ("35B", "9B", "30B", "80B"):
    for conc in ("c1", "c16", "c32"):
        s = DATA["selected"][m][conc]
        final_rows.append([names[m] if conc == "c1" else "", conc.upper(), f"{s['ttft']:.2f}", f"{s['decode']:.2f}",
                           f"{s['aggregate']:.2f}", f"{s['completed']}/{s['failed']}", loc[conc]])
spans = [((0,1),(0,3)),((0,4),(0,6)),((0,7),(0,9)),((0,10),(0,12))]
story += [styled_table(final_rows, [104, 34, 82, 75, 80, 65, 67], font_size=6.2, spans=spans,
                       aligns={1:"CENTER",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"CENTER"},
                       row_bgs={3:PALE_BLUE,6:PALE_BLUE,9:PALE_GREEN,12:PALE_GREEN}), Spacer(1, 8),
          Image(str(FIG / "figure-02-c16-c32-throughput.png"), width=7.0*inch, height=2.77*inch),
          P("图 4：最终配置在 C16/C32 下的聚合输出吞吐及 C32 每物理卡归一化值。", "CNSmall"),
          P("<b>最终选型：</b>若以单卡吞吐与成本为主，优先 30B，其次 9B；若必须使用 35B，保留 MTP4 且不开 EP；若以最大模型能力为主，选 80B TP4/MTP3，但应按两卡成本评估。", "CNNote"),
          PageBreak()]

# Page 10: RTX 5090 comparison supplied by the user.
story += [section("十", "与 RTX 5090 的吞吐对比"),
          P("本节仅保留截图中提供的 Qwen3.6-35B-A3B 与 Qwen3.5-9B 两组数据，统一以 C16 每卡吞吐展示。35B 的 RTX 5090 值是 8 卡总吞吐的每卡归一值，不代表单张 5090 脱离 TP2 后可独立复现。", "CNNote"),
          Image(str(FIG / "figure-05-rtx5090-comparison.png"), width=7.0*inch, height=2.35*inch),
          P("图 5：GR308 与 RTX 5090 的 C16 每卡吞吐。35B 使用每卡归一值；9B 为单卡同并发数据。", "CNSmall"),
          P("数据对照与有效性", "CNSub")]

rtx_rows = [["模型", "并发", "GR308 单卡吞吐", "RTX 5090 每卡吞吐", "GR/5090", "数据有效性"]]
for m in ("35B", "9B"):
    r = DATA["rtx5090"][m]
    rtx_rows.append([names[m], r["concurrency"], f"{r['gr308_tps']:.2f}",
                     f"{r['rtx5090_per_card_tps']:.2f}", f"{r['ratio_pct']:.2f}%", r["validity"]])
story += [styled_table(rtx_rows, [87, 42, 76, 83, 60, 159], font_size=6.2,
                       aligns={1:"CENTER",2:"RIGHT",3:"RIGHT",4:"RIGHT"},
                       row_bgs={1:PALE_ORANGE,2:PALE_GREEN}),
          Spacer(1, 9),
          P("<b>Qwen3.6-35B-A3B：</b>GR308 为 180.64 tok/s；RTX 5090 每卡归一值为 3990.23 ÷ 8 = 498.78 tok/s；GR/5090 为 <b>36.22%</b>。原 5090 测试采用 4×TP2，因此这个比例适合做八卡系统的每卡产出参考，不适合解释为两种 GPU 的独立单卡性能比。"),
          P("<b>Qwen3.5-9B：</b>GR308 为 247.16 tok/s；RTX 5090 为 444.40 tok/s；GR/5090 为 <b>55.62%</b>。两者均为 BF16、单卡、C16，但 5090 使用 ShareGPT、最大输出 256，GR308 使用固定输入/输出 3072/1024，因此仍存在负载差异。"),
          P("<b>结论：</b>在当前对照中，9B 的数据可比性明显高于 35B。35B 应标注为“多卡结果的每卡归一比较”；若用于采购或部署判断，还需在单张 RTX 5090 上使用同一权重、同一 C16 负载和相同输入/输出长度复测。", "CNNote"),
          P("5090 数据来源：用户提供的对照截图；35B 原始总吞吐 3990.23 tok/s（8 张 RTX 5090，4×TP2），9B 单卡 C16 吞吐 444.40 tok/s。", "CNSmall"),
          PageBreak()]

# Page 11: operating boundaries, exceptions, and evidence paths.
story += [section("十一", "运行边界、异常、验收条件与证据索引"),
          P("<b>运行边界：</b>精度均为 BF16；max model length=4096；关闭 prefix caching；默认同步调度，仅 30B 最终选 async；custom all-reduce 始终保持开启。"),
          P("<b>已知异常：</b>并发初始化多个 FHCCL/RCCL communicator 时曾出现通信初始化失败，改为上一服务 API ready 后再启动下一通信域后消失。80B 自动 KV 初始化失败；35B C64 和 9B 首次 C32 触发 Fantasy SSM 的非连续索引问题。"),
          P("<b>硬件状态：</b>各轮测试后 8 个逻辑设备健康状态为 OK，最终测试后显存回落至约 540 MiB/逻辑设备。"),
          P("<b>统计限制：</b>参数矩阵主要是单轮短测，没有 3 轮以上的配对重复、标准差或置信区间。本报告只做描述性工程判断；小于 3% 的差异视为基本持平，不宣称统计显著。", "CNWarn"),
          P("<b>上线前验收：</b>35B MTP4 和 80B MTP3 应在固定 seed、物理卡、CPU NUMA 亲和性和启动顺序后各执行至少 3 轮 64-request 复测；报告跨轮变异系数，并确认请求均完整输出 1024 tokens、failed=0。"),
          P("<b>主要远端数据路径：</b><br/>"
            "/data/infra/benchmarks/gr308-adaptive-short-matrix-20260915/<br/>"
            "/data/infra/benchmarks/gr308-final-high-throughput-20260915/<br/>"
            "/data/infra/benchmarks/gr308-35b-mtp4-short-matrix-ready-20260915/<br/>"
            "/data/infra/benchmarks/gr308-80b-short-tuning-20260915/<br/>"
            "/data/infra/benchmarks/gr308-moe-c32-ep-short-20260917-attempt4/<br/>"
            "/data/infra/benchmarks/gr308-final-c16-c32-last-mile-20260917/<br/>"
            "/data/infra/benchmarks/gr308-final-35b-c32-recheck-20260917/<br/>"
            "/data/infra/benchmarks/gr308-35b-30b-native-ep2-20260917/<br/>"
            "/data/infra/benchmarks/gr308-80b-tp4-ep4-20260917/<br/>"
            "/data/infra/benchmarks/gr308-9b-80b-c32-short-20260917/<br/>"
            "/data/infra/benchmarks/gr308-9b-c32-retry-20260917/", "CNSmall"),
          P("<b>报告结论：</b>本轮参数调优已基本收敛。当前最具部署性价比的是 30B 和 9B；35B 与 80B 已证明 MTP 的明确价值，但若要继续显著提升，需转向 MoE/MTP 内核、跨卡通信与固件同步层。", "CNNote"),
          P("数据来源：服务器原生 benchmark.json、result.json、launch.json、server.log 和 FANTASY SMI 状态；整理日期 2026-09-17。", "CNSmall")]

doc.build(story)
print(OUT)
