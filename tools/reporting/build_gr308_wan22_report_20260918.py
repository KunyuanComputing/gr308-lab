#!/usr/bin/env python3
"""Build the 2026-09-18 GR308 Wan2.2 tuning report."""

from pathlib import Path
import json
import statistics

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "output" / "风华_GR308_Wan2.2视频生成性能测试与调优报告_20260918"
DATA = json.loads((REPORT_DIR / "data" / "report_data.json").read_text(encoding="utf-8"))
FIG = REPORT_DIR / "figures"
OUT = ROOT / "output" / "pdf" / "风华_GR308_Wan2.2视频生成性能测试与调优报告_20260918.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(TTFont("Heiti", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("HeitiBold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))

BLUE = HexColor("#1675C1")
NAVY = HexColor("#0D2B4D")
GREEN = HexColor("#168A68")
ORANGE = HexColor("#D77A00")
RED = HexColor("#C74444")
GRAY = HexColor("#626A73")
LIGHT = HexColor("#F3F6FA")
GRID = HexColor("#C6D2DF")
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
styles.add(ParagraphStyle(name="Cell", fontName="Heiti", fontSize=7.0, leading=9.2))
styles.add(ParagraphStyle(name="CellSmall", fontName="Heiti", fontSize=6.2, leading=8.1))
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
            cell if hasattr(cell, "wrap") else P(
                cell,
                "CellHead" if ri == 0 else ("CellSmall" if font_size < 6.8 else "Cell"),
            )
            for cell in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if repeat else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E9EEF4")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#9FB2C6")),
    ]
    for row, bg in (row_bgs or {}).items():
        commands.append(("BACKGROUND", (0, row), (-1, row), bg))
    for start, end in spans or []:
        commands.append(("SPAN", start, end))
    for col, align in (aligns or {}).items():
        commands.append(("ALIGN", (col, 1), (col, -1), align))
    table.setStyle(TableStyle(commands))
    return table


def page_decor(canvas, doc):
    canvas.saveState()
    width, _ = LETTER
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(52, 35, width - 52, 35)
    canvas.setFont("Heiti", 7.3)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(width - 52, 20, f"风华 GR308 | Wan2.2 性能测试与调优 | 2026-09-18   {doc.page}")
    canvas.restoreState()


def result(name):
    return next(row for row in DATA["short_results"] if row["id"] == name)


def kpi_cards():
    values = [
        ("短测 E2E 降幅", "43.5%", "S2 vs FP32"),
        ("VAE decode 加速", "1.82x", "1144.41 s -> 630.31 s"),
        ("S2 首帧 SSIM", "0.9743", "单 prompt / seed"),
    ]
    rows = [[P(x[0], "CNSmall") for x in values], [
        P(f"<font size='16' color='#0D2B4D'><b>{value}</b></font><br/><font size='8'>{detail}</font>", "CNBody")
        for _, value, detail in values
    ]]
    table = Table(rows, colWidths=[169, 169, 169], rowHeights=[20, 54])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


doc = BaseDocTemplate(
    str(OUT),
    pagesize=LETTER,
    leftMargin=52,
    rightMargin=52,
    topMargin=50,
    bottomMargin=48,
    title="风华 GR308 Wan2.2 视频生成性能测试与调优报告",
    author="Codex",
    subject="Wan2.2-TI2V-5B SGLang Diffusers 短测、混合精度、质量与完整负载诊断",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page_decor))
story = []

# Page 1: cover and executive summary.
story += [
    P("性能测试数据研究报告", "CNSmall"),
    P("风华 GR308 Wan2.2 视频生成性能测试与调优报告", "CNTitle"),
    P("SGLang Diffusers | 1280×704 | VAE 混合精度 | 质量验证 | 121 帧完整负载诊断", "CNSubtitle"),
    P("<b>测试对象：</b>Wan2.2-TI2V-5B-Diffusers。"),
    P("<b>平台：</b>8 张 GR308 物理卡 / 16 个逻辑设备；单物理卡 56 GiB；驱动 1.1.0.21。"),
    P("<b>短测：</b>TP2、请求 1 帧但运行时实际调整为 5 帧、3 steps、固定 prompt/seed。"),
    Spacer(1, 7),
    kpi_cards(),
    Spacer(1, 13),
    section("一", "结论摘要"),
    P("<b>结论 1｜S2 是当前短测最优候选。</b>端到端从 1190.20 s 降至 <b>672.33 s</b>，VAE decode 从 1144.41 s 降至 <b>630.31 s</b>。"),
    P("<b>结论 2｜S2 首帧质量接近 FP32 工程参考。</b>PSNR 39.36 dB、luma SSIM 0.9743、RGB SSIM 0.9617；但样本量只有一帧。"),
    P("<b>结论 3｜全局 FP16 必须拒绝。</b>尽管 E2E 降低约 68%，但 PSNR 仅 14.25 dB，并出现确定性条带、块状断裂和颜色漂移。"),
    P("<b>结论 4｜VAE decode 是绝对瓶颈。</b>FP32 基线中占 E2E 约 96.2%，普通驻留和拓扑调整不足以改变该结构。"),
    P("<b>结论 5｜121 帧/20 步尚未跑通。</b>已确认 firmware semaphore timeout、异常 HIP OOM 和 SP8 backend 能力限制，因此不能报告同口径 GR308/RTX 5090 倍率。", "CNWarn"),
    PageBreak(),
]

# Page 2: methodology and baseline.
story += [section("二", "测试口径、证据等级与基线")]
method = [
    ["项目", "短测", "完整目标"],
    ["分辨率", "1280×704", "1280×704"],
    ["帧数", "CLI 1；运行时实际 5", "121"],
    ["Denoise steps", "3", "20"],
    ["采样", "UniPC / guidance 5 / seed 42", "UniPC / shift 5 / guidance 5 / seed 42"],
    ["证据", "成功输出 + perf.json + 首帧", "目前无成功 perf.json"],
]
story += [styled_table(method, [105, 185, 217]), Spacer(1, 8)]
story += [
    P("<b>证据分级：</b>成功且具有完整 perf.json 的结果用于性能比较；完成首帧全参考指标的结果可作为质量候选；未成功的容器日志只用于诊断。"),
    P("<b>关键口径修正：</b>短测命令中的 num_frames=1 被 SGLang 在双卡条件下调整为 5 帧。报告统一按“有效 5 帧短测”描述，避免误写成单帧性能。", "CNNote"),
    P("基线复现性", "CNSub"),
]
totals = DATA["baseline_repeats"]["total_seconds"]
decodes = DATA["baseline_repeats"]["decode_seconds"]
baseline_rows = [
    ["指标", "均值/s", "标准差/s", "最小/s", "最大/s"],
    ["E2E", f"{statistics.mean(totals):.2f}", f"{statistics.pstdev(totals):.2f}", f"{min(totals):.2f}", f"{max(totals):.2f}"],
    ["VAE decode", f"{statistics.mean(decodes):.2f}", f"{statistics.pstdev(decodes):.2f}", f"{min(decodes):.2f}", f"{max(decodes):.2f}"],
]
story += [styled_table(baseline_rows, [120, 95, 95, 95, 102], aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT"}),
          Spacer(1, 9), Image(str(FIG / "figure-01-stage-breakdown.png"), width=7.05*inch, height=3.44*inch),
          P("图 1：四个代表配置的阶段耗时。即使采用 S2，VAE decode 仍占主要部分。", "CNSmall"),
          PageBreak()]

# Page 3: optimization matrix.
story += [section("三", "混合精度优化矩阵")]
base_decode = result("FP32 baseline")["decode_seconds"]
rows = [["方案", "精度边界", "E2E/s", "Decode/s", "降幅", "峰值/MiB", "判断"]]
ids = ["FP32 baseline", "Global FP16", "M0", "M1", "M2", "M3", "S1", "S2"]
row_bgs = {2: PALE_RED, 7: PALE_GREEN, 8: PALE_GREEN}
for name in ids:
    row = result(name)
    reduction = "-" if name == "FP32 baseline" else f"{100*(1-row['decode_seconds']/base_decode):.1f}%"
    verdict = {
        "FP32 baseline":"参考", "Global FP16":"质量拒绝", "M0":"收益小", "M1":"收益有限",
        "M2":"有效", "M3":"质量候选", "S1":"有效", "S2":"短测最优",
    }[name]
    rows.append([name, row["configuration"], f"{row['total_seconds']:.2f}", f"{row['decode_seconds']:.2f}", reduction,
                 f"{row['peak_allocated_mb']:.0f}", verdict])
story += [styled_table(rows, [61, 190, 52, 57, 43, 58, 58], font_size=6.2, row_bgs=row_bgs,
                       aligns={2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT"}),
          Spacer(1, 8),
          P("<b>主要规律：</b>M0/M1 的收益接近基线波动；M2 开始进入有效区间；M3、S1、S2 表明高分辨率 up block 与最终 residual unit 是主要性能来源。"),
          P("<b>精度边界：</b>继续将倒数第 3 个高分辨率 residual unit 切到 FP16，会触发 5.80 GiB Conv3D 工作区分配失败。", "CNWarn"),
          Spacer(1, 7), Image(str(FIG / "figure-02-mixed-precision-path.png"), width=7.0*inch, height=3.42*inch),
          P("图 2：VAE 混合精度路径。S2 的 decode 降幅为 44.9%。", "CNSmall"),
          PageBreak()]

# Page 4: quality.
story += [section("四", "生成质量验证")]
quality = DATA["quality_results"]
qrows = [["方案", "PSNR/dB", "Luma SSIM", "RGB SSIM", "MAE/255", "P95 误差", "判断"]]
for name in ("Global FP16", "M3", "S2"):
    q = quality[name]
    qrows.append([name, f"{q['psnr_db']:.3f}", f"{q['ssim_luma']:.4f}", f"{q['ssim_rgb_mean']:.4f}",
                  f"{q['mae_0_255']:.2f}", f"{q['p95_abs_error']:.0f}", "拒绝" if name == "Global FP16" else "候选"])
story += [styled_table(qrows, [88, 67, 72, 68, 68, 67, 66], row_bgs={1:PALE_RED, 2:PALE_GREEN, 3:PALE_GREEN},
                       aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT"}),
          Spacer(1, 7), Image(str(FIG / "figure-03-quality-metrics.png"), width=7.0*inch, height=2.92*inch),
          P("图 3：相对 FP32 工程参考的首帧全参考指标。", "CNSmall"),
          Spacer(1, 7), Image(str(FIG / "figure-04-quality-triptych.png"), width=7.0*inch, height=1.79*inch),
          P("图 4：同 prompt、seed 的 FP32、Global FP16 与 S2 首帧。Global FP16 存在明显结构与颜色损坏。", "CNSmall"),
          P("<b>解释边界：</b>只有一个 prompt、一个 seed 和一个对齐首帧；没有覆盖闪烁、运动一致性、时序漂移与跨场景稳定性。", "CNNote"),
          PageBreak()]

# Page 5: full workload failures.
story += [section("五", "121 帧/20 步完整负载状态")]
frows = [["路径", "负载", "失败表现", "工程判断"]]
for failure in DATA["confirmed_failures"]:
    frows.append([failure["id"], failure["workload"], failure["failure"], failure["interpretation"]])
story += [styled_table(frows, [92, 106, 174, 135], font_size=6.2, row_bgs={1:PALE_RED,2:PALE_RED,3:PALE_RED,4:PALE_RED,5:PALE_ORANGE}),
          Spacer(1, 9),
          P("<b>TP2：</b>文本编码结束后，在长序列 denoise 阶段触发 DMA/firmware semaphore 30 s timeout。"),
          P("<b>SP2/SP4：</b>分别把失败申请缩小到 160 MiB 和 80 MiB，但在仍报告约 23 GiB 空闲时 OOM，说明问题不能用“显存容量不够”简单解释。"),
          P("<b>SP8：</b>native Diffusers fallback 无法执行请求的 SP/ring 拓扑，同时实验 Fantasy adapter 只允许 Torch SDPA，运行前即失败。"),
          P("多个 sequence-shard、offload 和 allocator 变体未形成有效 perf.json，因此不计入性能矩阵。", "CNWarn"),
          Spacer(1, 7),
          P("外部 RTX 5090 参考", "CNSub"),
          P("既有记录为 8×RTX 5090、1280×704、121 帧、3 步：U8+CFG1 57.206 s，U4+CFG2 37.222 s，后者快 1.5369×。GR308 尚无同口径完成结果，不能计算速度比。", "CNNote"),
          PageBreak()]

# Page 6: operator investigation.
story += [section("六", "高分辨率 Conv3D 深挖")]
deep_rows = [
    ["尝试", "卡数", "变化", "结果"],
    ["局部 autocast", "2", "仅目标 residual 的两次 Conv3D FP16", "OOM"],
    ["权重预转换", "2", "提前缓存 FP16 Conv3D 权重", "OOM"],
    ["扩大空间切分", "4", "TP4，缩小单卡空间分片", "OOM；denoise 57 s -> 86.9 s"],
    ["整块 FP16", "2", "倒数第 3 个 residual 整体 FP16", "OOM"],
]
story += [styled_table(deep_rows, [106, 50, 235, 116], row_bgs={1:PALE_RED,2:PALE_RED,3:PALE_RED,4:PALE_RED}),
          Spacer(1, 9),
          P("四种写法都在同一高分辨率阶段失败。最完整日志显示：单次申请 <b>5.80 GiB</b>，GPU 报告仍有约 <b>28.92 GiB</b> 空闲，却返回 HIP OOM。"),
          P("这更符合 FP16 Conv3D 算法的大工作区、PAL 虚拟地址分配或后端 allocator 交互问题，而不是模型静态权重无法容纳。"),
          P("从 GR308 文档可确认的工具只有 HIP 默认内存池、hipMallocAsync/hipFreeAsync、hipMemPoolTrimTo、hipMemGetInfo，以及 FHCCL 的 ALLOC/GRAPH/TUNING 日志。文档没有给出 Conv3D workspace limit、算法枚举或 PYTORCH_FANT_ALLOC_CONF 的正式说明。", "CNNote"),
          P("因此下一步应从参数调优转向：带 halo 的空间 tiled Conv3D、厂商低 workspace 算法、PAL/allocator 最小复现与固件修复。"),
          Spacer(1, 14),
          P("当前可运行边界", "CNSub"),
          P("S2 保留高风险高分辨率 residual、归一化和输出头的 FP32 路径，仅将已验证的 up blocks 与最后两个低风险 residual unit 下沉至 FP16。进一步扩大 FP16 范围尚无可运行证据。"),
          PageBreak()]

# Page 7: recommendation and launch outline.
story += [section("七", "推荐配置、部署判断与路线")]
recommend = [
    ["用途", "建议", "状态"],
    ["短测回归", "S2：FP16_UP_BLOCKS=3，FP16_FINAL_RESNETS=-2", "推荐"],
    ["全局 FP16", "禁止；确定性质量损坏", "拒绝"],
    ["121 帧生产", "等待 firmware、allocator 和 SP backend 问题修复", "未就绪"],
    ["扩卡", "不要假设卡数越多越快；TP4 已显示通信回归", "需单独验证"],
]
story += [styled_table(recommend, [100, 305, 102], row_bgs={1:PALE_GREEN,2:PALE_RED,3:PALE_RED,4:PALE_ORANGE}),
          Spacer(1, 8),
          P("S2 关键配置", "CNSub"),
          P("<font name='Courier'>SGLANG_WAN_VAE_FP16_UP_BLOCKS=3</font><br/><font name='Courier'>SGLANG_WAN_VAE_FP16_FINAL_RESNETS=-2</font><br/><font name='Courier'>--vae-precision fp32 --vae-decode-precision fp16</font><br/><font name='Courier'>--vae-config.use-parallel-decode true --vae-config.parallel-decode-mode auto</font>"),
          P("该配置依赖定制 Wan VAE 混合精度实现；只设置原生 vae-decode-precision=fp16 会进入全局 FP16，不等价于 S2。", "CNWarn"),
          P("优先级", "CNSub")]
priorities = [
    ["P0", "构造 FP16 Conv3D 最小复现，确认 5.80 GiB workspace 来源与可配置上限"],
    ["P0", "修复 121 帧 denoise 的 firmware semaphore timeout"],
    ["P1", "实现带 halo 的 tiled Conv3D，并做逐层数值验证"],
    ["P1", "用 hipMemGetInfo 与 FHCCL ALLOC/GRAPH/TUNING 日志区分 allocator/PAL/通信"],
    ["P2", "完整负载打通后再跑 U8+CFG1、U4+CFG2、TP/SP/CFG 正式矩阵"],
    ["P2", "扩展多 prompt、多 seed、121 帧时序质量与人工盲检"],
]
story += [styled_table([["优先级", "工作项"]] + priorities, [64, 443], row_bgs={1:PALE_RED,2:PALE_RED,3:PALE_ORANGE,4:PALE_ORANGE}),
          Spacer(1, 8),
          P("<b>最终部署判断：</b>当前 GR308 软件栈能够完成 Wan2.2 短视频短测并通过选择性混合精度显著加速，但尚未证明可稳定承载目标 121 帧/20 步任务。", "CNNote"),
          PageBreak()]

# Page 8: limitations and reproducibility.
story += [section("八", "限制与复现索引"),
          P("<b>统计限制：</b>性能消融是工程短测；质量样本 n=1；没有置信区间或显著性检验。"),
          P("<b>指标限制：</b>PSNR/SSIM 是相对 FP32 工程参考的一致性，不代表绝对审美质量或文本一致性。"),
          P("<b>系统限制：</b>失败可能同时涉及 SGLang distributed fallback、Fantasy adapter、PyTorch allocator、PAL VA manager 和固件同步；现有日志不足以唯一归因。"),
          P("<b>外推限制：</b>5 帧/3 步短测不能外推到 121 帧/20 步；没有有效的 GR308/RTX 5090 同口径比例。", "CNWarn"),
          P("复现索引", "CNSub")]
index_rows = [
    ["内容", "位置"],
    ["结构化数据", "output/风华_GR308_Wan2.2视频生成性能测试与调优报告_20260918/data/report_data.json"],
    ["可编辑正文", "output/风华_GR308_Wan2.2视频生成性能测试与调优报告_20260918/*.md"],
    ["质量原图与视频", "output/wan22_quality/"],
    ["质量计算脚本", "tools/compare_wan22_quality.py"],
    ["VAE 混合精度实现", "tools/fantasy_wanvae_mixed.py"],
    ["短测启动脚本", "tools/run_wan22_matrix_job.sh"],
]
story += [styled_table(index_rows, [118, 389], font_size=6.2), Spacer(1, 12),
          P("报告中的性能数字均来自成功 perf.json；失败表来自对应容器日志。未成功产生 perf.json 的配置不参与排名。", "CNNote")]

doc.build(story)
print(OUT)
