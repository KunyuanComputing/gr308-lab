from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

models = ["35B", "9B", "30B", "80B"]
ref_c1 = np.array([52.91, 45.09, 26.85, 33.23])
best_c1 = np.array([50.745855, 54.639813, 26.434389, 48.522532])
ref_c16 = np.array([12.74, 11.97, 11.70, 15.02])
best_c16 = np.array([13.934548, 18.904010, 15.223948, 19.707880])

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180, constrained_layout=True)
x = np.arange(len(models))
w = 0.36
for ax, ref, best, title in zip(
    axes,
    [ref_c1, ref_c16],
    [best_c1, best_c16],
    ["Concurrency 1", "Concurrency 16"],
):
    ax.bar(x - w / 2, ref, w, label="Reference", color="#9fb9d0")
    ax.bar(x + w / 2, best, w, label="Best measured", color="#e86a47")
    ax.set_xticks(x, models)
    ax.set_ylabel("Decode average TPS (tokens/s/request)")
    ax.set_title(title)
    for i, (r, b) in enumerate(zip(ref, best)):
        delta = (b / r - 1) * 100
        ax.text(i + w / 2, b + max(best) * 0.025, f"{delta:+.1f}%", ha="center", va="bottom", fontsize=8)
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("GR308 Qwen: best measured decode rate vs reference table", fontsize=13)
fig.savefig(OUT / "figure-01-reference-vs-best-decode-tps.png", bbox_inches="tight")
plt.close(fig)

roof = np.array([1066.0, 896.0, 800.0, 1420.0])
measured = np.array([180.635100, 247.162528, 225.671965, 255.930840])
eff = measured / roof * 100
fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180, constrained_layout=True)
ax.bar(x - w / 2, roof, w, label="Ideal batch roof", color="#b9c2c9")
ax.bar(x + w / 2, measured, w, label="Best C16 aggregate", color="#277da1")
ax.set_xticks(x, models)
ax.set_ylabel("Aggregate output throughput (tokens/s)")
ax.set_title("Best C16 aggregate throughput vs ideal batch roof")
for i, (v, e) in enumerate(zip(measured, eff)):
    ax.text(i + w / 2, v + 24, f"{v:.1f}\n({e:.1f}%)", ha="center", va="bottom", fontsize=8)
ax.legend(frameon=False)
fig.savefig(OUT / "figure-02-theoretical-roof-efficiency.png", bbox_inches="tight")
plt.close(fig)
