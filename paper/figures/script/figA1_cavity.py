"""论文附录 A 图（方腔算例）：(a) 侧壁差分加热方腔的物理模型与边界条件；(b) 参数域、训练池（按快照预算）与测试工况（cases/nc/cases.csv）。
用法（仓库根目录）：python paper/figures/script/figA1_cavity.py [输出路径，默认 paper/figures/Fig_A1.pdf；.png 用于预览]"""
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_A1.pdf"
rows = list(csv.DictReader(open(root / "cases" / "nc" / "cases.csv", encoding="utf-8")))

mm = 1 / 25.4
fig = plt.figure(figsize=(140 * mm, 62 * mm))
axa = fig.add_axes([0.035, 0.13, 0.30, 0.30 * 140 / 62])
axb = fig.add_axes([0.475, 0.17, 0.505, 0.60])

# (a) 方腔
cold, hot = cm(0.08), cm(0.92)
axa.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec="none"))
axa.plot([0, 0], [0, 1], color=cold, lw=2.4, solid_capstyle="butt")
axa.plot([1, 1], [0, 1], color=hot, lw=2.4, solid_capstyle="butt")
for y0 in (0, 1):
    axa.plot([0, 1], [y0, y0], color="0.2", lw=0.9)
    d = -1 if y0 == 0 else 1
    for x0 in [i / 14 for i in range(15)]:
        axa.plot([x0, x0 + 0.045], [y0, y0 + d * 0.045], color="0.45", lw=0.5)
axa.text(-0.07, 0.5, r"$T=-1/2$", rotation=90, ha="right", va="center", color=cold, fontsize=8)
axa.text(1.07, 0.5, r"$T=+1/2$", rotation=90, ha="left", va="center", color=hot, fontsize=8)
axa.text(0.5, 1.09, r"$\partial T/\partial y=0$", ha="center", va="bottom", fontsize=8)
axa.text(0.5, -0.09, r"$\partial T/\partial y=0$", ha="center", va="top", fontsize=8)
axa.text(0.5, 0.55, r"$Ra,\ Pr$", ha="center", va="center", fontsize=9)
axa.text(0.5, 0.41, r"$u=v=0$ on walls", ha="center", va="center", fontsize=7.5)
axa.add_patch(FancyArrowPatch((0.84, 0.91), (0.84, 0.72), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.15"))
axa.text(0.875, 0.815, r"$\mathbf{g}$", ha="left", va="center", fontsize=8)
o, L = 0.08, 0.17
for (dx, dy, lab, ha, va) in [(L, 0, r"$x$", "left", "center"), (0, L, r"$y$", "center", "bottom")]:
    axa.add_patch(FancyArrowPatch((o, o), (o + dx, o + dy), arrowstyle="-|>", mutation_scale=6, lw=0.7, color="0.15"))
    axa.text(o + dx + (0.02 if dx else 0), o + dy + (0.01 if dy else 0), lab, ha=ha, va=va, fontsize=8)
axa.text(-0.03, -0.03, "0", ha="right", va="top", fontsize=7)
axa.text(1.03, -0.03, "1", ha="left", va="top", fontsize=7)
axa.text(-0.03, 1.03, "1", ha="right", va="bottom", fontsize=7)
axa.set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2), aspect="equal")
axa.axis("off")
fig.text(0.01, 0.95, "(a)", fontsize=9, va="top")

# (b) 参数域
pick = lambda f: [(float(r["Ra"]), float(r["Pr"])) for r in rows if f(r)]
on = lambda r, k: r[k] == "1"
groups = [(r"$N_{12}$", pick(lambda r: on(r, "n12")), dict(marker="s", c=[cm(0.98)], ec="none", s=13)),
          (r"added in $N_{20}$", pick(lambda r: on(r, "n20") and not on(r, "n12")), dict(marker="D", c=[cm(0.82)], ec="none", s=11)),
          (r"added in $N_{30}$", pick(lambda r: on(r, "n30") and not on(r, "n12")), dict(marker="o", c=[cm(0.76)], ec="none", s=13)),
          (r"added in $N_{50}$", pick(lambda r: on(r, "n50") and not on(r, "n20") and not on(r, "n30")), dict(marker="^", c=["0.45"], ec="none", s=14)),
          ("interpolation test", pick(lambda r: r["role"] == "interp"), dict(marker="o", fc="none", ec=cm(0.12), lw=0.5, s=6)),
          ("extrapolation test", pick(lambda r: r["role"] == "extrap"), dict(marker="x", c=[cm(0.05)], lw=0.6, s=9))]
axb.add_patch(Rectangle((10, 0.1), 3.16e5 - 10, 9.9, fc=cm(0.62), ec="none", alpha=0.25, zorder=0))
axb.add_patch(Rectangle((3.16e5, 0.1), 1e6 - 3.16e5, 9.9, fc=cm(0.38), ec="none", alpha=0.35, zorder=0))
axb.add_patch(Rectangle((10, 0.1), 1e6 - 10, 9.9, fc="none", ec="0.25", lw=0.7, zorder=1))
axb.plot([3.16e5, 3.16e5], [0.1, 10], color="0.25", lw=0.7, ls="--", zorder=1)
for name, pts, kw in groups:
    axb.scatter(*zip(*pts), zorder=3, label=name, **kw)
axb.text(10 ** 3.25, 13.2, "training region", ha="center", va="bottom", fontsize=7.5)
axb.text(10 ** 5.75, 13.2, "extrap.", ha="center", va="bottom", fontsize=7.5)
axb.set(xscale="log", yscale="log", xlim=(10 / 1.5, 1e6 * 1.5), ylim=(0.1 / 1.3, 10 * 1.3))
axb.set_xlabel(r"$Ra$", labelpad=1)
axb.set_ylabel(r"$Pr$", labelpad=1)
axb.set_xticks([10 ** k for k in range(1, 7)])
axb.set_yticks([0.1, 0.316, 1, 3.16, 10], ["0.1", "0.316", "1", "3.16", "10"])
axb.minorticks_off()
axb.tick_params(length=2.5, pad=2)
axb.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=7,
           handletextpad=0.2, columnspacing=1.0, borderaxespad=0, labelspacing=0.35)
fig.text(0.40, 0.95, "(b)", fontsize=9, va="top")

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
