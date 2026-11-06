"""参数全域 Ra ∈ [10, 10⁶]、Pr ∈ [0.1, 10]，训练区 Ra ∈ [10, 3.16×10⁵]、Pr ∈ [0.1, 10]，及方腔自然对流工况表
cases/nc/cases.csv 中的训练池（按快照预算 N12、N20、N30、N50 区分）、内插测试与外推测试工况的对数坐标示意。
用法（仓库根目录）：python others/plot_domain.py"""
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams.update({"font.family": ["Times New Roman", "Microsoft YaHei", "Segoe UI Symbol"], "mathtext.fontset": "stix",
                            "font.size": 10.5, "axes.unicode_minus": False})
cmap = plt.get_cmap("RdBu_r")
here = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(here.parent / "cases" / "nc" / "cases.csv", encoding="utf-8")))
pick = lambda f: [(float(r["Ra"]), float(r["Pr"])) for r in rows if f(r)]
on = lambda r, k: r[k] == "1"
groups = [("N12（各档共有）", pick(lambda r: on(r, "n12")), dict(marker="s", c=[cmap(0.98)], ec="white", lw=0.4, s=26)),
          ("N20 新增（Pr 加密）", pick(lambda r: on(r, "n20") and not on(r, "n12")), dict(marker="D", c=[cmap(0.80)], ec="white", lw=0.4, s=24)),
          ("N30 新增（Ra 加密）", pick(lambda r: on(r, "n30") and not on(r, "n12")), dict(marker="o", c=[cmap(0.66)], ec="white", lw=0.4, s=26)),
          ("仅 N50", pick(lambda r: on(r, "n50") and not on(r, "n20") and not on(r, "n30")), dict(marker="^", c=[cmap(0.20)], ec="white", lw=0.4, s=28)),
          ("内插测试", pick(lambda r: r["role"] == "interp"), dict(marker="o", fc="none", ec=cmap(0.05), lw=0.7, s=12)),
          ("外推测试", pick(lambda r: r["role"] == "extrap"), dict(marker="x", c=[cmap(0.05)], lw=0.8, s=14))]

fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.add_patch(Rectangle((10, 0.1), 1e6 - 10, 10 - 0.1, fc=cmap(0.42), ec="none", alpha=0.35, zorder=0))
ax.add_patch(Rectangle((10, 0.1), 1e6 - 10, 10 - 0.1, fc="none", ec=cmap(0.05), lw=1.4, zorder=1))
ax.add_patch(Rectangle((10, 0.1), 3.16e5 - 10, 10 - 0.1, fc=cmap(0.75), ec="none", alpha=0.18, zorder=1))
ax.add_patch(Rectangle((10, 0.1), 3.16e5 - 10, 10 - 0.1, fc="none", ec=cmap(0.95), lw=1.4, ls="--", zorder=2))
for name, pts, kw in groups:
    ax.scatter(*zip(*pts), zorder=3, label=f"{name}：{len(pts)}", **kw)
ax.text(10 ** 3.25, 11.6, "训练区", ha="center", va="center", color=cmap(0.95), fontsize=9.5)
ax.text(10 ** 5.75, 11.6, "外推区", ha="center", va="center", color=cmap(0.05), fontsize=9.5)
ax.set(xscale="log", yscale="log", xlim=(10 / 1.6, 1e6 * 1.6), ylim=(0.1 / 1.25, 10 * 1.4), xlabel="$Ra$", ylabel="$Pr$")
ax.set_xticks([10 ** k for k in range(1, 7)])
ax.set_yticks([0.1, 0.316, 1, 3.16, 10], ["0.1", "0.316", "1", "3.16", "10"])
ax.minorticks_off()
ax.grid(True, which="major", lw=0.4, color="0.85")
ax.set_axisbelow(True)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.5,
          title="训练区 Ra ∈ [10, 3.16×10⁵]\nN12 ⊂ N20 ⊂ N50，N12 ⊂ N30 ⊂ N50", title_fontsize=9.5)
ax.set_title("参数全域 Ra ∈ [10, 10⁶]，Pr ∈ [0.1, 10]", fontsize=10.5)
fig.tight_layout()
fig.savefig(here / "domain.png", dpi=300)
