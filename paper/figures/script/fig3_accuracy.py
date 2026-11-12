"""论文附录 C 图（原 Fig. 3）：N12 下全部测试工况的 (a) 三种方法速度误差随 Ra 的分布；(b) SA-PODG 与 NE-PODG 的误差之比随 Ra 的分布，按 Pr 着色。
灰色竖线为 N12 训练快照所在的 Ra，阴影为外推区。数据 cases/nc/results/n12/{pod,sapodg,VS_v}_{interp,extrap}.csv。
用法（仓库根目录）：python paper/figures/script/fig3_accuracy.py [输出路径，默认 paper/figures/Fig_C1.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results" / "n12"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_C1.pdf"
load = lambda m: pd.concat([pd.read_csv(res / f"{m}_{r}.csv") for r in ("interp", "extrap")])
d = {m: load(m) for m in ("pod", "sapodg", "VS_v")}
assert all((d[m].id.values == d["VS_v"].id.values).all() for m in d)
Ra, Pr = d["VS_v"].Ra.values, d["VS_v"].Pr.values

mm = 1 / 25.4
fig, axs = plt.subplots(1, 2, figsize=(190 * mm, 70 * mm), gridspec_kw=dict(left=0.065, right=0.925, bottom=0.15,
                                                                             top=0.84, wspace=0.2))


def frame(ax, ylab):
    for r in (10, 316, 1e4, 3.16e5):
        ax.axvline(r, color="0.75", lw=0.6, zorder=0)
    ax.axvspan(3.16e5, 1.5e6, color="0.92", lw=0, zorder=0)
    ax.set(xscale="log", yscale="log", xlim=(7, 1.5e6))
    ax.set_xticks([10 ** j for j in range(1, 7)])
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(length=2.5, pad=2)
    ax.tick_params(which="minor", length=1.5)
    ax.set_xlabel(r"$Ra$", labelpad=1)
    ax.set_ylabel(ylab, labelpad=1)


# (a) 误差
ax = axs[0]
frame(ax, r"$e$")
for m, name, c, mk, s in [("pod", "POD–Galerkin", "0.55", "^", 9), ("sapodg", "SA-PODG", cm(0.1), "s", 8),
                          ("VS_v", "NE-PODG", cm(0.9), "o", 9)]:
    ax.scatter(Ra, d[m].e, s=s, marker=mk, c=[c], lw=0, alpha=0.85, label=name, zorder=3)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.15), ncol=3, frameon=False, fontsize=7, handletextpad=0.2,
          columnspacing=1.2, markerscale=1.3)
ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=9)
ax.text(5.5e5, 1.01, "extrap.", ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())

# (b) 误差之比，按 Pr 着色
ax = axs[1]
frame(ax, r"$e_{\mathrm{SA}}/e_{\mathrm{NE}}$")
ax.axhline(1, color="0.3", lw=0.7, ls=(0, (4, 2)), zorder=1)
sc = ax.scatter(Ra, d["sapodg"].e.values / d["VS_v"].e.values, s=11, c=Pr, cmap=cm, norm=LogNorm(0.1, 10), lw=0.3,
                edgecolors="0.3", zorder=3)
ax.set_ylim(0.85, 25)
ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, va="top", fontsize=9)
ax.text(5.5e5, 1.01, "extrap.", ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
pos = ax.get_position()
cb = fig.colorbar(sc, cax=fig.add_axes([pos.x1 + 0.012, pos.y0, 0.011, pos.height]))
cb.set_ticks([0.1, 0.316, 1, 3.16, 10], labels=["0.1", "0.316", "1", "3.16", "10"])
cb.minorticks_off()
cb.set_label(r"$Pr$", labelpad=2)
cb.ax.tick_params(labelsize=7, length=2.5, width=0.6)
cb.outline.set_linewidth(0.6)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=400)
