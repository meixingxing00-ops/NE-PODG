"""论文 Fig. 9：三种方法速度误差的中位数与最大值随快照预算的变化，(a) 内插、(b) 外推。
数据 cases/nc/results/n<B>/{pod,sapodg,VS_v}_{interp,extrap}.csv。
用法（仓库根目录）：python paper/figures/script/fig9_budget.py [输出路径，默认 paper/figures/Fig_09.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_09.pdf"
load = lambda b, m, r: pd.read_csv(res / f"n{b}" / f"{m}_{r}.csv")

mm = 1 / 25.4
fig, axs = plt.subplots(1, 2, figsize=(190 * mm, 70 * mm), gridspec_kw=dict(left=0.07, right=0.98, bottom=0.15,
                                                                             top=0.80, wspace=0.18))
budgets = [12, 20, 30, 50]
styles = [("pod", "POD–Galerkin", "0.45", "^"), ("sapodg", "SA-PODG", cm(0.1), "s"), ("VS_v", "NE-PODG", cm(0.9), "o")]
for k, (r, title) in enumerate([("interp", "interpolation"), ("extrap", "extrapolation")]):
    ax = axs[k]
    for m, name, c, mk in styles:
        e = [load(b, m, r).e for b in budgets]
        ax.plot(budgets, [x.median() for x in e], color=c, marker=mk, ms=3.5, lw=1.0)
        ax.plot(budgets, [x.max() for x in e], color=c, marker=mk, ms=3.5, lw=1.0, ls="--", mfc="white")
    ax.set(yscale="log", xlim=(9, 53))
    ax.set_xticks(budgets, [rf"$N_{{{b}}}$" for b in budgets])
    ax.tick_params(length=2.5, pad=2)
    ax.set_xlabel("snapshot budget", labelpad=1)
    ax.set_ylabel(r"$e$", labelpad=1)
    ax.set_title(title, fontsize=8, pad=3)
    ax.text(-0.11, 1.05, "(a)" if k == 0 else "(b)", transform=ax.transAxes, fontsize=9)
hs = [Line2D([], [], color=c, marker=mk, ms=3.5, lw=1.0, label=name) for _, name, c, mk in styles]
hs += [Line2D([], [], color="0.2", lw=1.0, label="median"),
       Line2D([], [], color="0.2", lw=1.0, ls="--", marker="o", ms=3.5, mfc="white", label="maximum")]
fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=7,
           handlelength=2.6, columnspacing=1.6)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
