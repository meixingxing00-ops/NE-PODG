"""论文 Fig. 9：方腔自然对流的增广维数与快照预算。(a) N12、(b) N50：逐工况增益中位数随增广维数 n_Z 的变化，实线为求解误差比
e_SA/e，点线为最佳逼近误差比 e_proj,W/e_proj；圆点连线为把 m = 128 的网络截断到前 n_Z 个奇异方向（实线种子 1234，虚线种子 1），
方块（种子 1234）与菱形（种子 1）为以 m = n_Z 重新训练的网络；深红实心为内插，浅红空心为外推。(c) 内插、(d) 外推：三种方法速度误差的中位数（实线）
与最大值（虚线）随快照预算的变化。数据 cases/nc/results/n<B>/。
用法（仓库根目录）：python paper/figures/script/fig9_dimension_budget.py [输出路径，默认 paper/figures/Fig_09.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_09.pdf"
load = lambda b, m, r: pd.read_csv(res / f"n{b}" / f"{m}_{r}.csv").set_index("id")
ROLE = {"interp": (cm(0.95), True), "extrap": (cm(0.7), False)}
DIMS = [8, 16, 32, 64, 128]

mm = 1 / 25.4
fig, axs = plt.subplots(2, 2, figsize=(190 * mm, 122 * mm), gridspec_kw=dict(left=0.07, right=0.98, bottom=0.08, top=0.9,
                                                                              wspace=0.18, hspace=0.42))


def gains(b, r, f):
    sa, d = load(b, "sapodg", r), load(b, f, r)
    d = d.loc[sa.index]
    return (sa.e / d.e).median(), (sa.e_proj / d.e_proj).median()


for k, b in enumerate((12, 50)):
    ax = axs[0, k]
    for r, (c, filled) in ROLE.items():
        for seed, ls in (("v", "-"), ("v1", "--")):
            g = [gains(b, r, f"VS_{seed}_z{n}" if n < 128 else f"VS_{seed}") for n in DIMS]
            ax.plot(DIMS, [x[0] for x in g], color=c, ls=ls, marker="o", ms=3, lw=1.0, mfc=c if filled else "white")
            if seed == "v":
                ax.plot(DIMS, [x[1] for x in g], color=c, ls=":", lw=0.9)
            m = [gains(b, r, f"VS_{seed}m{n}")[0] for n in DIMS[1:4]]
            ax.plot(DIMS[1:4], m, ls="", marker="s" if seed == "v" else "D", ms=4, mec=c, mfc=c if filled else "white", mew=0.8)
    ax.axhline(1, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set(xscale="log", yscale="log", xlim=(6.5, 160))
    ax.set_xticks(DIMS, [str(n) for n in DIMS])
    ax.minorticks_off()
    yt = [1, 2, 5, 10] if b == 12 else [1, 1.5, 2, 3, 4]
    ax.set_yticks(yt, [f"{t:g}" for t in yt])
    ax.tick_params(length=2.5, pad=2)
    ax.set_xlabel(r"enrichment dimension $n_Z$", labelpad=1)
    ax.set_ylabel("median gain", labelpad=1)
    ax.set_title(rf"$N_{{{b}}}$", fontsize=8, pad=3)
    ax.text(-0.11, 1.04, "(a)" if k == 0 else "(b)", transform=ax.transAxes, fontsize=9)

budgets = [12, 20, 30, 50]
styles = [("pod", "POD–Galerkin", "0.45", "^"), ("sapodg", "SA-PODG", cm(0.1), "s"), ("VS_v", "NE-PODG", cm(0.9), "o")]
for k, (r, title) in enumerate([("interp", "interpolation"), ("extrap", "extrapolation")]):
    ax = axs[1, k]
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
    ax.text(-0.11, 1.04, "(c)" if k == 0 else "(d)", transform=ax.transAxes, fontsize=9)

h1 = [Line2D([], [], color=ROLE["interp"][0], lw=1.0, marker="o", ms=3, label="interpolation"),
      Line2D([], [], color=ROLE["extrap"][0], lw=1.0, marker="o", ms=3, mfc="white", label="extrapolation"),
      Line2D([], [], color="0.2", lw=1.0, label=r"truncated, $e_{\mathrm{SA}}/e$"),
      Line2D([], [], color="0.2", lw=0.9, ls=":", label=r"truncated, $e_{\mathrm{proj},W}/e_{\mathrm{proj}}$"),
      Line2D([], [], color="0.2", lw=1.0, ls="--", label="truncated, seed 1"),
      Line2D([], [], ls="", marker="s", ms=4, mec="0.2", mfc="0.2", label=r"retrained, $m=n_Z$"),
      Line2D([], [], ls="", marker="D", ms=4, mec="0.2", mfc="0.2", label="retrained, seed 1")]
fig.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False, fontsize=6.5,
           handlelength=2.2, columnspacing=1.0)
h2 = [Line2D([], [], color=c, marker=mk, ms=3.5, lw=1.0, label=name) for _, name, c, mk in styles]
h2 += [Line2D([], [], color="0.2", lw=1.0, label="median"),
       Line2D([], [], color="0.2", lw=1.0, ls="--", marker="o", ms=3.5, mfc="white", label="maximum")]
fig.legend(handles=h2, loc="center", bbox_to_anchor=(0.5, 0.475), ncol=5, frameon=False, fontsize=6.5,
           handlelength=2.6, columnspacing=1.6)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
print(f"→ {out}")
