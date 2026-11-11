"""论文 Fig. 5：N12、Ra = 10⁶、Pr = 0.562（E022）的 (a) x = 0.5 处 u 沿 y、(b) y = 0.5 处 v 沿 x、(c) 热壁局部 Nu 沿 y；
高保真、POD–Galerkin、SA-PODG、NE-PODG 与网络直接预测。数据 cases/nc/results/n12/profiles_E022.npz（cases/nc/scripts/profiles.py）。
用法（仓库根目录）：python paper/figures/script/fig5_profiles.py [输出路径，默认 paper/figures/Fig_05.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
d = np.load(root / "cases" / "nc" / "results" / "n12" / "profiles_E022.npz")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_05.pdf"

styles = [("teacher", "network prediction", dict(color=cm(0.32), lw=1.0, ls=":")),
          ("pod", "POD–Galerkin", dict(color="0.55", lw=1.0, ls="-.")),
          ("sapodg", "SA-PODG", dict(color=cm(0.08), lw=1.0, ls="--")),
          ("fom", "high-fidelity", dict(color="k", lw=1.6, ls="-")),
          ("VS_v", "NE-PODG", dict(color=cm(0.9), lw=1.0, ls=(0, (4, 2))))]
order = ["fom", "VS_v", "sapodg", "pod", "teacher"]                       # 图例顺序
# (量的坐标键, 量, 横轴标签, 纵轴标签, 放大区间, 放大图位置)
panels = [("y", "u", r"$y$", r"$u$ at $x=0.5$", (0.80, 0.93), [0.53, 0.62, 0.43, 0.34]),
          ("x", "v", r"$x$", r"$v$ at $y=0.5$", (0.015, 0.065), [0.33, 0.10, 0.40, 0.33]),
          ("yc", "Nu", r"$y$", r"local $Nu$ on hot wall", (0.0, 0.12), [0.53, 0.55, 0.43, 0.40])]

mm = 1 / 25.4
fig, axs = plt.subplots(1, 3, figsize=(190 * mm, 62 * mm), gridspec_kw=dict(left=0.06, right=0.99, bottom=0.17,
                                                                             top=0.83, wspace=0.28))
for k, (ax, (xk, q, xl, yl, zx, pos)) in enumerate(zip(axs, panels)):
    x = d[xk]
    for m, name, kw in styles:
        ax.plot(x, d[f"{q}_{m}"], label=name, **kw)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xl, labelpad=1)
    ax.set_ylabel(yl, labelpad=1)
    ax.tick_params(length=2.5, pad=2)
    ax.text(-0.2, 1.04, f"({'abc'[k]})", transform=ax.transAxes, fontsize=9)
    # 峰值附近局部放大：高保真、SA-PODG、NE-PODG
    sel = (x >= zx[0]) & (x <= zx[1])
    ys = np.concatenate([d[f"{q}_{m}"][sel] for m in ("fom", "sapodg", "VS_v")])
    pad = 0.08 * (ys.max() - ys.min())
    ins = ax.inset_axes(pos)
    for m, name, kw in styles[2:]:
        ins.plot(x[sel], d[f"{q}_{m}"][sel], **kw)
    ins.set(xlim=zx, ylim=(ys.min() - pad, ys.max() + pad))
    ins.tick_params(labelsize=6, length=2, pad=1)
    for s in ins.spines.values():
        s.set_linewidth(0.6)
    for c in ax.indicate_inset_zoom(ins, edgecolor="0.4", lw=0.5)[1]:
        c.set_visible(False)
hs, ls = axs[0].get_legend_handles_labels()
lab = dict(zip([m for m, _, _ in styles], zip(hs, ls)))
fig.legend(*zip(*[lab[m] for m in order]), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
           fontsize=7, handlelength=2.8, columnspacing=1.6)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
