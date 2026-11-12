"""论文附录 C 图：混合对流（E012）与多孔方腔（E020）外推工况的速度场与冷壁传热，均为 N4 下 SA-PODG 误差最接近外推中位数的工况。
每行一个算例：高保真速度大小（附流线）；SA-PODG 与 NE-PODG 的速度大小误差 (|U| − |U*|)/max|U*|（同一行色标相同）；
冷壁局部 Nu 的相对误差沿 y 的分布。数据 cases/<算例>/results/n4/fields_<编号>.npz（各算例 scripts/fields.py）。
用法（仓库根目录）：python paper/figures/script/figC2_cavities.py [输出路径，默认 paper/figures/Fig_C2.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cmap = plt.get_cmap("RdBu_r")
div = matplotlib.colors.ListedColormap(cmap(np.linspace(0.08, 0.92, 256)))
red = matplotlib.colors.ListedColormap(cmap(np.linspace(0.5, 0.9, 256)))
SA, NE = cmap(0.12), cmap(0.88)
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_C2.pdf"
rows = [("mixed", "E012", lambda d: rf"Mixed convection, $Re={d['Re']:g}$, $Ri={d['Ri']:g}$"),
        ("porous", "E020", lambda d: rf"Porous cavity, $Ra^\ast={d['Ras']:g}$, $Da={d['Da']:.3g}$")]

mm = 1 / 25.4
fig, axs = plt.subplots(2, 4, figsize=(190 * mm, 92 * mm), gridspec_kw=dict(left=0.04, right=0.985, bottom=0.08, top=0.9,
                                                                             wspace=0.38, hspace=0.42,
                                                                             width_ratios=[1, 1, 1, 1.15]))
for r, (c, cid, title) in enumerate(rows):
    d = np.load(root / "cases" / c / "results" / "n4" / f"fields_{cid}.npz")
    x, xc = d["x"], d["xc"]
    Uref = d["U_fom"]
    ax = axs[r, 0]
    im = ax.pcolormesh(x, x, Uref.T, cmap=red, shading="flat", rasterized=True)
    ax.contour(x, x, d["psi_fom"].T, levels=np.linspace(np.min(d["psi_fom"]), np.max(d["psi_fom"]), 14)[1:-1], colors="k",
               linewidths=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2)
    ax.set_title(title(d), fontsize=7.5, loc="left", pad=3)
    lim = np.abs(d["U_sapodg"] - Uref).max() / Uref.max()
    for j, (k, lab) in enumerate((("sapodg", "SA-PODG"), ("VS_v", "NE-PODG"))):
        ax = axs[r, 1 + j]
        im = ax.pcolormesh(x, x, ((d[f"U_{k}"] - Uref) / Uref.max()).T, cmap=div, vmin=-lim, vmax=lim, shading="flat",
                           rasterized=True)
        ax.set_title(lab, fontsize=7.5, pad=3)
        if j == 1:
            cb = fig.colorbar(im, ax=axs[r, 1:3].tolist(), fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6, length=2)
            cb.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
            cb.ax.yaxis.get_offset_text().set_fontsize(6)
    for a in axs[r, :3]:
        a.set_aspect("equal")
        a.set_xticks([0, 0.5, 1])
        a.set_yticks([0, 0.5, 1])
        a.tick_params(length=2, pad=1.5, labelsize=6.5)
    ax = axs[r, 3]
    for k, col in (("sapodg", SA), ("VS_v", NE)):
        ax.plot(np.abs(d[f"Nu_{k}"] / d["Nu_fom"] - 1), xc, color=col, lw=0.9)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"relative error of local $Nu$ on cold wall", labelpad=1, fontsize=7)
    ax.set_ylabel(r"$y$", labelpad=1)
    ax.tick_params(which="both", length=2, pad=1.5)
fig.legend(handles=[plt.Line2D([], [], color=SA, lw=0.9, label="SA-PODG"), plt.Line2D([], [], color=NE, lw=0.9, label="NE-PODG")],
           loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2, frameon=False, fontsize=7)
fig.savefig(out, dpi=400)
print(f"→ {out}")
