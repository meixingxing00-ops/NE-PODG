"""论文 Fig. 8：增广机制。(a) nc，N12：SA-PODG 与未训练（r1–r3）、训练（v、v1、v2）网络构造的 NE-PODG 的主场误差 e，
三个种子合并；上方为 1 − Γ 与 χ_B/χ_W 的中位数；(b) 六个算例默认预算下 1 − Γ = (e_proj,B / e_proj,W)² 的分布；(c) χ_B/χ_W 的分布。
细线为最小到最大值，粗段为四分位区间，点为中位数（实心内插、空心外推）。(b)、(c) 只用两者都解出且 SA-PODG 的 e_proj > 1e-9 的工况；
(a) 中 × 为未收敛工况。数据 paper/figures/data/paired.csv 与 cases/nc/results/n12/。
用法（仓库根目录）：python paper/figures/script/fig8_mechanism.py [输出路径，默认 paper/figures/Fig_08.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cmap = plt.get_cmap("RdBu_r")
SA, NE, RN = cmap(0.12), cmap(0.88), cmap(0.65)
CASES = ["nc", "graetz", "conjugate", "mixed", "porous", "bfs"]
NAMES = ["NC", "CG", "CC", "MC", "PC", "BFS"]
EPS = 1e-9
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_08.pdf"
P = pd.read_csv(root / "paper" / "figures" / "data" / "paired.csv")
R = root / "cases" / "nc" / "results" / "n12"


def summ(ax, x, v, color, filled, bad=()):
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([x, x], [v.min(), v.max()], color=color, lw=0.7, solid_capstyle="butt")
    ax.plot([x, x], [q1, q3], color=color, lw=2.6, solid_capstyle="butt")
    ax.plot(x, q2, "o", ms=3.4, mec=color, mfc=color if filled else "white", mew=0.8, zorder=3)
    ax.plot([x] * len(bad), bad, "x", ms=3.5, color=color, mew=0.8, zorder=3)


def style(ax):
    ax.set_yscale("log")
    ax.tick_params(which="both", right=True, length=2.5, pad=2)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(which="minor", length=1.5)


mm = 1 / 25.4
fig, axs = plt.subplots(1, 3, figsize=(190 * mm, 66 * mm), gridspec_kw=dict(left=0.06, right=0.995, bottom=0.12, top=0.88,
                                                                             wspace=0.28, width_ratios=[1, 1.25, 1.25]))
ax = axs[0]
for j, role in enumerate(("interp", "extrap")):
    sa = pd.read_csv(R / f"sapodg_{role}.csv").set_index("id")
    filled = role == "interp"
    summ(ax, j * 3 + 0, sa.e.to_numpy(), SA, filled)
    for i, (keys, col) in enumerate(((("r1", "r2", "r3"), RN), (("v", "v1", "v2"), NE))):
        t = [pd.read_csv(R / f"VS_{k}_{role}.csv").set_index("id").loc[sa.index] for k in keys]
        e = np.concatenate([x.e.to_numpy() for x in t])
        ok = np.concatenate([(x.converged.astype(str) == "True").to_numpy() for x in t])
        m = np.concatenate([(sa.e_proj > EPS).to_numpy()] * 3) & ok
        ep, epw = np.concatenate([x.e_proj.to_numpy() for x in t]), np.concatenate([sa.e_proj.to_numpy()] * 3)
        ew = np.concatenate([sa.e.to_numpy()] * 3)
        g1 = np.median(((ep / epw) ** 2)[m])
        chi = np.median(((e / ep) / (ew / epw))[m])
        x = j * 3 + 1 + i
        summ(ax, x, e[ok], col, filled, e[~ok])
        ax.text(x, 3e1, f"{g1:.2g}\n{chi:.2f}", ha="center", va="top", fontsize=6.5)
ax.text(-0.5, 3e1, r"$1-\Gamma$" + "\n" + r"$\chi_B/\chi_W$", ha="left", va="top", fontsize=6.5)
style(ax)
ax.set_ylim(1e-10, 1e2)
ax.set_xlim(-0.7, 5.6)
ax.set_xticks([1, 4])
ax.set_xticklabels(["Interpolation", "Extrapolation"])
ax.axvline(2.5, color="0.85", lw=0.5, zorder=0)
ax.set_ylabel(r"Main-field error $e$", labelpad=2)
for k, (a, lab) in enumerate(((axs[1], r"$1-\Gamma$"), (axs[2], r"$\chi_B/\chi_W$"))):
    for i, c in enumerate(CASES):
        for role, dr in (("interp", -0.17), ("extrap", 0.17)):
            p = P[(P.case == c) & (P.role == role) & P.solved_sa & P.solved_ne & (P.ep_sa > EPS)]
            v = (p.ep_ne / p.ep_sa) ** 2 if k == 0 else (p.e_ne / p.ep_ne) / (p.e_sa / p.ep_sa)
            summ(a, i + dr, v.to_numpy(), NE, role == "interp")
    style(a)
    a.set_xticks(range(len(CASES)))
    a.set_xticklabels(NAMES)
    a.set_xlim(-0.55, len(CASES) - 0.45)
    for i in range(1, len(CASES)):
        a.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
    a.set_ylabel(lab, labelpad=2)
axs[2].axhline(1, color="0.6", lw=0.6, ls="--", zorder=0)
fig.legend(handles=[plt.Line2D([], [], color=c, lw=2.6, label=l) for c, l in
                    ((SA, "SA-PODG"), (RN, "NE-PODG, untrained network"), (NE, "NE-PODG, trained network"))]
           + [plt.Line2D([], [], ls="", marker="o", ms=3.4, mec="0.3", mfc="0.3", label="Interpolation"),
              plt.Line2D([], [], ls="", marker="o", ms=3.4, mec="0.3", mfc="white", label="Extrapolation"),
              plt.Line2D([], [], ls="", marker="x", ms=3.5, color="0.3", label="Not converged")],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=7, handlelength=1.6,
           columnspacing=1.2)
for a, t in zip(axs, ("(a)", "(b)", "(c)")):
    a.text(0.03, 0.03 if a is ax else 0.97, t, transform=a.transAxes, va="bottom" if a is ax else "top", fontsize=8)
fig.savefig(out, dpi=300)
print(f"→ {out}")
