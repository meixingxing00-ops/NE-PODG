"""论文 Fig. 3：六个算例默认预算下的跨算例精度。(a) 内插、外推测试中 POD-G、SA-PODG、NE-PODG 主场误差 e 的分布：细线为最小到最大值，
粗段为四分位区间，点为中位数（实心内插、空心外推），× 为闭合未满足残差收敛判据的工况；(b) 两者都解出的工况上 e_SA/e_NE 的逐工况分布，
横线为中位数。数据 paper/figures/data/paired.csv（stats.py）。
用法（仓库根目录）：python paper/figures/script/fig3_cross.py [输出路径，默认 paper/figures/Fig_03.pdf；.png 用于预览]"""
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
COL = {"pod": "0.45", "sa": cmap(0.12), "ne": cmap(0.88)}
LAB = {"pod": "POD–Galerkin", "sa": "SA-PODG", "ne": "NE-PODG"}
CASES = ["nc", "graetz", "conjugate", "mixed", "porous", "bfs"]
NAMES = ["Natural\nconvection", "Couette–\nGraetz", "Conjugate\nchannel", "Mixed\nconvection", "Porous\ncavity",
         "Backward-\nfacing step"]
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_03.pdf"
P = pd.read_csv(root / "paper" / "figures" / "data" / "paired.csv")

mm = 1 / 25.4
fig, (ax, bx) = plt.subplots(2, 1, figsize=(190 * mm, 118 * mm), sharex=True,
                             gridspec_kw=dict(left=0.075, right=0.99, bottom=0.11, top=0.93, hspace=0.08))
ROLE = {"interp": (-0.19, True), "extrap": (0.19, False)}
for i, c in enumerate(CASES):
    for role, (dr, filled) in ROLE.items():
        p = P[(P.case == c) & (P.role == role)]
        for k, dm in (("pod", -0.085), ("sa", 0.0), ("ne", 0.085)):
            x = i + dr + dm
            s = p[p[f"solved_{k}"]][f"e_{k}"].to_numpy()
            q1, q2, q3 = np.percentile(s, [25, 50, 75])
            ax.plot([x, x], [s.min(), s.max()], color=COL[k], lw=0.7, solid_capstyle="butt")
            ax.plot([x, x], [q1, q3], color=COL[k], lw=2.6, solid_capstyle="butt")
            ax.plot(x, q2, "o", ms=3.4, mec=COL[k], mfc=COL[k] if filled else "white", mew=0.8, zorder=3)
            bad = p[~p[f"solved_{k}"]][f"e_{k}"]
            ax.plot([x] * len(bad), bad, "x", ms=3.5, color=COL[k], mew=0.8, zorder=3)
        q = p[p.solved_sa & p.solved_ne]
        r = (q.e_sa / q.e_ne).to_numpy()
        rng = np.random.default_rng(i * 2 + filled)
        xs = i + dr + rng.uniform(-0.1, 0.1, len(r))
        bx.plot(xs, r, "o", ms=2.0, mec=COL["ne"], mfc=COL["ne"] if filled else "white", mew=0.5, alpha=0.8)
        bx.plot([i + dr - 0.13, i + dr + 0.13], [np.median(r)] * 2, color="k", lw=1.1)
ax.set_yscale("log")
ax.set_ylabel(r"Main-field error $e$", labelpad=2)
bx.set_yscale("log")
bx.set_ylabel(r"$e_{\mathrm{SA}}/e_{\mathrm{NE}}$", labelpad=2)
bx.axhline(1, color="0.6", lw=0.6, ls="--", zorder=0)
bx.set_xticks(range(len(CASES)))
bx.set_xticklabels(NAMES)
bx.set_xlim(-0.55, len(CASES) - 0.45)
for a in (ax, bx):
    for i in range(1, len(CASES)):
        a.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
    a.tick_params(which="both", right=True, length=2.5, pad=2)
    a.tick_params(axis="x", which="both", length=0)
    a.tick_params(which="minor", length=1.5)
h = [plt.Line2D([], [], color=COL[k], lw=2.6, label=LAB[k]) for k in ("pod", "sa", "ne")]
h += [plt.Line2D([], [], ls="", marker="o", ms=3.4, mec="0.3", mfc="0.3", label="Interpolation"),
      plt.Line2D([], [], ls="", marker="o", ms=3.4, mec="0.3", mfc="white", label="Extrapolation"),
      plt.Line2D([], [], ls="", marker="x", ms=3.5, color="0.3", label="Not converged")]
ax.legend(handles=h, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7, handlelength=1.6,
          columnspacing=1.2)
for a, t in ((ax, "(a)"), (bx, "(b)")):
    a.text(0.006, 0.97, t, transform=a.transAxes, va="top", fontsize=8)
fig.savefig(out, dpi=300)
print(f"→ {out}")
