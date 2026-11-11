"""论文 Fig. 6：N12 网络 (a) 训练损失；(b) 网络直接预测与 NE-PODG 的速度误差 e，按训练快照、内插、外推分组（横线为中位数）。
数据 cases/nc/models/v/n12_loss.csv、cases/nc/results/n12/teacher_v_{train,interp,extrap}.csv、VS_v_{interp,extrap}.csv。
用法（仓库根目录）：python paper/figures/script/fig6_training.py [输出路径，默认 paper/figures/Fig_06.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
nc = root / "cases" / "nc"
res = nc / "results" / "n12"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_06.pdf"
loss = pd.read_csv(nc / "models" / "v" / "n12_loss.csv")
cnet, cne = cm(0.18), cm(0.9)

mm = 1 / 25.4
fig, axs = plt.subplots(1, 2, figsize=(190 * mm, 66 * mm), gridspec_kw=dict(left=0.07, right=0.98, bottom=0.16,
                                                                             top=0.86, wspace=0.22))
# (a) 训练损失；竖虚线为学习率降低处
ax = axs[0]
for s in (1e4, 2e4):
    ax.axvline(s, color="0.7", lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.plot(loss.step, loss.value, color=cnet, lw=1.0)
ax.set(yscale="log", xlim=(0, 3e4))
ax.set_xticks([0, 1e4, 2e4, 3e4], ["0", r"$1\times10^4$", r"$2\times10^4$", r"$3\times10^4$"])
ax.tick_params(which="both", top=True, right=True)
ax.tick_params(length=2.5, pad=2)
ax.tick_params(which="minor", length=1.5)
ax.set_xlabel("training step", labelpad=1)
ax.set_ylabel("training loss", labelpad=1)
ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=9)

# (b) 速度误差分组
ax = axs[1]
rng = np.random.default_rng(0)
groups = [("training\nsnapshots", [("teacher_v_train", cnet)]),
          ("interpolation", [("teacher_v_interp", cnet), ("VS_v_interp", cne)]),
          ("extrapolation", [("teacher_v_extrap", cnet), ("VS_v_extrap", cne)])]
for i, (_, items) in enumerate(groups):
    for j, (f, c) in enumerate(items):
        x0 = i + (0 if len(items) == 1 else (j - 0.5) * 0.36)
        e = pd.read_csv(res / f"{f}.csv").e.values
        ax.scatter(x0 + rng.uniform(-0.1, 0.1, len(e)), e, s=6, color=c, lw=0, alpha=0.75, zorder=3)
        ax.plot([x0 - 0.15, x0 + 0.15], [np.median(e)] * 2, color="0.1", lw=1.2, zorder=4)
ax.set(yscale="log", xlim=(-0.6, 2.6), ylim=(5e-11, 3))
ax.set_xticks(range(3), [g for g, _ in groups])
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", which="both", right=True)
ax.tick_params(axis="y", length=2.5, pad=2)
ax.tick_params(axis="y", which="minor", length=1.5)
ax.set_ylabel(r"$e$", labelpad=1)
ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, va="top", fontsize=9)
hs = [Line2D([], [], ls="none", marker="o", ms=3.5, color=cnet, label="network prediction"),
      Line2D([], [], ls="none", marker="o", ms=3.5, color=cne, label="NE-PODG"),
      Line2D([], [], color="0.1", lw=1.2, label="median")]
ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7,
          handletextpad=0.3, columnspacing=1.4)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=400)
