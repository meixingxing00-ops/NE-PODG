"""论文 Fig. 6：N12 网络 (a) 训练损失；(b) 网络直接输出与 NE-PODG 的速度误差 e 在训练快照、内插、外推三组上的中位数，
竖线为最小值到最大值。数据 paper/figures/data/n12_loss.csv（N12 网络 v 的训练损失，每 300 步平均）、
cases/nc/results/n12/teacher_v_{train,interp,extrap}.csv、VS_v_{interp,extrap}.csv。
用法（仓库根目录）：python paper/figures/script/fig6_training.py [输出路径，默认 paper/figures/Fig_06.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cm = matplotlib.colors.ListedColormap(plt.get_cmap("RdBu_r")([0.08 + 0.84 * i / 255 for i in range(256)]))  # 两端截去
root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results" / "n12"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_06.pdf"
loss = pd.read_csv(root / "paper" / "figures" / "data" / "n12_loss.csv")
cnet, cne = cm(0.15), cm(0.9)

mm = 1 / 25.4
fig, axs = plt.subplots(1, 2, figsize=(190 * mm, 64 * mm), gridspec_kw=dict(left=0.07, right=0.98, bottom=0.17,
                                                                             top=0.87, wspace=0.22))
# (a) 训练损失；竖虚线为学习率降为 1/10 处
ax = axs[0]
for s in (1e4, 2e4):
    ax.axvline(s, color="0.7", lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.plot(loss["step"], loss["loss"], color=cnet, lw=1.0)
ax.plot(loss["step"].iloc[-1], loss["loss"].iloc[-1], "o", ms=3.5, color=cnet)
ax.annotate(r"$1.6\times10^{-5}$", (loss["step"].iloc[-1], loss["loss"].iloc[-1]), xytext=(-4, 9),
            textcoords="offset points", ha="right", fontsize=7)
ax.set(yscale="log", xlim=(0, 3.1e4))
ax.set_xticks([0, 1e4, 2e4, 3e4], ["0", r"$1\times10^4$", r"$2\times10^4$", r"$3\times10^4$"])
ax.tick_params(which="both", top=True, right=True)
ax.tick_params(length=2.5, pad=2)
ax.tick_params(which="minor", length=1.5)
ax.set_xlabel("training step", labelpad=1)
ax.set_ylabel("training loss", labelpad=1)
ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=9)

# (b) 中位数（点线）与最小值–最大值（竖线）
ax = axs[1]
series = [("network output", cnet, "o", [(0, "teacher_v_train"), (1, "teacher_v_interp"), (2, "teacher_v_extrap")], -0.05),
          ("NE-PODG", cne, "s", [(1, "VS_v_interp"), (2, "VS_v_extrap")], 0.05)]
for name, c, mk, items, dx in series:
    xs, med = [], []
    for x, f in items:
        e = pd.read_csv(res / f"{f}.csv").e
        xs.append(x + dx)
        med.append(e.median())
        ax.plot([x + dx] * 2, [e.min(), e.max()], color=c, lw=1.0, solid_capstyle="butt", alpha=0.55)
        for v in (e.min(), e.max()):
            ax.plot([x + dx - 0.04, x + dx + 0.04], [v, v], color=c, lw=1.0, alpha=0.55)
    ax.plot(xs, med, color=c, marker=mk, ms=4.5, lw=1.0, label=name, zorder=3)
ax.set(yscale="log", xlim=(-0.4, 2.4), ylim=(5e-11, 5))
ax.set_xticks(range(3), ["training snapshots", "interpolation", "extrapolation"])
ax.tick_params(axis="x", length=0, pad=4)
ax.tick_params(axis="y", which="both", right=True)
ax.tick_params(axis="y", length=2.5, pad=2)
ax.tick_params(axis="y", which="minor", length=1.5)
ax.set_ylabel(r"$e$", labelpad=1)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7, columnspacing=1.6)
ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, va="top", fontsize=9)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=400)
