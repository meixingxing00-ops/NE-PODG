"""论文 Fig. 1：六个算例的物理模型与边界条件（示意，长度不按比例）。(a) 方腔自然对流；(b) Couette–Graetz；(c) 通道共轭传热；
(d) 顶盖驱动混合对流；(e) 多孔方腔自然对流；(f) 后台阶流动传热。红、蓝分别为加热与冷却（或定温）边界，灰色斜线为固壁。
用法（仓库根目录）：python paper/figures/script/fig1_model.py [输出路径，默认 paper/figures/Fig_01.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 7.5,
                            "pdf.fonttype": 42})
cm = plt.get_cmap("RdBu_r")
HOT, COLD, WALL, SOLID = cm(0.9), cm(0.1), "0.2", cm(0.62)
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_01.pdf"


def wall(ax, p, q, side, color=WALL, lw=0.9, hatch=True):
    """p→q 的壁面；side 为斜线所在一侧的单位法向。"""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw if color == WALL else 2.0, solid_capstyle="butt", zorder=3)
    if hatch:
        n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / 0.07))
        for t in np.linspace(0, 1, n + 1)[:-1]:
            x0, y0 = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
            ax.plot([x0, x0 + 0.04 * side[0] + 0.03 * np.sign(q[0] - p[0] + q[1] - p[1])],
                    [y0, y0 + 0.04 * side[1] + 0.0], color="0.55", lw=0.4, zorder=2)


def arrow(ax, p, q, lw=0.7, ms=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color="0.15", zorder=4))


def gravity(ax, x, y):
    arrow(ax, (x, y), (x, y - 0.18))
    ax.text(x + 0.03, y - 0.09, r"$\mathbf{g}$", ha="left", va="center")


def cavity(ax, lid=False, porous=False):
    if porous:
        rng = np.random.default_rng(3)
        ax.scatter(rng.uniform(0.05, 0.95, 160), rng.uniform(0.05, 0.95, 160), s=4, c=["0.6"], ec="none", alpha=0.7, zorder=1)
    wall(ax, (0, 0), (0, 1), (-1, 0), COLD, hatch=False)
    wall(ax, (1, 0), (1, 1), (1, 0), HOT, hatch=False)
    wall(ax, (0, 0), (1, 0), (0, -1))
    if lid:
        ax.plot([0, 1], [1, 1], color=WALL, lw=0.9, zorder=3)
        arrow(ax, (0.3, 1.07), (0.7, 1.07))
        ax.text(0.5, 1.11, r"$u=1,\ \partial_yT=0$", ha="center", va="bottom")
    else:
        wall(ax, (0, 1), (1, 1), (0, 1))
    ax.text(-0.05, 0.5, r"$T=-\frac{1}{2}$", rotation=90, ha="right", va="center", color=COLD)
    ax.text(1.05, 0.5, r"$T=+\frac{1}{2}$", rotation=90, ha="left", va="center", color=HOT)
    ax.text(0.5, -0.1, r"$\partial_yT=0$", ha="center", va="top")
    if not lid:
        ax.text(0.5, 1.1, r"$\partial_yT=0$", ha="center", va="bottom")
    gravity(ax, 0.85, 0.9 if not lid else 0.82)
    ax.set(xlim=(-0.3, 1.3), ylim=(-0.3, 1.32))


mm = 1 / 25.4
fig, axs = plt.subplots(2, 3, figsize=(190 * mm, 92 * mm), gridspec_kw=dict(left=0.01, right=0.99, bottom=0.02, top=0.97,
                                                                             wspace=0.06, hspace=0.12))
axs = axs.ravel()
# (a) nc
cavity(axs[0])
axs[0].text(0.45, 0.5, r"$Ra,\ Pr$", ha="center", va="center", fontsize=8.5)
# (d) mixed
cavity(axs[3], lid=True)
axs[3].text(0.45, 0.5, r"$Re,\ Ri$" + "\n" + r"$Pr=0.71$", ha="center", va="center", fontsize=8.5)
# (e) porous
cavity(axs[4], porous=True)
axs[4].text(0.45, 0.5, r"$Ra^\ast,\ Da$" + "\n" + r"$Pr=0.71$", ha="center", va="center", fontsize=8.5,
            bbox=dict(fc="white", ec="none", pad=0.5))
# (b) graetz：x ∈ [0, 1 + L]，示意长度 3
ax = axs[1]
Lx = 3.2
wall(ax, (0, 1), (Lx, 1), (0, 1), COLD)
wall(ax, (0, 0), (1, 0), (0, -1))
wall(ax, (1, 0), (Lx, 0), (0, -1), HOT, hatch=False)
ax.plot([0, 0], [0, 1], color="0.5", lw=0.6, ls=":")
ax.plot([Lx, Lx], [0, 1], color="0.5", lw=0.6, ls=":")
for y in (0.25, 0.5, 0.75):
    arrow(ax, (0.12, y), (0.12 + 0.5 * y, y), lw=0.6, ms=5)
ax.plot([0.12, 0.62], [0, 1], color="0.4", lw=0.5)
ax.text(0.7, 0.86, r"$u=y$", ha="left", va="center")
ax.text(Lx / 2, 1.08, r"$T=0$", ha="center", va="bottom", color=COLD)
ax.text(-0.06, 0.5, r"$T=0$", ha="right", va="center", rotation=90)
ax.text(0.5, -0.09, r"$\partial_nT=0$", ha="center", va="top")
ax.text((1 + Lx) / 2, -0.09, r"$Pe^{-1}\partial_nT=1$", ha="center", va="top", color=HOT)
ax.text(Lx + 0.06, 0.5, r"$\partial_nT=0$", ha="left", va="center", rotation=90)
arrow(ax, (1, -0.42), (Lx, -0.42), lw=0.5, ms=4)
arrow(ax, (Lx, -0.42), (1, -0.42), lw=0.5, ms=4)
ax.text((1 + Lx) / 2, -0.47, r"$L$", ha="center", va="top")
ax.text(2.1, 0.5, r"$L,\ Pe$", ha="center", va="center", fontsize=8.5)
ax.set(xlim=(-0.35, Lx + 0.35), ylim=(-0.75, 1.4))
# (c) conjugate：流体 (0,5)×(0,1)，固体 (0,5)×(−0.25,0)，示意长度 3.4
ax = axs[2]
Lx, d = 3.4, 0.3
ax.add_patch(Rectangle((0, -d), Lx, d, fc=SOLID, ec="none", alpha=0.45, zorder=0))
wall(ax, (0, 1), (Lx, 1), (0, 1))
xs = np.linspace(0, Lx, 60)
g = np.where(xs / Lx * 5 <= 1, np.sin(np.pi * np.minimum(xs / Lx * 5, 1) / 2) ** 2, 1.0)
for i in range(len(xs) - 1):
    ax.plot(xs[i:i + 2], [-d, -d], color=cm(0.5 + 0.4 * g[i]), lw=2.0, solid_capstyle="butt", zorder=3)
ax.plot([0, Lx], [0, 0], color="0.3", lw=0.6, ls="--")
for y in (0.2, 0.5, 0.8):
    arrow(ax, (0.1, y), (0.1 + 0.55 * 4 * y * (1 - y), y), lw=0.6, ms=5)
ys = np.linspace(0, 1, 30)
ax.plot(0.1 + 0.55 * 4 * ys * (1 - ys), ys, color="0.4", lw=0.5)
ax.text(0.75, 0.82, r"$u=6y(1-y)$", ha="left", va="center")
ax.text(Lx / 2, 1.08, r"$\partial_yT=0$", ha="center", va="bottom")
ax.text(Lx / 2, -d - 0.07, r"$T_s=g(x)$", ha="center", va="top", color=HOT)
ax.text(Lx / 2, -d / 2, r"solid, $k_s$", ha="center", va="center")
ax.text(-0.06, 0.38, r"$T=0$", ha="right", va="center", rotation=90)
ax.text(Lx + 0.06, 0.38, r"$\partial_xT=0$", ha="left", va="center", rotation=90)
ax.text(Lx * 0.62, 0.45, r"fluid, $k_f$" + "\n" + r"$Pe,\ k_s/k_f$", ha="center", va="center", fontsize=8)
ax.set(xlim=(-0.35, Lx + 0.35), ylim=(-0.75, 1.4))
# (f) bfs：入口 (−4,0)×(0,1)，下游 (0,40)×(−S,1)，示意长度入口 0.9、下游 3.6
ax = axs[5]
xi, xo, S = -0.9, 3.0, 0.8
wall(ax, (xi, 1), (xo, 1), (0, 1))
wall(ax, (xi, 0), (0, 0), (0, -1))
wall(ax, (0, 0), (0, -S), (-1, 0))
wall(ax, (0, -S), (xo, -S), (0, -1), HOT, hatch=False)
ax.plot([xi, xi], [0, 1], color="0.5", lw=0.6, ls=":")
ax.plot([xo, xo], [-S, 1], color="0.5", lw=0.6, ls=":")
ys = np.linspace(0, 1, 30)
ax.plot(xi + 0.08 + 0.45 * 4 * ys * (1 - ys), ys, color="0.4", lw=0.5)
for y in (0.25, 0.5, 0.75):
    arrow(ax, (xi + 0.08, y), (xi + 0.08 + 0.45 * 4 * y * (1 - y), y), lw=0.6, ms=5)
t = np.linspace(0, 2 * np.pi, 80)
ax.plot(0.75 + 0.7 * np.cos(t), -S / 2 + 0.06 + 0.24 * np.sin(t), color="0.45", lw=0.5, ls="--")
ax.text(xi - 0.06, 0.5, r"$T=0$", ha="right", va="center", rotation=90)
ax.text((xi + xo) / 2, 1.08, r"$\partial_yT=0$", ha="center", va="bottom")
ax.text(1.6, -S - 0.07, r"$T=1$", ha="center", va="top", color=HOT)
ax.text(xo + 0.06, 0.1, "outflow", ha="left", va="center", rotation=90)
arrow(ax, (-0.12, 0), (-0.12, -S), lw=0.5, ms=4)
arrow(ax, (-0.12, -S), (-0.12, 0), lw=0.5, ms=4)
ax.text(-0.17, -S / 2, r"$S$", ha="right", va="center")
ax.text(2.1, 0.45, r"$Re,\ ER$" + "\n" + r"$Pr=0.71$", ha="center", va="center", fontsize=8.5)
arrow(ax, (0, -S + 0.1), (1.45, -S + 0.1), lw=0.5, ms=4)
arrow(ax, (1.45, -S + 0.1), (0, -S + 0.1), lw=0.5, ms=4)
ax.text(0.72, -S + 0.14, r"$x_r$", ha="center", va="bottom", fontsize=7.5)
ax.set(xlim=(xi - 0.35, xo + 0.35), ylim=(-S - 0.55, 1.4))
for ax, lab in zip(axs, "abcdef"):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.0, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
fig.savefig(out, dpi=300)
print(f"→ {out}")
