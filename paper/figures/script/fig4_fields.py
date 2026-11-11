"""论文 Fig. 4：N12、Ra = 10⁶、Pr = 0.562（E022）的高保真速度大小（流线）与温度（等温线），以及 POD–Galerkin、SA-PODG、
NE-PODG 的速度误差 |U − U_ref|/max|U_ref| 与温度误差 (T − T_ref)/ΔT（百分数，线性色标，同行共用）。
数据 cases/nc/results/n12/profiles_E022.npz（cases/nc/scripts/profiles.py），整体误差取 cases/nc/results/n12/<方法>_extrap.csv。
用法（仓库根目录）：python paper/figures/script/fig4_fields.py [输出路径，默认 paper/figures/Fig_04.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "pdf.fonttype": 42})
rdbu = plt.get_cmap("RdBu_r")                       # 两端截去，避免最深处接近黑色
cm = LinearSegmentedColormap.from_list("rdbu", rdbu(np.linspace(0.08, 0.92, 256)))
reds = LinearSegmentedColormap.from_list("reds", rdbu(np.linspace(0.5, 0.9, 256)))
root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results" / "n12"
d = np.load(res / "profiles_E022.npz")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_04.pdf"
x = d["nodes"]
methods = [("pod", "POD–Galerkin"), ("sapodg", "SA-PODG"), ("VS_v", "NE-PODG")]
summ = {m: pd.read_csv(res / f"{m}_extrap.csv").set_index("id").loc["E022"] for m, _ in methods}

mm = 1 / 25.4
fig = plt.figure(figsize=(190 * mm, 82 * mm))
w, h = 34 / 190, 34 / 82
cols = [0.05, 0.335, 0.524, 0.713]
rows = [0.515, 0.03]
cbw = 0.010


def box(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])


def colorbar(sc, x0, y0, label, ticks=None):
    cb = fig.colorbar(sc, cax=fig.add_axes([x0, y0, cbw, h]))
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    cb.outline.set_linewidth(0.6)


Uref, Tref = d["U_fom"], d["T_fom"]
# 误差以百分数画线性色标：速度为误差大小（红色半段），温度为带号误差（RdBu_r，以 0 为中心）；超出范围的取端色
spec = [("U", Uref, Normalize(0, Uref.max()), reds, r"$|\mathbf{U}|$", np.abs(d["psi_fom"]),
         np.linspace(0, np.abs(d["psi_fom"]).max(), 9)[1:-1], Normalize(0, 5), reds,
         lambda a, b: np.abs(a - b) / Uref.max() * 100,
         r"$|\mathbf{U}-\mathbf{U}_{\mathrm{ref}}|/\max|\mathbf{U}_{\mathrm{ref}}|$ (%)", "e", [0, 1, 2, 3, 4, 5]),
        ("T", Tref, Normalize(-0.5, 0.5), cm, r"$T$", Tref, np.linspace(-0.4, 0.4, 9), Normalize(-2, 2), cm,
         lambda a, b: (a - b) * 100, r"$(T-T_{\mathrm{ref}})/\Delta T$ (%)", "e_T", [-2, -1, 0, 1, 2])]
for r, (q, ref, norm, cmap, lab, iso, levels, enorm, ecmap, ef, elab, ek, eticks) in enumerate(spec):
    y0 = rows[r]
    ax = fig.add_axes([cols[0], y0, w, h])
    sc = ax.pcolormesh(x, x, ref.T, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    xi = x if q == "U" else (x[:-1] + x[1:]) / 2
    ax.contour(xi, xi, iso.T, levels=levels, colors="k", linewidths=0.4, linestyles="solid")
    box(ax)
    if r == 0:
        ax.set_title("high-fidelity", fontsize=8, pad=3)
    ax.text(-0.08, 0.5, "velocity" if q == "U" else "temperature", rotation=90, ha="right", va="center",
            transform=ax.transAxes, fontsize=8)
    colorbar(sc, cols[0] + w + 0.008, y0, lab, None if q == "U" else [-0.5, -0.25, 0, 0.25, 0.5])
    for k, (m, name) in enumerate(methods):
        ax = fig.add_axes([cols[k + 1], y0, w, h])
        sc = ax.pcolormesh(x, x, ef(d[f"{q}_{m}"], ref).T, cmap=ecmap, norm=enorm, shading="flat", rasterized=True)
        box(ax)
        if r == 0:
            ax.set_title(name, fontsize=8, pad=3)
        val = summ[m][ek]
        sym = r"$e$" if ek == "e" else r"$e_T$"
        ax.text(0.04, 0.95, f"{sym} = {100 * val:.2g}%", transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    colorbar(sc, cols[3] + w + 0.008, y0, elab, eticks)

out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=400)
