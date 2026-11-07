"""受热段平均温度 T_av(L, Pe) 曲面，与 Quarteroni, Rozza & Manzoni (2011) 图 8 左图同坐标（线性 L ∈ [1, 10]、Pe ∈ [0.1, 100]）作定性对照。
结果 results/fom/tav_surface.png。用法（仓库根目录）：python -m cases.graetz.fom.tav_surface"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from cases.graetz.case import ROOT, mesh

ch = mesh()
L, Pe = np.linspace(1, 10, 31), np.linspace(0.1, 100, 31)
Z = np.array([[ch.tav(ch.solve(l, p), p) for l in L] for p in Pe])
cmap = ListedColormap(plt.cm.RdBu_r(np.linspace(0.5, 1, 256)))
fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(projection="3d")
X, Y = np.meshgrid(L, Pe)
s = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="k", linewidth=0.2)
ax.set_xlabel("$L$")
ax.set_ylabel("$Pe$")
ax.set_zlabel(r"$T_{\mathrm{av}}$")
ax.view_init(30, -135)
fig.colorbar(s, shrink=0.6)
fig.tight_layout()
fig.savefig(ROOT / "results" / "fom" / "tav_surface.png", dpi=150)
print(f"T_av {Z.min():.4g}–{Z.max():.4g}")
