"""对流格式与网格收敛（实例化设计 §5.2）。
网格族 grid(40r, 1.5, 160r, 2.0, 80r, 2.0)，r = 0.5、1、2；参照为 r = 4 的 LUDS 解（25.6 万单元）。r = 1 上另比较 CDS、QUICK。
工况：Pe = 316 与 0.1 各取 L = 1、10，另加 L = 3.16、Pe = 10。误差：参考 L² 温度相对误差（参照解在单元中心上线性插值）、
T_av 相对误差、x̂ = 1.05、1.5、1.95 处局部 Nu 的最大相对误差；min T / max T 与沿流向温度下降的单元面数（振荡指标）。
结果 results/fom/grid_study.csv。用法（仓库根目录）：python -m cases.graetz.fom.grid_study"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from solvers.fv.fv import Channel, grid

ROOT = Path(__file__).resolve().parents[1]
XN = np.array([1.05, 1.5, 1.95])
CASES = [(1, 316), (10, 316), (1, 0.1), (10, 0.1), (3.16, 10)]
mesh = lambda r, sch: Channel(*grid(int(40 * r), 1.5, int(160 * r), 2.0, int(80 * r), 2.0), scheme=sch)
nu_at = lambda ch, T, Pe: np.interp(XN, ch.xhc[ch.i1:], ch.nusselt(T, Pe))

ref = {}
ch = mesh(4, "luds")
for L, Pe in CASES:
    T = ch.solve(L, Pe)
    ref[L, Pe] = (RegularGridInterpolator((ch.xhc, ch.yc), T.reshape(ch.nx, ch.ny), bounds_error=False, fill_value=None),
                  ch.tav(T, Pe), nu_at(ch, T, Pe))

out = ROOT / "results" / "fom" / "grid_study.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["scheme", "r", "cells", "L", "Pe", "Tav", "e_T", "e_Tav", "e_Nu", "Tmin_rel", "n_dec"])
    for sch, r in [("luds", 0.5), ("luds", 1), ("luds", 2), ("cds", 1), ("quick", 1), ("luds", 4)]:
        ch = mesh(r, sch)
        P = np.stack(np.meshgrid(ch.xhc, ch.yc, indexing="ij"), -1).reshape(-1, 2)
        for L, Pe in CASES:
            I, tav, nu = ref[L, Pe]
            T = ch.solve(L, Pe)
            Tr = I(P)
            Tm = T.reshape(ch.nx, ch.ny)
            wr.writerow([sch, r, ch.n, L, Pe, ch.tav(T, Pe), np.sqrt(ch.Mh @ (T - Tr) ** 2 / (ch.Mh @ Tr ** 2)),
                         abs(ch.tav(T, Pe) / tav - 1), np.abs(nu_at(ch, T, Pe) / nu - 1).max(), T.min() / T.max(),
                         int((np.diff(Tm, axis=0) < -1e-12 * T.max()).sum())])
            fh.flush()
print(f"→ {out}")
