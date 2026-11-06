"""网格收敛：Pr = 0.1、1、10，沿 Ra = 10 → 10⁶ 延拓，各网格（N，s）在每个 Ra 上记录平均 Nu、壁面最大局部 Nu、|ψ|max、
竖直中线 u_max、水平中线 v_max。结果 cases/nc/results/fom/grid_study.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.fom.grid_study"""
import csv
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from solvers.mac.mac import Cavity, nodes

RA = [10, 31.6, 100, 316, 1000, 3160, 1e4, 3.16e4, 1e5, 3.16e5, 1e6]
GRIDS = [(64, 2.5), (96, 2.5), (128, 2.0), (128, 2.5), (128, 3.0), (192, 2.5), (256, 2.0), (256, 2.5), (256, 3.0),
         (384, 2.5), (512, 2.5)]


def peak(t, f):
    """样条插值后的最大值。"""
    cs, k = CubicSpline(t, f), np.argmax(f)
    return float(-minimize_scalar(lambda z: -cs(z), bounds=(t[max(k - 1, 0)], t[min(k + 1, len(t) - 1)]), method="bounded").fun)


out = Path(__file__).resolve().parents[1] / "results" / "fom" / "grid_study.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["N", "s", "Ra", "Pr", "Nu", "Nu_loc_max", "psi_max", "u_max", "v_max", "iters", "time"])
    for n, s in GRIDS:
        x = nodes(n, s)
        cav = Cavity(x, x, thermal=True)
        for pr in (0.1, 1, 10):
            psi = T = None
            for ra in RA:
                t0 = time.perf_counter()
                psi, T, st = cav.solve(np.sqrt(pr / ra), 1 / np.sqrt(ra * pr), psi, T)
                assert st["converged"], (n, s, ra, pr)
                nc, _, nu, _ = cav.nusselt(T)
                u, v = cav.faces(psi)
                h = n // 2
                row = [n, s, ra, pr, nu, nc.max(), np.abs(cav.psi_full(psi)).max(),
                       peak(np.r_[0, cav.yc, 1], np.r_[0, u[h], 0]), peak(np.r_[0, cav.xc, 1], np.r_[0, v[:, h], 0]),
                       st["iters"], time.perf_counter() - t0]
                wr.writerow(row)
                fh.flush()
        print(f"N={n} s={s} done", flush=True)
