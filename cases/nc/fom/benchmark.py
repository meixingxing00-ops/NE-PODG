"""论文 Table 1：Pr = 0.71，Ra = 10³–10⁶ 的平均 Nu（算例网格 nodes(128, 2.0)），及 Ra = 10⁶ 下 128、256、512 网格（s = 2.0）的平均 Nu。
结果 cases/nc/results/fom/benchmark.csv。用法（仓库根目录，PINNGPU）：python -m cases.nc.fom.benchmark"""
import csv
from pathlib import Path

import numpy as np

from solvers.mac.mac import Cavity, nodes

PR = 0.71
out = Path(__file__).resolve().parents[1] / "results" / "fom" / "benchmark.csv"
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["N", "s", "Ra", "Pr", "Nu_C", "Nu_H"])
    for n in (128, 256, 512):
        x = nodes(n, 2.0)
        cav = Cavity(x, x, thermal=True)
        psi = T = None
        for k in np.arange(1.0, 6.01, 0.5):
            ra = 10 ** k
            psi, T, st = cav.solve(np.sqrt(PR / ra), 1 / np.sqrt(ra * PR), psi, T)
            assert st["converged"], (n, ra)
            if k in (3, 4, 5, 6) and (n == 128 or k == 6):
                _, _, nc, nh = cav.nusselt(T)
                wr.writerow([n, 2.0, ra, PR, nc, nh])
                print(n, ra, nc, nh, flush=True)
