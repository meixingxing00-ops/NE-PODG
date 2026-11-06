"""高保真解：cases.csv 全部工况，网格 nodes(128, 2.0)。每个 Pr 沿 log10 Ra 延拓：经过训练池的 Ra 档（间隔 0.5 个数量级）
直到目标 Ra，以前一个解为初值。每个工况存 cases/nc/data/<id>.npz：
  Ra、Pr、N、s、x、y（网格节点）、psi（角点，(N+1, N+1)，边界为 0）、T（单元中心，(N, N)）、
  u（竖直面，(N+1, N)）、v（水平面，(N, N+1)）、Nu_C、Nu_H（冷、热壁局部 Nu，(N,)）、Nu_C_mean、Nu_H_mean、
  iters、res、converged。数组下标为 [i, j]（x 方向在前）。
用法（仓库根目录，PINNGPU）：python -m cases.nc.fom.snapshots"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from solvers.mac.mac import Cavity, nodes

N, S = 128, 2.0
LADDER = [10, 31.6, 100, 316, 1000, 3160, 1e4, 3.16e4, 1e5, 3.16e5]

root = Path(__file__).resolve().parents[1]
(root / "data").mkdir(exist_ok=True)
cases = list(csv.DictReader(open(root / "cases.csv", encoding="utf-8")))
by_pr = defaultdict(list)
for c in cases:
    by_pr[float(c["Pr"])].append(c)

x = nodes(N, S)
cav = Cavity(x, x, thermal=True)
t0 = time.perf_counter()
for pr, cs in sorted(by_pr.items()):
    target = {float(c["Ra"]): c for c in cs}
    psi = T = None
    for ra in sorted(set([r for r in LADDER if r < max(target)]) | set(target)):
        psi, T, st = cav.solve(np.sqrt(pr / ra), 1 / np.sqrt(ra * pr), psi, T)
        assert st["converged"], (ra, pr)
        if ra in target:
            u, v = cav.faces(psi)
            nc, nh, ncm, nhm = cav.nusselt(T)
            np.savez(root / "data" / f"{target[ra]['id']}.npz", Ra=ra, Pr=pr, N=N, s=S, x=x, y=x,
                     psi=cav.psi_full(psi), T=T.reshape(N, N), u=u, v=v, Nu_C=nc, Nu_H=nh, Nu_C_mean=ncm, Nu_H_mean=nhm,
                     iters=st["iters"], res=st["res"], converged=st["converged"])
print(f"{len(cases)} 个工况，{time.perf_counter() - t0:.0f} s")
