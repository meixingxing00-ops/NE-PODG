"""闭合：基础空间 V⊕S（SA 空间）上再加网络方向，其余（闭合、初值、误差）与 cases/nc/scripts/run.py 相同。space：
  sa            V⊕S
  sane          V⊕S⊕Z，Z 为查询点网络方向去 V⊕S 分量
  sanefix       V⊕S⊕Z，Z 为 Ra 最高训练点处的网络方向（固定）
  vne           V⊕Z，Z 为查询点网络方向去 V 分量（即 NE-PODG）
  <名>:<模型>   以 explore/models/<模型>.pt 代替 cases/nc/models/n<B>_v.pt，<名> 为 sane、sanefix 或 vne
  vpoly<d>      V⊕P，P 为总次数 ≤ d 的多项式场 𝓑·P_a(x)P_b(y)（固定）
  sapoly<d>     V⊕S⊕P
结果 explore/results/n<B>/<space>_<role>.csv（模型名中的冒号写作下划线）。
用法（仓库根目录，PINNGPU）：python -m cases.nc.explore.run B role space"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import NC, cases, eta
from cases.nc.explore.common import Budget, cat, extend, poly
from nepodg import candidates, closure, metrics

HERE = Path(__file__).resolve().parent
budget, role, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
nc = NC()
bd = Budget(nc, budget)
kind, _, model = mode.partition(":")
if kind.startswith("sane") or kind == "vne":
    net = candidates.load(HERE / "models" / f"{model}.pt", nc.X.device)[0] if model else bd.net()
dirs = lambda e: candidates.directions(net, nc.X, e, nc.admissible)
init = closure.Init(bd.etas, bd.pod.a)
if kind.startswith("vpoly") or kind == "vne":
    bd.VS = bd.Vb
if "poly" in kind:
    fixed = extend(bd.VS, poly(nc, int(kind.split("poly")[1])), nc.whiten)[:2]
    augment = lambda mu: fixed
elif kind == "sa":
    augment = lambda mu: None
elif kind in ("sane", "vne"):
    augment = lambda mu: extend(bd.VS, dirs(eta(mu)), nc.whiten)[:2]
else:
    E = np.array(bd.etas)
    fixed = extend(bd.VS, dirs(E[E[:, 0].argmax()]), nc.whiten)[:2]
    augment = lambda mu: fixed
Mp = nc.cav.Mp
out = HERE / "results" / f"n{budget}" / f"{mode.replace(':', '_')}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "role", "Ra", "Pr", "nB", "e", "e_proj", "e_real", "e_T", "e_Nu", "converged", "iters", "time"])
    for c in [c for c in cases() if c["role"] == role]:
        psi_r, T_r, mu = nc.load(c["id"])
        t0 = time.perf_counter()
        Z = augment(mu)
        B, Bw = cat(bd.VS, Z) if Z is not None else bd.VS
        c0 = np.concatenate([init(eta(mu)), np.zeros(B.shape[1] - bd.pod.V.shape[1])])
        y, T, st = closure.solve(nc.reduced, bd.pod.mean, B, mu, c0)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(y, psi_r, bd.pod.mean, Bw, nc.whiten, nc.lift)
        T = T.cpu().numpy()
        eT = np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2))
        nu, nu_r = nc.cav.nusselt(T)[2], nc.cav.nusselt(T_r)[2]
        wr.writerow([c["id"], role, c["Ra"], c["Pr"], B.shape[1], e, ep, er, eT, abs(nu / nu_r - 1),
                     st["converged"], st["iters"], t])
        fh.flush()
print(f"→ {out}")
