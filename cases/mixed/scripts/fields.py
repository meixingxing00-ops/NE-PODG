"""论文附录 C 数据：预算 N<B> 下单个工况的高保真（fom）、SA-PODG（sapodg）与 NE-PODG（VS_vm256）的流场、温度与冷壁局部 Nu。
存节点 x（两个方向相同）、单元中心 xc，完整流函数 psi_<方法>（ψ_L + y，含边界角点），单元中心速度大小 U_<方法>、温度 T_<方法>
（[i, j]）与冷壁局部 Nu_<方法>（沿 y）。结果 results/n<B>/fields_<编号>.npz。
用法（仓库根目录，PINNGPU）：python -m cases.mixed.scripts.fields B 编号"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.mixed import case
from nepodg import closure, experiment

budget, cid = int(sys.argv[1]), sys.argv[2]
g = case.Mixed()
cav = g.cav
rom = lambda y0, B, mu, c0: closure.solve(g.reduced, y0, B, mu, c0)
y_r, T_r, mu = g.load(cid)
sols = {"fom": (y_r, T_r)}
for k in ("sapodg", "VS_vm256"):
    y, T, st = experiment.solve(case, g, budget, cid, k, rom)
    assert st["res"] <= 1e-6, (k, st)
    sols[k] = (y, T.cpu().numpy())
n = len(cav.dx)
out = dict(Re=mu[0], Ri=mu[1], x=cav.x, xc=cav.xc)
for k, (y, T) in sols.items():
    psi = (g.psi_L + y).cpu().numpy()
    u, v = cav.faces(psi)
    out[f"psi_{k}"] = cav.psi_full(psi)
    out[f"U_{k}"] = np.hypot((u[:-1] + u[1:]) / 2, (v[:, :-1] + v[:, 1:]) / 2)
    out[f"T_{k}"] = np.asarray(T).reshape(n, n)
    nu = cav.nusselt(np.asarray(T))
    out[f"Nu_{k}"] = nu[0]
    print(k, f"Nu_C {nu[2]:.6f}")
path = case.ROOT / "results" / f"n{budget}" / f"fields_{cid}.npz"
np.savez(path, **out)
print(f"→ {path}")
