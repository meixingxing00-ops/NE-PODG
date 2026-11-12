"""论文 Fig. 6 数据：预算 N<B> 下单个工况的高保真（fom）、SA-PODG（sapodg）与 NE-PODG（VS_v）的流场、温度与受热壁局部 Nu。
存物理节点 x、y（随 ER），角点流函数 psi_<方法>（[i, j]，非活动角点为 nan），单元中心速度大小 U_<方法> 与温度 T_<方法>（非活动单元为 nan），
受热壁面中心 xw 上的局部 Nu_<方法>（长度 S）与下壁剪切 shear_<方法>（单元中心 xs）。结果 results/n<B>/fields_<编号>.npz。
用法（仓库根目录，PINNGPU）：python -m cases.bfs.scripts.fields B 编号"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.bfs import case
from nepodg import closure, experiment

budget, cid = int(sys.argv[1]), sys.argv[2]
g = case.BFS()
m, gr = g.m, g.m.grid


def rom(y0, B, mu, c0):
    q, _, st = closure.solve(g.reduced, y0, B, mu, c0)
    return q, g.temperature(q, mu), st


q_r, T_r, mu = g.load(cid)
mu = np.asarray(mu)
sols = {"fom": (q_r, T_r)}
for k in ("sapodg", "VS_v"):
    q, T, st = experiment.solve(case, g, budget, cid, k, rom)
    assert st["res"] <= 1e-6, (k, st)
    sols[k] = (q, T)
op = m.operators(mu)
S = mu[1] - 1
out = dict(Re=mu[0], ER=mu[1], x=np.asarray(op.x), y=np.asarray(op.y))
for k, (q, T) in sols.items():
    y = g.state(q)
    u, v = m.faces(y, mu)
    U = np.hypot(0.5 * (u[:-1] + u[1:]), 0.5 * (v[:, :-1] + v[:, 1:]))
    Tc = np.full((gr.nx, gr.ny), np.nan)
    Tc[gr.ci, gr.cj] = T
    psi = np.full((gr.nx + 1) * (gr.ny + 1), np.nan)
    full = m.full(y, mu)
    psi[np.unique(gr.ends)] = full[np.unique(gr.ends)]
    nu = m.nusselt(y, T, mu, "heated", S, 0.)
    rec = m.recirculation(y, mu)
    out.update({f"psi_{k}": psi.reshape(gr.nx + 1, gr.ny + 1), f"U_{k}": np.where(np.isnan(Tc), np.nan, U), f"T_{k}": Tc,
                f"Nu_{k}": nu["Nu"], f"shear_{k}": rec["shear"]})
    out["xw"], out["xs"] = nu["x"], rec["x"]
    print(k, f"x_r/S {rec['primary_reattachment'] / S:.4f}  Nu_avg {nu['average']:.5f}")
path = case.ROOT / "results" / f"n{budget}" / f"fields_{cid}.npz"
np.savez(path, **out)
print(f"→ {path}")
