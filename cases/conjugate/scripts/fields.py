"""论文 Fig. 6 数据：预算 N<B> 下单个工况的高保真（fom）、SA-PODG（sapodg）与 NE-PODG（VS_v）温度场与界面热流、局部 Nu。
存节点 x、y（y < 0 为固体），单元温度 T_<方法>（[i, j]），单元中心 xc 上的界面热流 q_<方法> 与局部 Nu nu_<方法>。
结果 results/n<B>/fields_<编号>.npz。用法（仓库根目录，PINNGPU）：python -m cases.conjugate.scripts.fields B 编号"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.conjugate import case
from nepodg import experiment

budget, cid = int(sys.argv[1]), sys.argv[2]
g = case.Conjugate()
ch = g.ch
u_r, mu = g.load(cid)
rom = lambda y0, B, mu_, c0: (g.solve(y0, B, mu_), None, None)
sols = {"fom": u_r, **{m: experiment.solve(case, g, budget, cid, m, rom)[0] for m in ("sapodg", "VS_v")}}
out = dict(Pe=mu[0], ratio=mu[1], x=ch.x, y=ch.y, xc=ch.xc)
for k, u in sols.items():
    o = g.outputs(u, mu)
    out[f"T_{k}"] = g.temperature(u).reshape(ch.nx, ch.ny)
    out[f"q_{k}"], out[f"nu_{k}"] = o["q"], o["nu"]
    print(k, f"Q {o['Q']:.6f}  Tw {o['Tw']:.6f}  Tout {o['Tout']:.6f}")
path = case.ROOT / "results" / f"n{budget}" / f"fields_{cid}.npz"
np.savez(path, **out)
print(f"→ {path}")
