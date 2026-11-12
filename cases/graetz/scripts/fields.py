"""论文 Fig. 6 数据：预算 N<B> 下单个工况的高保真（fom）、SA-PODG（sapodg）与 NE-PODG（VS_v）温度场与受热壁局部 Nu。
存物理节点 x（随 L）、y，单元温度 T_<方法>（[i, j]），受热单元中心 xw 与局部 Nu_<方法>。结果 results/n<B>/fields_<编号>.npz。
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.fields B 编号"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.graetz import case
from nepodg import experiment

budget, cid = int(sys.argv[1]), sys.argv[2]
g = case.Graetz()
ch = g.ch
T_r, mu = g.load(cid)
rom = lambda y0, B, mu_, c0: (g.solve(y0, B, mu_), None, None)
sols = {"fom": T_r, **{m: experiment.solve(case, g, budget, cid, m, rom)[0] for m in ("sapodg", "VS_v")}}
x, _ = ch.coords(mu[0])
xc = 0.5 * (x[1:] + x[:-1])
out = dict(L=mu[0], Pe=mu[1], x=x, y=ch.y, xw=xc[ch.i1:])
for k, T in sols.items():
    T = T.cpu().numpy()
    out[f"T_{k}"] = T.reshape(ch.nx, ch.ny)
    out[f"Nu_{k}"] = g.nusselt(T, mu)
    print(k, f"T_av {g.tav(T, mu):.6f}  Nu(x̂=2) {out[f'Nu_{k}'][-1]:.5f}")
path = case.ROOT / "results" / f"n{budget}" / f"fields_{cid}.npz"
np.savez(path, **out)
print(f"→ {path}")
