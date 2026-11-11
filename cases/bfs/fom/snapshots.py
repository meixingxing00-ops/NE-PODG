"""全部工况的高保真解；训练池点另求对 η 的稳态灵敏度（实例化设计 §5）。按 cases.csv 的顺序逐点延拓（prepare.reach），
以前一个解为初值。每个工况存 data/<编号>.npz：Re、ER、q = y − ℓ_S（自由角点，case.stokes_lift）、T（活动单元）、xr_S、Nu_max；
训练池另有 q_e1、q_e2（= ∂y/∂η，提升与参数无关）。
用法（仓库根目录，PINNGPU）：python -m cases.bfs.fom.snapshots"""
import os
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.bfs.case import ROOT, cases, stokes_lift
from cases.bfs.fom.config import deta, model
from cases.bfs.fom.grid_study import outputs
from cases.bfs.fom.prepare import reach

out = ROOT / "data"
out.mkdir(exist_ok=True)
m = model()
ell = stokes_lift(m)
y = prev = None
t0 = time.perf_counter()
for c in cases():
    mu = np.array([float(c["Re"]), float(c["ER"])])
    y, T, st, _ = reach(m, mu, y, prev)
    assert st["converged"], c
    o = outputs(m, y, T, mu)
    rec = dict(Re=mu[0], ER=mu[1], q=y - ell, T=T, xr_S=o["xr_S"], Nu_max=o["Nu_max"])
    if c["role"] == "pool":
        sy, _, _ = m.sensitivity(y, T, mu)
        sy = sy * deta(mu)
        rec.update(q_e1=sy[:, 0], q_e2=sy[:, 1])
    np.savez(out / f"{c['id']}.npz", **rec)
    prev = mu
print(f"{len(cases())} 个工况，{time.perf_counter() - t0:.0f} s → {out}")
