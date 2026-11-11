"""全部工况的高保真解；训练池点另求对 η 的稳态灵敏度（实例化设计 §5）。按 Da 分组，沿 log10 Ra* 从 10 起延拓
（solvers.mac.study.advance），以前一个解为初值。每个工况存 data/<编号>.npz：Ras、Da、psi（内部角点）、T（单元）、Nu（冷壁平均）；
训练池另有 psi_e1、psi_e2。η 的灵敏度：a = (ln10/2)(ν q_ν + κ q_κ + d q_d)，q_η₁ = −0.75 a，q_η₂ = a − ln10·d q_d。
用法（仓库根目录，PINNGPU）：python -m cases.porous.fom.snapshots"""
import os
import time
from collections import defaultdict

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.porous.case import ROOT, cases, cavity, coefficients
from solvers.mac.study import advance

out = ROOT / "data"
out.mkdir(exist_ok=True)
cav = cavity()
mapping = lambda z: (lambda c: (c[0], c[1], dict(drag=c[2])))(coefficients(10.0 ** np.asarray(z)))
groups = defaultdict(list)
for c in cases():
    groups[float(c["Da"])].append(c)
t0, k = time.perf_counter(), 0
for da, cs in sorted(groups.items()):
    psi = T = None
    z = np.array([1.0, np.log10(da)])
    for c in sorted(cs, key=lambda c: float(c["Ras"])):
        mu = float(c["Ras"]), da
        target = np.log10(mu)
        psi, T, st = advance(cav, z, target, mapping, psi, T)
        z = target
        nu, kap, d = coefficients(mu)
        rec = dict(Ras=mu[0], Da=da, psi=psi, T=T, Nu=cav.nusselt(T)[2], iters=st["total_iters"])
        if c["role"] == "pool":
            q = cav.sensitivity(psi, T, nu, kap, parameters=("nu", "kap", "drag"), drag=d)
            a = np.log(10) / 2 * (nu * q["nu"] + kap * q["kap"] + d * q["drag"])
            e1, e2 = -0.75 * a, a - np.log(10) * d * q["drag"]
            rec.update(psi_e1=e1[:cav.npsi], psi_e2=e2[:cav.npsi])
        np.savez(out / f"{c['id']}.npz", **rec)
        k += 1
print(f"{k} 个工况，{time.perf_counter() - t0:.0f} s → {out}")
