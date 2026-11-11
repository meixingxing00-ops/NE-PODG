"""全部工况的高保真解；训练池点另求对 η 的稳态灵敏度（实例化设计 §5）。按 Ri 分组，沿 log10 Re 从 20 起延拓
（solvers.mac.study.advance），以前一个解为初值。每个工况存 data/<编号>.npz：Re、Ri、y = ψ − ψ_L（内部角点）、T（单元）、
Nu（冷壁平均）；训练池另有 psi_e1、psi_e2。q_z₁ = −ln10(ν q_ν + κ q_κ)，q_z₂ = ln10·Ri q_b，
q_η₁ = (log10 5/2) q_z₁，q_η₂ = (log10 50/2) q_z₂。
用法（仓库根目录，PINNGPU）：python -m cases.mixed.fom.snapshots"""
import os
import time
from collections import defaultdict

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.mixed.case import ROOT, cases, cavity, coefficients, stokes_lift
from solvers.mac.study import advance

out = ROOT / "data"
out.mkdir(exist_ok=True)
cav, psi_L = cavity(), stokes_lift()
mapping = lambda z: (lambda c: (c[0], c[1], dict(buoyancy=c[2])))(coefficients(10.0 ** np.asarray(z)))
groups = defaultdict(list)
for c in cases():
    groups[float(c["Ri"])].append(c)
t0, k = time.perf_counter(), 0
for ri, cs in sorted(groups.items()):
    psi = T = None
    z = np.log10([20.0, ri])
    for c in sorted(cs, key=lambda c: float(c["Re"])):
        mu = float(c["Re"]), ri
        target = np.log10(mu)
        psi, T, st = advance(cav, z, target, mapping, psi, T)
        z = target
        nu, kap, b = coefficients(mu)
        rec = dict(Re=mu[0], Ri=ri, y=psi - psi_L, T=T, Nu=cav.nusselt(T)[2], iters=st["total_iters"])
        if c["role"] == "pool":
            q = cav.sensitivity(psi, T, nu, kap, buoyancy=b, parameters=("nu", "kap", "buoyancy"))
            z1 = -np.log(10) * (nu * q["nu"] + kap * q["kap"])
            z2 = np.log(10) * b * q["buoyancy"]
            rec.update(psi_e1=(np.log10(5) / 2 * z1)[:cav.npsi], psi_e2=(np.log10(50) / 2 * z2)[:cav.npsi])
        np.savez(out / f"{c['id']}.npz", **rec)
        k += 1
print(f"{k} 个工况，{time.perf_counter() - t0:.0f} s → {out}")
