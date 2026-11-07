"""训练池 50 个快照的稳态灵敏度（方法设计 §2.2）：在高保真解处解 J ∂q/∂ν = −∂R/∂ν、J ∂q/∂κ = −∂R/∂κ（solvers/mac，Cavity.sensitivity），
经链式法则得对数参数的导数
  ∂q/∂log10 Ra = −(ln10/2)(ν ∂q/∂ν + κ ∂q/∂κ)，∂q/∂log10 Pr = (ln10/2)(ν ∂q/∂ν − κ ∂q/∂κ)，q = [ψ, T]。
每个快照存 data/sens/<编号>.npz：Ra、Pr，以及内部角点流函数与单元中心温度对 log10 Ra、log10 Pr 的导数
psi_Ra、psi_Pr（(N−1)²）、T_Ra、T_Pr（N²）。对参数坐标 η 的换算见 case.NC.sens。
用法（仓库根目录，PINNGPU）：python -m cases.nc.fom.sensitivities"""
import time

import numpy as np

from cases.nc.case import N, ROOT, S, SENS, cases, nu_kap
from solvers.mac.mac import Cavity, nodes

SENS.mkdir(exist_ok=True)
x = nodes(N, S)
cav = Cavity(x, x, thermal=True)
c = np.log(10) / 2
t0 = time.perf_counter()
pool = [r for r in cases() if r["role"] == "pool"]
for r in pool:
    d = np.load(ROOT / "data" / f"{r['id']}.npz")
    psi, T, mu = d["psi"][1:-1, 1:-1].ravel(), d["T"].ravel(), (float(d["Ra"]), float(d["Pr"]))
    nu, kap = nu_kap(mu)
    dnu, dkap = cav.sensitivity(psi, T, nu, kap)
    qRa, qPr = -c * (nu * dnu + kap * dkap), c * (nu * dnu - kap * dkap)
    n = cav.npsi
    np.savez(SENS / f"{r['id']}.npz", Ra=mu[0], Pr=mu[1], psi_Ra=qRa[:n], psi_Pr=qPr[:n], T_Ra=qRa[n:], T_Pr=qPr[n:])
print(f"{len(pool)} 个快照，{time.perf_counter() - t0:.0f} s → {SENS}")
