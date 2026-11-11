"""全部工况的高保真温度增量 u = T − ℓ_h；训练池点另解 ∂u/∂η₁、∂u/∂η₂（复用 A 的 LU 分解）。
η₁ = log10 Pe − 2、η₂ = (log10 ρ − 0.5)/1.5：∂/∂η₁ = ln10·Pe ∂/∂Pe，∂/∂η₂ = 1.5 ln10·ρ ∂/∂ρ。
每个工况存 data/<编号>.npz：Pe、ratio、u（单元，k = i·ny + j）；训练池另有 u_e1、u_e2。
用法（仓库根目录）：python -m cases.conjugate.fom.snapshots"""
import os
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.conjugate.case import ROOT, Conjugate, cases

out = ROOT / "data"
out.mkdir(exist_ok=True)
g = Conjugate()
t0 = time.perf_counter()
stat = {}
for r in cases():
    mu = float(r["Pe"]), float(r["ratio"])
    d = dict(Pe=mu[0], ratio=mu[1])
    if r["role"] == "pool":
        T, TPe, Tr = g.ch.solve(*mu, sens=True)
        d["u_e1"], d["u_e2"] = np.log(10) * mu[0] * TPe, 1.5 * np.log(10) * mu[1] * Tr
    else:
        T = g.ch.solve(*mu)
    d["u"] = T - g.ell
    np.savez(out / f"{r['id']}.npz", **d)
    stat.setdefault(r["role"], []).append((np.sqrt(g.ch.Mh @ d["u"] ** 2), np.sqrt(g.ch.Mh @ T ** 2), T.min(), T.max()))
print(f"{sum(len(s) for s in stat.values())} 个工况，{time.perf_counter() - t0:.0f} s → {out}")
for role, s in stat.items():
    s = np.array(s)
    print(f"{role}: ‖u‖ {s[:, 0].min():.3g}–{s[:, 0].max():.3g}，‖T‖ {s[:, 1].min():.3g}–{s[:, 1].max():.3g}，"
          f"T ∈ [{s[:, 2].min():.2g}, {s[:, 3].max():.4g}]")
