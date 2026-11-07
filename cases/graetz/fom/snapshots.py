"""全部工况的高保真温度场；训练池点另解 ∂T/∂η₁、∂T/∂η₂（复用 A 的 LU 分解）。
每个工况存 data/<编号>.npz：L、Pe、T（单元温度，k = i·ny + j）、Tav；训练池另有 T_e1、T_e2。
用法（仓库根目录）：python -m cases.graetz.fom.snapshots"""
import os
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.graetz.case import ROOT, cases, mesh

out = ROOT / "data"
out.mkdir(exist_ok=True)
ch = mesh()
t0 = time.perf_counter()
stat = {}
for r in cases():
    mu = float(r["L"]), float(r["Pe"])
    d = dict(L=mu[0], Pe=mu[1])
    if r["role"] == "pool":
        d["T"], d["T_e1"], d["T_e2"] = ch.solve(*mu, sens=True)
    else:
        d["T"] = ch.solve(*mu)
    d["Tav"] = ch.tav(d["T"], mu[1])
    b = ch.balance(d["T"], *mu)
    np.savez(out / f"{r['id']}.npz", **d)
    s = stat.setdefault(r["role"], [])
    s.append((np.sqrt(ch.Mh @ d["T"] ** 2), d["Tav"], d["T"].min() / d["T"].max(), abs(b[0] - sum(b[1:])) / b[0]))
print(f"{sum(len(s) for s in stat.values())} 个工况，{time.perf_counter() - t0:.0f} s → {out}")
for role, s in stat.items():
    s = np.array(s)
    print(f"{role}: ‖T‖ {s[:, 0].min():.3g}–{s[:, 0].max():.3g}，T_av {s[:, 1].min():.4g}–{s[:, 1].max():.4g}，"
          f"min T/max T ≥ {s[:, 2].min():.1e}，收支误差 ≤ {s[:, 3].max():.1e}")
