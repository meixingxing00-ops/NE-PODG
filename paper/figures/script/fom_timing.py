"""论文附录 C 数据：六个算例内插测试工况上高保真单次求解的耗时。每个工况从默认预算中参数坐标 η 最近的训练快照出发求解到收敛
（非线性算例为 Newton，容差与生成快照时相同；线性算例为直接法），计时只含求解（非线性算例含辅助场）。另记迭代数、收敛与否，以及主场与
存档参考解的相对差（按算例度量）。输出 paper/figures/data/fom_timing.csv。用法（仓库根目录，PINNGPU）：python paper/figures/script/fom_timing.py"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
import torch

from cases.bfs import case as cb
from cases.conjugate import case as cc
from cases.graetz import case as cg
from cases.mixed import case as cm
from cases.nc import case as cn
from cases.porous import case as cp

DEFAULT = {"nc": 12, "graetz": 4, "conjugate": 4, "mixed": 4, "porous": 4, "bfs": 4}


def solver(name):
    """返回 (算例模块, 算例对象, f(初值快照, μ) → (主场 numpy, 迭代数, 收敛))。"""
    if name == "nc":
        g = cn.NC()

        def f(s, mu):
            nu, kap = cn.nu_kap(mu)
            psi, T, st = g.cav.solve(nu, kap, s[0].cpu().numpy(), s[1])
            return psi, st["iters"], st["converged"]
        return cn, g, f
    if name == "graetz":
        g = cg.Graetz()
        return cg, g, lambda s, mu: (g.ch.solve(*mu), 1, True)
    if name == "conjugate":
        g = cc.Conjugate()
        return cc, g, lambda s, mu: (g.ch.solve(*mu) - g.ell, 1, True)
    if name == "mixed":
        g = cm.Mixed()
        psiL = g.psi_L.cpu().numpy()

        def f(s, mu):
            nu, kap, ri = cm.coefficients(mu)
            psi, T, st = g.cav.solve(nu, kap, psiL + s[0].cpu().numpy(), s[1], tol=1e-10, maxit=35, buoyancy=ri)
            return psi - psiL, st["iters"], st["converged"]
        return cm, g, f
    if name == "porous":
        g = cp.Porous()

        def f(s, mu):
            nu, kap, d = cp.coefficients(mu)
            psi, T, st = g.cav.solve(nu, kap, s[0].cpu().numpy(), s[1], tol=1e-10, maxit=35, drag=d)
            return psi, st["iters"], st["converged"]
        return cp, g, f
    g = cb.BFS()

    def f(s, mu):
        y, T, st = g.m.solve(np.asarray(mu), state=g.state(s[0]), tol=1e-9)
        return y - g.ell_np, st.get("iterations", st.get("iters", np.nan)), st["converged"]
    return cb, g, f


rows = []
for name, b in DEFAULT.items():
    case, g, f = solver(name)
    ids = [c["id"] for c in case.cases() if c[f"n{b}"] == "1"]
    snaps = [g.load(i) for i in ids]
    E = np.array([case.eta(s[-1]) for s in snaps])
    f(snaps[0], snaps[0][-1])                                   # 预热（GPU、分解缓存）
    for c in [c for c in case.cases() if c["role"] == "interp"]:
        ref = g.load(c["id"])
        mu = ref[-1]
        s = snaps[int(np.argmin(np.linalg.norm(E - case.eta(mu), axis=1)))]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q, it, ok = f(s, mu)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        q = torch.as_tensor(np.asarray(q), device=ref[0].device)
        d = float(torch.linalg.norm(g.whiten(q - ref[0])) / torch.linalg.norm(g.whiten(ref[0])))
        rows.append(dict(case=name, id=c["id"], time=t, iters=it, converged=ok, diff=d))
    r = pd.DataFrame([x for x in rows if x["case"] == name])
    print(f"{name:9s} median {1e3 * r.time.median():.4g} ms  iters med {r.iters.median():.0f}  "
          f"converged {int(r.converged.sum())}/{len(r)}  max diff {r['diff'].max():.1e}", flush=True)
    del g
    torch.cuda.empty_cache()
out = root / "paper" / "figures" / "data" / "fom_timing.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"→ {out}")
