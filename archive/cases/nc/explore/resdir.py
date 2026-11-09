"""方程残差生成的增广方向（N6，不用网络）：先在 V⊕S 上闭合得 ψ⁰，再由离散残差生成方向、加入空间、重新闭合，迭代 K 次。
  visc：d = (Cᵀ K_m C)⁻¹ r_ψ(ψ, T)（固定的粘性算子预条件，分解一次，在线代价为一次三角求解）
  newton：d = [J(ψ, T)⁻¹ R(ψ, T)]_ψ（完整耦合雅可比的 Newton 步，每次一次稀疏 LU，作为"一步方程信息"的上限）
T 取闭合给出的辅助温度（满足能量方程，R_T ≈ 0）。只算 Ra > 10³ 的测试工况。
结果 explore/results/resdir_n6.csv：每种方向每次迭代后的闭合误差 e 与最佳逼近误差 e_proj。
用法：python -m cases.nc.explore.resdir"""
import csv
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spl
import torch

from cases.nc.case import NC, eta, nu_kap
from cases.nc.explore.common import Budget, cat, extend, tests
from nepodg import closure, metrics

HERE = Path(__file__).resolve().parent
K = 3
nc = NC()
cav = nc.cav
bd = Budget(nc, 6)
init = closure.Init(bd.etas, bd.pod.a)
lu_visc = spl.splu(cav.CKC.tocsc())
n = cav.npsi


def direction(kind, psi, T, mu):
    nu, kap = nu_kap(mu)
    if kind == "visc":
        return lu_visc.solve(cav.residual(psi, T, nu, kap)[0])
    r = np.concatenate(cav.residual(psi, T, nu, kap))
    return spl.splu(cav.jac_full(psi, T, nu, kap).tocsc()).solve(r)[:n]


cols = ["id", "role", "Ra", "Pr", "e0", "ep0"] + [f"{k}_{x}{i}" for k in ("visc", "newton") for i in range(1, K + 1) for x in ("e", "ep")]
with open(HERE / "results" / "resdir_n6.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        if float(c["Ra"]) <= 1e3:
            continue
        psi_r, T_r, mu = nc.load(c["id"])
        B0, Bw0 = bd.VS
        c0 = np.concatenate([init(eta(mu)), np.zeros(B0.shape[1] - bd.pod.V.shape[1])])
        y0, T0, _ = closure.solve(nc.reduced, bd.pod.mean, B0, mu, c0)
        e, ep, _ = metrics.errors(y0, psi_r, bd.pod.mean, Bw0, nc.whiten, nc.lift)
        row = [c["id"], c["role"], c["Ra"], c["Pr"], e, ep]
        for kind in ("visc", "newton"):
            base, y, T = bd.VS, y0, T0
            for i in range(K):
                d = torch.as_tensor(direction(kind, y.cpu().numpy(), T.cpu().numpy(), mu), device=y.device)
                base = cat(base, extend(base, d[:, None], nc.whiten)[:2])
                B, Bw = base
                cc = (Bw.T @ nc.whiten(y - bd.pod.mean)).cpu().numpy()          # 上一解在新基下的系数
                y, T, st = closure.solve(nc.reduced, bd.pod.mean, B, mu, cc)
                e, ep, _ = metrics.errors(y, psi_r, bd.pod.mean, Bw, nc.whiten, nc.lift)
                row += [e if st["converged"] else np.nan, ep]
        wr.writerow(row)
        fh.flush()
        print(c["id"], c["Ra"], c["Pr"], " ".join(f"{v:.2e}" for v in row[4::2]), flush=True)
print(f"→ {HERE / 'results' / 'resdir_n6.csv'}")
