"""公共核心在 N50 上的核对（不含网络）：
1. POD 重构训练快照；训练快照的 POD 系数代入约化方程的残差；辅助场温度与快照温度一致。
2. 约化雅可比与中心差分。
3. 互补空间的退化情形：候选方向在 span V 内时为空，与 span V 正交时全部保留。
4. 只用 POD 空间的闭合：训练点与若干测试点，误差 e、e_proj、e_real 与耗时。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.core_checks"""
import time

import numpy as np
import torch

from cases.nc.case import NC, cases, eta, nu_kap
from nepodg import closure, metrics, space

nc = NC()
pool = [c for c in cases() if c["n50"] == "1"]
snaps = [nc.load(c["id"]) for c in pool]
U = torch.stack([s[0] for s in snaps], 1)
pod = space.POD(U, nc.whiten)
print(f"N50：n_V = {pod.V.shape[1]}，σ_min/σ_1 = {float(pod.sigma[pod.V.shape[1] - 1] / pod.sigma[0]):.1e}")

# 1
rec, rg, dT = [], [], []
for i, (psi, T, mu) in enumerate(snaps):
    y = pod.mean + pod.V @ pod.a[:, i]
    rec.append(float(torch.linalg.norm(nc.whiten(y - psi)) / torch.linalg.norm(nc.whiten(psi))))
    g, _, Tr = nc.reduced(y, pod.V, mu)
    g0, _, _ = nc.reduced(pod.mean, pod.V, mu)
    rg.append(float(torch.linalg.norm(g) / torch.linalg.norm(g0)))
    dT.append(float(np.abs(Tr.cpu().numpy() - T).max()))
print(f"1. 重构相对误差 ≤ {max(rec):.1e}；‖g(a_i)‖/‖g(0)‖ ≤ {max(rg):.1e}；max|S(ψ_i) − T_i| ≤ {max(dT):.1e}")

# 2
rng = np.random.default_rng(0)
psi, _, mu = nc.load("I050")
c, d, e = rng.standard_normal(pod.V.shape[1]) * 1e-3, rng.standard_normal(pod.V.shape[1]), 1e-6
gt = lambda c: nc.reduced(pod.mean + pod.V @ torch.as_tensor(c, device=pod.V.device), pod.V, mu)
G = gt(c)[1].cpu().numpy()
fd = (gt(c + e * d)[0] - gt(c - e * d)[0]).cpu().numpy() / (2 * e)
print(f"2. 约化雅可比与中心差分相对差 {np.linalg.norm(fd - G @ d) / np.linalg.norm(G @ d):.1e}")

# 3
R = torch.randn(nc.n, 10, dtype=torch.float64, device=pod.V.device)
Z, _, s = space.complement(pod.V @ R[:pod.V.shape[1]], nc.whiten, pod)
Rp = R - pod.V @ (pod.Vw.T @ nc.whiten(R))
Z2, Zw2, s2 = space.complement(Rp, nc.whiten, pod)
print(f"3. 候选方向在 span V 内：n_Z = {Z.shape[1]}（s_max = {float(s.max()):.1e}）；与 span V 正交：n_Z = {Z2.shape[1]}，"
      f"s ∈ [{float(s2.min()):.6f}, {float(s2.max()):.6f}]，|V_wᵀ Z_w| ≤ {float((pod.Vw.T @ Zw2).abs().max()):.1e}")

# 4
init = closure.Init([eta(m) for _, _, m in snaps], pod.a)
print("4. POD 空间闭合（初值为 POD 系数插值）：")
for cid in ["P001", "P025", "P050", "I001", "I030", "I060", "I090", "E001", "E014", "E027"]:
    psi_r, _, mu = nc.load(cid)
    t0 = time.perf_counter()
    y, _, st = closure.solve(nc.reduced, pod.mean, pod.V, mu, init(eta(mu)))
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    err = metrics.errors(y, psi_r, pod.mean, pod.Vw, nc.whiten, nc.lift)
    print(f"   {cid}  Ra {mu[0]:<8g} Pr {mu[1]:<5g} conv {st['converged']} iters {st['iters']:>2}  "
          f"e {err[0] * 100:.4f}%  e_proj {err[1] * 100:.4f}%  e_real {err[2] * 100:.4f}%  {t:.2f} s")

# 耗时：k = n_V + 128 个方向
B = torch.cat([pod.V, torch.randn(nc.n, 128, dtype=torch.float64, device=pod.V.device)], 1)
nc.reduced(pod.mean, B, mu)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    nc.reduced(pod.mean, B, mu)
torch.cuda.synchronize()
print(f"单次 reduced（k = {B.shape[1]}）：{(time.perf_counter() - t0) / 10 * 1000:.1f} ms")
