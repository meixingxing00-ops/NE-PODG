"""论文 Fig. 4、Fig. 5 数据：预算 N<B> 下单个工况的全场与剖面。方法：高保真（fom）、POD–Galerkin（pod）、
SA-PODG（sapodg）、NE-PODG（VS_vm256）与网络直接输出（teacher，以其速度在全网格上解能量方程得温度）。
全场：角点流函数 psi（含边界）、单元中心速度大小 U 与温度 T（[i, j]，x 在前）；剖面：x = 0.5 处 u 沿 y、y = 0.5 处 v 沿 x
（含壁面零值），热壁局部 Nu 沿 y。结果 results/n<B>/profiles_<编号>.npz。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.profiles B 编号"""
import sys

import numpy as np
import torch

from cases.nc.case import MODELS, NC, ROOT, N, cases, eta, nu_kap
from nepodg import candidates, closure, space

budget, cid = int(sys.argv[1]), sys.argv[2]
nc = NC()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
train = [nc.load(i) for i in ids]
pod = space.POD(torch.stack([s[0] for s in train], 1), nc.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
sa, _, _ = space.sensitivity_space(pod, nc.sens(ids).reshape(nc.n, -1), nc.whiten)
net = candidates.load(MODELS / "vm256" / f"n{budget}.pt", nc.X.device)
psi_r, T_r, mu = nc.load(cid)
e_ = eta(mu)

sols = {"fom": (psi_r.cpu().numpy(), T_r)}
for name, base in [("pod", pod), ("sapodg", sa), ("VS_vm256", sa)]:
    B = base.V
    if name == "VS_vm256":
        Z = space.complement(candidates.directions(net, nc.X, e_, nc.admissible), nc.whiten, sa)[0]
        B = torch.cat([B, Z], 1)
    c0 = np.concatenate([init(e_), np.zeros(B.shape[1] - pod.V.shape[1])])
    y, T, st = closure.solve(nc.reduced, pod.mean, B, mu, c0)
    assert st["converged"], name
    sols[name] = (y.cpu().numpy(), T.cpu().numpy())
et = torch.as_tensor(e_, dtype=torch.float32, device=nc.X.device)
with torch.no_grad():
    y = nc.admissible((net.fields(nc.X, et) @ net.readout(et)).double())
sols["teacher"] = (y.cpu().numpy(), nc.op.temperature(y, nu_kap(mu)[1]).cpu().numpy())

cav, h = nc.cav, N // 2
out = {"Ra": mu[0], "Pr": mu[1], "y": np.r_[0, cav.yc, 1], "x": np.r_[0, cav.xc, 1], "yc": cav.yc, "nodes": cav.x}
for k, (psi, T) in sols.items():
    u, v = cav.faces(psi)
    out[f"psi_{k}"] = cav.psi_full(psi)
    out[f"U_{k}"] = np.hypot((u[:-1] + u[1:]) / 2, (v[:, :-1] + v[:, 1:]) / 2)
    out[f"T_{k}"] = T.reshape(N, N)
    out[f"u_{k}"] = np.r_[0, u[h], 0]
    out[f"v_{k}"] = np.r_[0, v[:, h], 0]
    out[f"Nu_{k}"] = cav.nusselt(T)[1]
    print(k, f"|u|max {np.abs(out[f'u_{k}']).max():.5f}  |v|max {np.abs(out[f'v_{k}']).max():.5f}  Nu_H {cav.nusselt(T)[3]:.5f}")
path = ROOT / "results" / f"n{budget}" / f"profiles_{cid}.npz"
np.savez(path, **out)
print(f"→ {path}")
