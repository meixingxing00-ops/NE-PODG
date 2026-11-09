"""导数监督（Sobolev）训练：快照值 + 各训练点稳态灵敏度共同监督候选方向网络（candidates.Net，结构与主线相同）。
中心化读出 f(η) = admissible(χ(X; η) w(η))，û = ū + f。损失（M 范数，经 whiten）：
  L = mean_i ‖L(f(η_i) − (u_i − ū))‖² / ‖L u_i‖² + λ · mean_{i,k} ‖L(∂f/∂η_k(η_i) − s_ik)‖² / ‖L s_ik‖²
s_i1 = 2.25 ∂ψ/∂log10 Ra，s_i2 = ∂ψ/∂log10 Pr（η₁ = (log10 Ra − 3.25)/2.25，η₂ = log10 Pr）；∂f/∂η 用前向模式自动微分（torch.func.jacfwd）。
步数、学习率与降学习率、批量、种子与主线 train.py 相同。模型存 explore/models/n<B>sob.pt。
用法：python -m cases.nc.explore.sob B [steps] [λ]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacfwd, vmap

from cases.nc.case import NC
from cases.nc.explore.common import Budget
from nepodg import candidates

HERE = Path(__file__).resolve().parent
STEPS, LR, BATCH, SEED = 100000, 1e-3, 8, 1234
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS
lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
nc = NC()
bd = Budget(nc, budget)
U = bd.U
mean = U.mean(1)
D = U - mean[:, None]
Ns = U.shape[1]
S = bd.Sraw.reshape(-1, Ns, 2) * torch.as_tensor([2.25, 1.0], device=U.device, dtype=U.dtype)   # (n, Ns, 2)：对 η 的导数
E = torch.as_tensor(np.array(bd.etas), dtype=torch.float32, device=U.device)
su = torch.linalg.norm(nc.whiten(U), dim=0) ** 2
ss = torch.stack([torch.linalg.norm(nc.whiten(S[:, :, k]), dim=0) ** 2 for k in range(2)], 1)  # (Ns, 2)


def value_and_grads(net, e):
    """e (b, 2) → f (n, b) 与 ∂f/∂η (n, b, 2)，容许坐标，float64；对样本与两个切向一次向量化前向模式求导。"""
    def g(x):
        y = net.fields(nc.X, x) @ net.readout(x)
        return y, y
    J, y = vmap(jacfwd(g, has_aux=True))(e)                                    # (b, P, 2)、(b, P)
    b = e.shape[0]
    return nc.admissible(y.T.double()), nc.admissible(J.permute(1, 0, 2).reshape(-1, 2 * b).double()).reshape(-1, b, 2)


def check(net):
    """训练点上值与灵敏度的相对误差（M 范数）中位数。"""
    with torch.no_grad():
        f, df = value_and_grads(net, E)
        ev = torch.linalg.norm(nc.whiten(f - D), dim=0) / su.sqrt()
        es = torch.stack([torch.linalg.norm(nc.whiten(df[:, :, k] - S[:, :, k]), dim=0) for k in range(2)], 1) / ss.sqrt()
    return float(ev.median()), float(es.median())


torch.manual_seed(SEED)
gen = np.random.default_rng(SEED)
net = candidates.Net(**NET).to(U.device)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, 3 * steps // 4], 0.1)
hist, acc, t0, log = [], np.zeros(2), time.perf_counter(), max(1, steps // 20)
for it in range(1, steps + 1):
    idx = torch.as_tensor(gen.choice(Ns, min(BATCH, Ns), replace=False), device=U.device)
    f, df = value_and_grads(net, E[idx])
    lv = (torch.linalg.norm(nc.whiten(f - D[:, idx]), dim=0) ** 2 / su[idx]).mean()
    ls = torch.stack([torch.linalg.norm(nc.whiten(df[:, :, k] - S[:, idx, k]), dim=0) ** 2 / ss[idx, k] for k in range(2)]).mean()
    loss = lv + lam * ls
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    acc += [float(lv), float(ls)]
    if it % log == 0:
        hist.append(tuple(acc / log))
        print(it, f"{acc[0] / log:.3e} {acc[1] / log:.3e}", f"{time.perf_counter() - t0:.0f}s", flush=True)
        acc[:] = 0
net.eval()
print("训练点相对误差中位数（值、灵敏度）", " ".join(f"{v:.3e}" for v in check(net)))
(HERE / "models").mkdir(exist_ok=True)
candidates.save(HERE / "models" / f"n{budget}sob{'' if lam == 1.0 else f'_l{lam:g}'}.pt", net, hist)
print("done")
