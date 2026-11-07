"""空间变形的上限检验（oracle）：把训练快照（及其灵敏度）经变形 x → φ_θ(x) 输运后加到 V⊕S 上，
θ 对每个测试工况用目标本身拟合，给出"已知最优变形"时的最佳逼近误差。
变形在计算坐标 (ξ, ζ) ∈ [0, 1]²（均匀，物理网格为其 tanh 加密像）上定义：
  ξ' = ξ + Σ_{k=1..K, l=0..L} a_kl sin(kπξ) cos(lπζ)，ζ' 同理（a、b 两组），保持四壁不动、壁面切向可滑移；
ψ 在壁上为 0 且法向导数为 0，故按壁面偶延拓（reflection）做双三次插值，变形后仍满足两类边界条件。
变体（每个都是 V⊕S 上新增的维数）：
  near1_s：最近训练快照，可分离变形（L = 0），1 维
  near1：最近训练快照，一般变形，1 维
  near3：最近训练快照及其两个灵敏度，同一变形，3 维
  all：全部训练快照，同一变形，N_B 维
  far1：对照，最远训练快照，一般变形，1 维（检验增益来自输运还是来自对目标的拟合自由度）
只算 Ra > 10³ 的测试工况。结果 explore/results/warp_n<B>.csv（同时记拟合后的变形幅度 amp = max|位移|）。
用法：python -m cases.nc.explore.warp B"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cases.nc.case import N, NC, eta
from cases.nc.explore.common import Budget, eproj, target, tests
from solvers.mac.gpu import DEV

OUT = Path(__file__).resolve().parent / "results"
K, L, SEEDS = 4, 2, 3


def full(Y):
    """内部角点 (n, k) → (k, 1, N+1, N+1)，壁上补 0；张量下标 [x, y]。"""
    G = torch.zeros(Y.shape[1], N + 1, N + 1, dtype=Y.dtype, device=Y.device)
    G[:, 1:-1, 1:-1] = Y.T.reshape(-1, N - 1, N - 1)
    return G[:, None]


s = torch.linspace(0, 1, N + 1, dtype=torch.float64, device=DEV)
SX, SY = torch.meshgrid(s, s, indexing="ij")
kk = torch.arange(1, K + 1, dtype=torch.float64, device=s.device)
ll = torch.arange(0, L + 1, dtype=torch.float64, device=s.device)
# 基函数 (K, L+1, N+1, N+1)：x 位移 sin(kπξ)cos(lπζ)，y 位移 sin(kπζ)cos(lπξ)
BX = torch.sin(kk[:, None, None, None] * np.pi * SX) * torch.cos(ll[None, :, None, None] * np.pi * SY)
BY = torch.sin(kk[:, None, None, None] * np.pi * SY) * torch.cos(ll[None, :, None, None] * np.pi * SX)


def warp(G, th, sep):
    a, b = th.reshape(2, K, L + 1)
    if sep:
        a, b = a * (ll == 0), b * (ll == 0)
    dx, dy = (a[..., None, None] * BX).sum((0, 1)), (b[..., None, None] * BY).sum((0, 1))
    # grid_sample：grid[..., 0] 对应最后一维（y），grid[..., 1] 对应倒数第二维（x），坐标 [−1, 1]
    g = torch.stack([2 * (SY + dy) - 1, 2 * (SX + dx) - 1], -1)[None].expand(G.shape[0], -1, -1, -1)
    W = F.grid_sample(G, g, mode="bicubic", padding_mode="reflection", align_corners=True)
    return W[:, 0, 1:-1, 1:-1].reshape(G.shape[0], -1).T, float(torch.sqrt(dx ** 2 + dy ** 2).max())


def fit(G, r, VSw, sc, sep):
    """θ 使 r（目标去 V⊕S 后的余量）在变形场去 V⊕S 后的张成中误差最小；多起点 L-BFGS。"""
    def err(th):
        W = nc.whiten(warp(G, th, sep)[0])
        Q = torch.linalg.qr(W - VSw @ (VSw.T @ W))[0]
        return torch.linalg.norm(r - Q @ (Q.T @ r)) / sc
    best = (np.inf, None)
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        th = (0.02 * torch.randn(2 * K * (L + 1), dtype=torch.float64, device=s.device)).requires_grad_()
        opt = torch.optim.LBFGS([th], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            e = err(th)
            e.backward()
            return e
        opt.step(closure)
        with torch.no_grad():
            e = float(err(th))
        if e < best[0]:
            best = (e, th.detach())
    return best[0], warp(G, best[1], sep)[1]


budget = int(sys.argv[1])
nc = NC()
bd = Budget(nc, budget)
E = np.array(bd.etas)
VSw = bd.VS[1]
Gs = full(bd.U)
Gsens = full(bd.Sraw)
OUT.mkdir(exist_ok=True)
cols = ["id", "role", "Ra", "Pr", "dist", "VS", "near1_s", "near1", "near3", "all", "far1",
        "amp_near1", "amp_near3", "amp_all", "amp_far1"]
with open(OUT / f"warp_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        if float(c["Ra"]) <= 1e3:
            continue
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        d = np.linalg.norm(E - eta(mu), axis=1)
        j, jf = int(d.argmin()), int(d.argmax())
        r = dw - VSw @ (VSw.T @ dw)
        e1s, _ = fit(Gs[j:j + 1], r, VSw, sc, True)
        e1, a1 = fit(Gs[j:j + 1], r, VSw, sc, False)
        e3, a3 = fit(torch.cat([Gs[j:j + 1], Gsens[2 * j:2 * j + 2]]), r, VSw, sc, False)
        ea, aa = fit(Gs, r, VSw, sc, False)
        ef, af = fit(Gs[jf:jf + 1], r, VSw, sc, False)
        wr.writerow([c["id"], c["role"], c["Ra"], c["Pr"], d.min(), eproj(dw, sc, VSw), e1s, e1, e3, ea, ef, a1, a3, aa, af])
        fh.flush()
        print(c["id"], c["Ra"], c["Pr"], f"VS {eproj(dw, sc, VSw):.2e} near1 {e1:.2e} near3 {e3:.2e} all {ea:.2e} far1 {ef:.2e}", flush=True)
print(f"→ {OUT / f'warp_n{budget}.csv'}")
