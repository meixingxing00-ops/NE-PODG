"""N6 稀疏快照下的候选方向网络：快照监督 + 无快照参数点上的离散方程残差。
网络：ψ 网络（candidates.Net，角点采样，经容许映射）与 T 网络（同结构，单元中心采样），读出 û = ū + Φ w，T̂ = T̄ + Φ_T w_T。
离散残差与 MAC 求解器相同（Cavity.residual）：
  r_ψ = Cᵀ[S_m((A_m w)⊙(H_m w)) + ν(K_m w − b_m) − B_t T]，R_T = S_e((A_e w)⊙(H_e T)) + κ(K_e T − b_e)，w = C ψ
以固定的线性部分预条件：e_ψ = (Cᵀ K_m C)⁻¹ r_ψ / ν，e_T = K_e⁻¹ R_T / κ（低 Ra 时即误差的一阶估计）；
逐点相对量 p = ‖L e_ψ‖² / ‖L ψ̂‖² + ‖e_T‖²_Mp / ‖T̂‖²_Mp。
  sup：只用快照，ψ 损失同 candidates.train，T 同形式；存 models/n6sup.pt（ψ）与 n6sup_T.pt
  phys：从 n6sup 继续，损失 log(数据项) + λ·mean log p，残差点每步在 log10 Ra ∈ [1, 6]、log10 Pr ∈ [−1, 1] 均匀重采样；
        存 models/n6phys.pt 与 n6phys_T.pt
用法：python -m cases.nc.explore.phys sup|phys [steps]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import N, NC, eta, nu_kap
from cases.nc.explore.common import Budget
from nepodg import candidates
from solvers.mac.gpu import DEV, _t

HERE = Path(__file__).resolve().parent
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)
LR, SEED, NCOL, LAM, RA, PR = 1e-3, 1234, 8, 1.0, (1.0, 6.0), (-1.0, 1.0)

mode = sys.argv[1]
nc = NC()
cav = nc.cav
bd = Budget(nc, 6)
U = bd.U
T = torch.as_tensor(np.stack([s[1] for s in bd.snaps], 1), device=DEV)
mean, Tmean = U.mean(1), T.mean(1)
E = torch.as_tensor(np.array(bd.etas), dtype=torch.float32, device=DEV)
xc = torch.as_tensor(cav.xc)
Xc = torch.stack(torch.meshgrid(2 * xc - 1, 2 * xc - 1, indexing="ij"), -1).reshape(-1, 2).float().to(DEV)
sqMp = torch.as_tensor(np.sqrt(cav.Mp), device=DEV)

C, Sm, Am, Hm, Km, Bt = (_t(M) for M in (cav.C, cav.mo.S, cav.mo.A, cav.mo.H, cav.mo.K, cav.Bt))
Se, Ae, He, Ke = (_t(M) for M in (cav.en.S, cav.en.A, cav.en.H, cav.en.K))
CT = _t(cav.C.T)
bm, be = (torch.as_tensor(b, device=DEV) for b in (cav.mo.b, cav.en.b))
Lp = torch.linalg.cholesky(torch.as_tensor(cav.CKC.toarray(), device=DEV))
Lt = torch.linalg.cholesky(torch.as_tensor(cav.en.K.toarray(), device=DEV))
torch.cuda.empty_cache()                                       # 稠密 Cholesky 因子各约 2 GB，释放构造时的临时矩阵


def residual(psi, Th, nu, kap):
    """psi (n, b)、Th (nT, b)、nu、kap (b,) → 预条件残差 e_ψ、e_T。"""
    w = C @ psi
    Rw = Sm @ ((Am @ w) * (Hm @ w)) + nu * (Km @ w - bm[:, None])
    rp = CT @ (Rw - Bt @ Th)
    RT = Se @ ((Ae @ w) * (He @ Th)) + kap * (Ke @ Th - be[:, None])
    return torch.cholesky_solve(rp, Lp) / nu, torch.cholesky_solve(RT, Lt) / kap


def predict(net, tnet, e):
    """e (b, 2) → ψ̂ (n, b)、T̂ (nT, b)。"""
    P = torch.stack([nc.admissible((net.fields(nc.X, x) @ net.readout(x)).double()) for x in e], 1)
    Q = torch.stack([(tnet.fields(Xc, x) @ tnet.readout(x)).double() for x in e], 1)
    return mean[:, None] + P, Tmean[:, None] + Q


def data_loss(net, tnet):
    P, Q = predict(net, tnet, E)
    lp = (torch.linalg.norm(nc.whiten(P - U), dim=0) ** 2 / torch.linalg.norm(nc.whiten(U), dim=0) ** 2).mean()
    lt = (torch.linalg.norm(sqMp[:, None] * (Q - T), dim=0) ** 2 / torch.linalg.norm(sqMp[:, None] * T, dim=0) ** 2).mean()
    return lp, lt


def phys_loss(net, tnet, gen):
    lra, lpr = gen.uniform(*RA, NCOL), gen.uniform(*PR, NCOL)
    mus = [(10 ** a, 10 ** b) for a, b in zip(lra, lpr)]
    e = torch.as_tensor(np.array([eta(m) for m in mus]), dtype=torch.float32, device=DEV)
    nk = torch.as_tensor(np.array([nu_kap(m) for m in mus]), device=DEV)
    P, Q = predict(net, tnet, e)
    ep, et = residual(P, Q, nk[:, 0], nk[:, 1])
    p = torch.linalg.norm(nc.whiten(ep), dim=0) ** 2 / torch.linalg.norm(nc.whiten(P), dim=0).detach() ** 2 + \
        torch.linalg.norm(sqMp[:, None] * et, dim=0) ** 2 / torch.linalg.norm(sqMp[:, None] * Q, dim=0).detach() ** 2
    return torch.log(p).mean()


def save(name, net, tnet, hist):
    candidates.save(HERE / "models" / f"{name}.pt", net, mean, hist)
    candidates.save(HERE / "models" / f"{name}_T.pt", tnet, Tmean, hist)


# 残差核对：高保真快照处 e_ψ、e_T 应为求解器容差量级，与 Cavity.residual 一致
nk = torch.as_tensor(np.array([nu_kap(s[2]) for s in bd.snaps]), device=DEV)
ep, et = residual(U, T, nk[:, 0], nk[:, 1])
r0 = cav.residual(1.1 * U[:, -1].cpu().numpy(), 0.9 * T[:, -1].cpu().numpy(), *nu_kap(bd.snaps[-1][2]))  # 扰动状态
w0 = C @ (1.1 * U[:, -1:])
rp0 = (CT @ (Sm @ ((Am @ w0) * (Hm @ w0)) + nk[-1, 0] * (Km @ w0 - bm[:, None]) - Bt @ (0.9 * T[:, -1:])))[:, 0].cpu().numpy()
print("快照处相对预条件残差", [f"{float(a):.1e}" for a in torch.linalg.norm(nc.whiten(ep), dim=0) / torch.linalg.norm(nc.whiten(U), dim=0)],
      "与 Cavity.residual 差", f"{np.abs(rp0 - r0[0]).max() / np.abs(r0[0]).max():.1e}", flush=True)

steps = int(sys.argv[2]) if len(sys.argv) > 2 else (100000 if mode == "sup" else 20000)
torch.manual_seed(SEED)
gen = np.random.default_rng(SEED)
(HERE / "models").mkdir(exist_ok=True)
if mode == "sup":
    net, tnet = candidates.Net(**NET).to(DEV), candidates.Net(**NET).to(DEV)
    lr = LR
else:
    net = candidates.load(HERE / "models" / "n6sup.pt", DEV)[0].train()
    tnet = candidates.load(HERE / "models" / "n6sup_T.pt", DEV)[0].train()
    lr = LR / 100
opt = torch.optim.Adam(list(net.parameters()) + list(tnet.parameters()), lr=lr)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, 3 * steps // 4], 0.1)
hist, t0, log = [], time.perf_counter(), max(1, steps // 20)
for it in range(1, steps + 1):
    lp, lt = data_loss(net, tnet)
    if mode == "sup":
        loss = lp + lt
        rec = (float(lp), float(lt))
    else:
        lf = phys_loss(net, tnet, gen)
        loss = torch.log(lp + lt) + LAM * lf
        rec = (float(lp), float(lt), float(lf))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    hist.append(rec)
    if it % log == 0:
        m = np.mean(hist[-log:], 0)
        print(it, " ".join(f"{v:.3e}" for v in m), f"{time.perf_counter() - t0:.0f}s", flush=True)
save("n6sup" if mode == "sup" else "n6phys", net.eval(), tnet.eval(), hist)
print("done")
