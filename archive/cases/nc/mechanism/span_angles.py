"""候选空间随参数的变化与有效维数（cases/nc/实验记录.md「候选空间随参数的变化与有效维数」）：15 个参数点上白化 Φ 的奇异值分布、
15 个空间的并、与 (Ra, Pr) = (10, 1) 的主角正弦、float32 与 float64 求值的方向偏离。输出到屏幕。
用法（仓库根目录，PINNGPU）：PYTHONPATH=. python archive/cases/nc/mechanism/span_angles.py B kind..."""
import copy, sys
import numpy as np, torch
from cases.nc.case import MODELS, NC, cases, eta, kind
from nepodg import candidates, space

B, kinds = int(sys.argv[1]), sys.argv[2:]
nc = NC()
ids = [c["id"] for c in cases() if c[f"n{B}"] == "1"]
pod = space.POD(torch.stack([nc.load(i)[0] for i in ids], 1), nc.whiten)
base = space.sensitivity_space(pod, nc.sens(ids).reshape(nc.n, -1), nc.whiten)[0]
RA, PR = [10, 316, 1e4, 3.16e5, 1e6], [0.1, 1, 10]
G = [(ra, pr) for ra in RA for pr in PR]
cnt = lambda x, th: [int((x > t).sum()) for t in th]

def sines(Ua, Ub):  # Ua 各方向与 span Ub 的主角正弦（升序）
    return torch.linalg.svdvals(Ua - Ub @ (Ub.T @ Ua)).flip(0)

def union(Us):  # 并的奇异值平方：同一空间 → k 个 1；互不相关 → 约 1/len
    return torch.linalg.svdvals(torch.cat(Us, 1)) ** 2 / len(Us)

def orth(Phi):
    P, s, _ = torch.linalg.svd(nc.whiten(Phi).cpu(), full_matrices=False)
    return P[:, s / s[0] > 1e-8], s / s[0]

for k in kinds:
    trained, seed, m, sc = kind(k)
    if trained:
        net = candidates.load(MODELS / k / f"n{B}.pt", nc.X.device)
    else:
        torch.manual_seed(seed); net = candidates.Net(m=m); net.trunk[0].weight.data.mul_(sc); net = net.to(nc.X.device).eval()
    U, Z, SG = {}, {}, {}
    for mu in G:
        Phi = candidates.directions(net, nc.X, eta(mu), nc.admissible)
        U[mu], SG[mu] = orth(Phi)
        Z[mu] = space.complement(Phi, nc.whiten, base)[1].cpu()
    print(f"\n##### N{B} {k}")
    dec = np.array([[int(((s > lo) & (s <= hi)).sum()) for lo, hi in [(1e-2, 2), (1e-4, 1e-2), (1e-6, 1e-4), (0, 1e-6)]] for s in SG.values()])
    print("σ/σ1 各量级方向数 (>1e-2, 1e-4–1e-2, 1e-6–1e-4, <1e-6) 中位数:", np.median(dec, 0), " dim U:", sorted({u.shape[1] for u in U.values()}), " nZ:", sorted({z.shape[1] for z in Z.values()}))
    th = [0.9, 0.5, 0.1, 0.01]
    for name, D in [("U", U), ("Z", Z)]:
        print(f"并（15 点）{name}: σ²>0.9/0.5/0.1/0.01 个数", cnt(union(list(D.values())), th))
    for r in (16, 32, 64):
        print(f"并 U 前 {r}: ", cnt(union([u[:, :r] for u in U.values()]), th))
    # 两两：固定点 (10,1) 与其余点
    a = (10, 1)
    for mu in [(1e4, 1), (3.16e5, 1), (1e6, 1), (1e6, 0.1), (1e6, 10)]:
        su, sz = sines(U[a], U[mu]), sines(Z[a], Z[mu])
        print(f"(10,1) vs {mu}: U 正弦 <0.01/<0.1/>0.5 个数 {int((su<0.01).sum())}/{int((su<0.1).sum())}/{int((su>0.5).sum())} 中位 {su.median():.3f} | "
              f"Z {int((sz<0.01).sum())}/{int((sz<0.1).sum())}/{int((sz>0.5).sum())} 中位 {sz.median():.3f}")
    # float32 与 float64 求值的同一 η：U64 各方向在 span U32 外的分量，按 σ 量级
    mu = (1e6, 1)
    n64 = copy.deepcopy(net).double()
    e = torch.as_tensor(eta(mu), dtype=torch.float64, device=nc.X.device)
    with torch.no_grad():
        U64, s64 = orth(nc.admissible(n64.fields(nc.X.double(), e)))
    U32 = U[mu]
    res = torch.linalg.norm(U64 - U32 @ (U32.T @ U64), dim=0)
    s64 = s64[:U64.shape[1]]
    out = []
    for lo, hi in [(1e-2, 2), (1e-4, 1e-2), (1e-6, 1e-4), (0, 1e-6)]:
        sel = (s64 > lo) & (s64 <= hi)
        out.append(f"{int(sel.sum())}:{res[sel].max():.1e}" if sel.any() else "0")
    print(f"float64 vs float32 {mu}: 各量级 个数:最大偏离", " ".join(out))
