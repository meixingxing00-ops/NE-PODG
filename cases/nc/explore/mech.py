"""机制诊断（只算投影，不闭合），预算 N_B：
1. 网络方向取固定 η（训练区中心 (0, 0)、Ra 最低训练点、Ra 最高训练点）时，V⊕S⊕Z 的最佳逼近误差；与随机网络在全部训练点的并集对照。
2. 余项 r = (I − P)(u − ū) 的空间分布：P 取 V⊕S 与 V⊕S⊕Z_ne，近壁带（到最近壁距离 < 0.02、0.05、0.1）内的速度能量占比。
3. 测试余项族的秩（用测试数据，只作诊断）：中、高 Ra 内插与外推工况在 V⊕S 之外的余项，奇异值与留一捕获率。
结果 explore/results/mech_n<B>.csv（1、2）与终端输出（3）。
用法（仓库根目录，PINNGPU）：python -m cases.nc.explore.mech B"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import NC, eta
from cases.nc.explore.common import Budget, cat, eproj, extend, target, tests
from nepodg import candidates

OUT = Path(__file__).resolve().parent / "results"
budget = int(sys.argv[1])
nc = NC()
bd = Budget(nc, budget)
net = bd.net()
torch.manual_seed(1)
rnd = candidates.Net(**net.cfg).to(nc.X.device).eval()
dirs = lambda m, e: candidates.directions(m, nc.X, e, nc.admissible)
E = np.array(bd.etas)
fixed = {"c": np.zeros(2), "lo": E[E[:, 0].argmin()], "hi": E[E[:, 0].argmax()]}
Zf = {k: extend(bd.VS, dirs(net, e), nc.whiten)[:2] for k, e in fixed.items()}
Zru = extend(bd.VS, torch.cat([dirs(rnd, e) for e in bd.etas], 1), nc.whiten)[:2]

# 面中心到最近壁的距离（白化向量的分量顺序：u 面后接 v 面）
cav = nc.cav
xu, yu = np.meshgrid(cav.x[1:-1], cav.yc, indexing="ij")
xv, yv = np.meshgrid(cav.xc, cav.y[1:-1], indexing="ij")
xf, yf = np.concatenate([xu.ravel(), xv.ravel()]), np.concatenate([yu.ravel(), yv.ravel()])
dwall = torch.as_tensor(np.minimum(np.minimum(xf, 1 - xf), np.minimum(yf, 1 - yf)), device=bd.U.device)
BANDS = [0.02, 0.05, 0.1]
area = [float((cav.Mu * (dwall.cpu().numpy() < b)).sum() / cav.Mu.sum()) for b in BANDS]
print(f"近壁带面积占比 {', '.join(f'{a:.3f}' for a in area)}")


def band(r):
    e = r ** 2
    return [float(e[dwall < b].sum() / e.sum()) for b in BANDS]


cols = ["id", "role", "Ra", "Pr", "VS", "VSfc", "VSflo", "VSfhi", "VSrandU", "nRandU"] + \
       [f"bVS{b}" for b in BANDS] + [f"bNE{b}" for b in BANDS]
R = []
OUT.mkdir(exist_ok=True)
with open(OUT / f"mech_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        r_vs = dw - bd.VS[1] @ (bd.VS[1].T @ dw)
        Zn = extend(bd.VS, dirs(net, eta(mu)), nc.whiten)[1]
        Bn = torch.cat([bd.VS[1], Zn], 1)
        r_ne = dw - Bn @ (Bn.T @ dw)
        wr.writerow([c["id"], c["role"], c["Ra"], c["Pr"], eproj(dw, sc, bd.VS[1])] +
                    [eproj(dw, sc, cat(bd.VS, Zf[k])[1]) for k in fixed] +
                    [eproj(dw, sc, cat(bd.VS, Zru)[1]), Zru[0].shape[1]] + band(r_vs) + band(r_ne))
        if float(c["Ra"]) > 1e3:
            R.append(r_vs / sc)

# 3
R = torch.stack(R, 1)
s = torch.linalg.svdvals(R)
en = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
print(f"测试余项 {R.shape[1]} 个：能量 90/99/99.9% 所需维数 "
      + ", ".join(str(int((en < q).sum()) + 1) for q in (0.9, 0.99, 0.999)))
cap = {k: [] for k in (1, 2, 4, 8, 16)}
for j in range(R.shape[1]):
    P = torch.linalg.svd(torch.cat([R[:, :j], R[:, j + 1:]], 1), full_matrices=False)[0]
    for k in cap:
        cap[k].append(float(torch.linalg.norm(R[:, j] - P[:, :k] @ (P[:, :k].T @ R[:, j])) / torch.linalg.norm(R[:, j])))
print("留一：其余测试余项的前 k 个主方向，留出余项的相对残留（中位数）"
      + ", ".join(f"k={k} {np.median(v):.3f}" for k, v in cap.items()))
