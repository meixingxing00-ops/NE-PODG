"""通用光滑基对照（只算投影）：V⊕S 上加与网络同样经边界因子 𝓑 的多项式场 𝓑·P_a(x)P_b(y)（Legendre，坐标缩放到 [−1, 1]），
  poly_t：总次数 a + b ≤ 14（120 个）；poly_q：a, b ≤ 10（121 个）；poly_t2：总次数 ≤ 20（231 个）。
另记网络主干最后隐层（128 个 tanh 特征，Ra 最高训练点处）经 𝓑 后作为方向的结果 hid。
基础空间 base 为 VS（默认）或 V（只用 POD）。结果 explore/results/generic_<base>_n<B>.csv。
用法：python -m cases.nc.explore.generic B [base]"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial import legendre

from cases.nc.explore.common import Budget, cat, eproj, extend, target, tests
from cases.nc.case import NC

OUT = Path(__file__).resolve().parent / "results"
budget = int(sys.argv[1])
nc = NC()
bd = Budget(nc, budget)
base = sys.argv[2] if len(sys.argv) > 2 else "VS"
BS = bd.VS if base == "VS" else bd.Vb
X = nc.X.double().cpu().numpy()
L = [legendre.legval(X[:, 0], np.eye(21)[a]) for a in range(21)], [legendre.legval(X[:, 1], np.eye(21)[b]) for b in range(21)]
poly = lambda pairs: nc.admissible(torch.as_tensor(np.stack([L[0][a] * L[1][b] for a, b in pairs], 1), device=bd.U.device))
D = {"poly_t": [(a, b) for a in range(15) for b in range(15) if a + b <= 14],
     "poly_q": [(a, b) for a in range(11) for b in range(11)],
     "poly_t2": [(a, b) for a in range(21) for b in range(21) if a + b <= 20]}
Z = {k: extend(BS, poly(p), nc.whiten)[:2] for k, p in D.items()}
net = bd.net()
E = np.array(bd.etas)
e = torch.as_tensor(E[E[:, 0].argmax()], dtype=torch.float32, device=nc.X.device)
with torch.no_grad():
    H = net.trunk[:-2](torch.cat([nc.X, e.expand(len(nc.X), -1)], 1)).double()
Z["hid"] = extend(BS, nc.admissible(H), nc.whiten)[:2]
print(", ".join(f"{k} {v[0].shape[1]}" for k, v in Z.items()))
OUT.mkdir(exist_ok=True)
with open(OUT / f"generic_{base}_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "role", "Ra", "Pr", base] + list(Z))
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        wr.writerow([c["id"], c["role"], c["Ra"], c["Pr"], eproj(dw, sc, BS[1])] + [eproj(dw, sc, cat(BS, z)[1]) for z in Z.values()])
