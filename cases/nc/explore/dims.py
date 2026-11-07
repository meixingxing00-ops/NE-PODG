"""维数–精度曲线（只算投影）：基础空间 V 或 V⊕S 上加总次数 ≤ d 的多项式场（d = 2、4、…、20），
与网络方向按白化能量取前 k 个（k = 4、8、16、32、64、128）。结果 explore/results/dims_<base>_n<B>.csv。
用法：python -m cases.nc.explore.dims B base"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import NC, eta
from cases.nc.explore.common import Budget, cat, eproj, extend, extend_k, poly, target, tests
from nepodg import candidates

OUT = Path(__file__).resolve().parent / "results"
budget, base = int(sys.argv[1]), sys.argv[2]
nc = NC()
bd = Budget(nc, budget)
BS = bd.VS if base == "VS" else bd.Vb
DEG, KS = list(range(2, 21, 2)), [4, 8, 16, 32, 64, 128]
P = {d: extend(BS, poly(nc, d), nc.whiten)[:2] for d in DEG}
net = bd.net()
with open(OUT / f"dims_{base}_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "role", "Ra", "Pr", base] + [f"p{d}" for d in DEG] + [f"ne{k}" for k in KS])
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        Phi = candidates.directions(net, nc.X, eta(mu), nc.admissible)
        wr.writerow([c["id"], c["role"], c["Ra"], c["Pr"], eproj(dw, sc, BS[1])] +
                    [eproj(dw, sc, cat(BS, P[d])[1]) for d in DEG] +
                    [eproj(dw, sc, cat(BS, extend_k(BS, Phi, nc.whiten, k))[1]) for k in KS])
print(", ".join(f"d{d}:{P[d][0].shape[1]}" for d in DEG))
