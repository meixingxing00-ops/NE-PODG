"""导数监督网络的最佳逼近诊断（不闭合）。对基准网络 a 与导数监督网络 b，每个测试工况记录：
  V、VS（V⊕S）
  Va、VSa、Vb、VSb：V 或 V⊕S 上加查询点 128 个网络方向（去基础空间分量）
  Vb_d：V 上加 b 的查询点方向及其对 η 的导数方向（128 × 3）
  Vb_h：V 上加 b 在查询点的预测 f 与 ∂f/∂η（3 个方向）
  Vb_S：V 上加 b 在各训练点的 ∂f/∂η（2 N_B 个，网络版的 S）
  preda、predb：网络直接输出误差
模型名 main 指主线只用值监督的网络 cases/nc/models/v/n<B>.pt，其余为 explore/models/<名>.pt。结果 explore/results/sobeval_n<B>.csv。
用法：python -m cases.nc.explore.sobeval B a b"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacfwd

from cases.nc.case import NC, eta
from cases.nc.explore.common import Budget, cat, eproj, extend, target, tests
from nepodg import candidates

HERE = Path(__file__).resolve().parent
budget, na, nb = int(sys.argv[1]), sys.argv[2], sys.argv[3]
nc = NC()
bd = Budget(nc, budget)
load = lambda m: bd.net() if m == "main" else candidates.load(HERE / "models" / f"{m}.pt", nc.X.device)[0]
A, Bn = load(na), load(nb)
dev = nc.X.device


@torch.no_grad()
def fields_and_grads(net, q):
    """查询点 q：方向 Φ (n, m)、∂Φ/∂η (n, 2m)、预测 f (n,)、∂f/∂η (n, 2)。"""
    e = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=dev)
    F = lambda x: net.fields(nc.X, x)
    g = lambda x: F(x) @ net.readout(x)
    dPhi = jacfwd(F)(e)                                                           # (P, m, 2)
    df = jacfwd(g)(e)                                                             # (P, 2)
    adm = lambda Y: nc.admissible(Y.double())
    return adm(F(e)), adm(dPhi.reshape(len(nc.X), -1)), adm(g(e)), adm(df)


S_net = torch.cat([fields_and_grads(Bn, q)[3] for q in bd.etas], 1)
aug = lambda base, X: cat(base, extend(base, X, nc.whiten)[:2])[1]
cols = ["id", "role", "Ra", "Pr", "V", "VS", "Va", "VSa", "Vb", "VSb", "Vb_d", "Vb_h", "Vb_S", "preda", "predb"]
with open(HERE / "results" / f"sobeval_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        q = eta(mu)
        Pa, _, fa, _ = fields_and_grads(A, q)
        Pb, dPb, fb, dfb = fields_and_grads(Bn, q)
        r = [c["id"], c["role"], c["Ra"], c["Pr"], eproj(dw, sc, bd.Vb[1]), eproj(dw, sc, bd.VS[1]),
             eproj(dw, sc, aug(bd.Vb, Pa)), eproj(dw, sc, aug(bd.VS, Pa)),
             eproj(dw, sc, aug(bd.Vb, Pb)), eproj(dw, sc, aug(bd.VS, Pb)),
             eproj(dw, sc, aug(bd.Vb, torch.cat([Pb, dPb], 1))),
             eproj(dw, sc, aug(bd.Vb, torch.cat([fb[:, None], dfb], 1))),
             eproj(dw, sc, aug(bd.Vb, S_net)),
             float(torch.linalg.norm(nc.whiten(fa) - dw) / sc), float(torch.linalg.norm(nc.whiten(fb) - dw) / sc)]
        wr.writerow(r)
        fh.flush()
print(f"→ {HERE / 'results' / f'sobeval_n{budget}.csv'}")
