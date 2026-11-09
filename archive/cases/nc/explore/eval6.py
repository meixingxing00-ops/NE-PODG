"""N6 的最佳逼近诊断：V⊕S 上加各网络方向、教师方向、多项式，以及网络预测本身的误差。
每个模型 <名>：VS<名>（查询点 128 个方向去 V⊕S 分量）、t<名>（只加教师方向 û − ū）、pred<名>（‖û − u‖_M / ‖u‖_M）。
结果 explore/results/eval6.csv。用法：python -m cases.nc.explore.eval6 n6sup n6phys ..."""
import csv
import sys
from pathlib import Path

import torch

from cases.nc.case import NC, eta
from cases.nc.explore.common import Budget, cat, eproj, extend, poly, target, tests
from nepodg import candidates

HERE = Path(__file__).resolve().parent
names = sys.argv[1:]
nc = NC()
bd = Budget(nc, 6)
nets = {m: candidates.load(HERE / "models" / f"{m}.pt", nc.X.device) for m in names}
P = {d: cat(bd.VS, extend(bd.VS, poly(nc, d), nc.whiten)[:2])[1] for d in (6, 10, 14)}
cols = ["id", "role", "Ra", "Pr", "V", "VS", "VSp6", "VSp10", "VSp14"] + [f"{k}{m}" for m in names for k in ("VS", "t", "pred")]
print(f"n_V {bd.Vb[0].shape[1]}，n_VS {bd.VS[0].shape[1]}")
with open(HERE / "results" / "eval6.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        r = [c["id"], c["role"], c["Ra"], c["Pr"], eproj(dw, sc, bd.Vb[1]), eproj(dw, sc, bd.VS[1])] + \
            [eproj(dw, sc, P[d]) for d in (6, 10, 14)]
        q = eta(mu)
        e_ = torch.as_tensor(q, dtype=torch.float32, device=nc.X.device)
        for m, net in nets.items():
            Phi = candidates.directions(net, nc.X, q, nc.admissible)
            with torch.no_grad():
                t = nc.admissible((net.fields(nc.X, e_) @ net.readout(e_)).double())
            r += [eproj(dw, sc, cat(bd.VS, extend(bd.VS, Phi, nc.whiten)[:2])[1]),
                  eproj(dw, sc, cat(bd.VS, extend(bd.VS, t[:, None], nc.whiten)[:2])[1]),
                  float(torch.linalg.norm(nc.whiten(t) - dw) / sc)]
        wr.writerow(r)
print(f"→ {HERE / 'results' / 'eval6.csv'}")
