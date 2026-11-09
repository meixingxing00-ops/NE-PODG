"""第一阶段：以 SA 空间 V⊕S 为基础，现有网络方向去 V⊕S 分量后的增量（只算最佳逼近误差，不闭合）。
每个测试工况记录：
  V、V⊕S、V⊕Z_ne（NE-PODG 原空间）
  V⊕S⊕t：只加教师方向 t = Φ(μ) w(μ)
  V⊕S⊕Z_ne^k：网络 128 个原始方向去投影后按白化能量的前 k 个（k = 8、16、32、64、128）
  V⊕S⊕Z_rand^k：未训练网络（种子 1）同样处理
  V⊕S⊕Z_near：最近训练点 η_j 处的网络方向（固定、局部）
  V⊕S⊕Z_union^128 / Z_union：全部训练点处网络方向的并（固定），按能量取前 128 个 / 全部
dist 为查询点到最近训练点的 η 距离。结果 explore/results/stage1_n<B>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.explore.stage1 B"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import NC, eta
from cases.nc.explore.common import Budget, cat, eproj, extend, extend_k, target, tests
from nepodg import candidates, space

OUT = Path(__file__).resolve().parent / "results"
KS = [8, 16, 32, 64, 128]


budget = int(sys.argv[1])
nc = NC()
bd = Budget(nc, budget)
net = bd.net()
torch.manual_seed(1)
rnd = candidates.Net(**net.cfg).to(nc.X.device).eval()
dirs = lambda m, e: candidates.directions(m, nc.X, e, nc.admissible)
E = np.array(bd.etas)
union = torch.cat([dirs(net, e) for e in bd.etas], 1)
Zu = extend(bd.VS, union, nc.whiten)
Zu128 = extend_k(bd.VS, union, nc.whiten, 128)
print(f"n{budget}：n_V {bd.Vb[0].shape[1]}，n_VS {bd.VS[0].shape[1]}，并集方向 {Zu[0].shape[1]}")
OUT.mkdir(exist_ok=True)
cols = ["id", "role", "Ra", "Pr", "dist", "V", "VS", "VZne", "VSt"] + [f"VSne{k}" for k in KS] + \
       [f"VSrand{k}" for k in KS] + ["VSnear", "VSunion128", "VSunion"]
with open(OUT / f"stage1_n{budget}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(cols)
    for c in tests():
        dw, sc, mu = target(nc, bd.pod.mean, c["id"])
        q = eta(mu)
        d = np.linalg.norm(E - q, axis=1)
        e_ = torch.as_tensor(q, dtype=torch.float32, device=nc.X.device)
        Phi = dirs(net, q)
        with torch.no_grad():
            t = nc.admissible((net.fields(nc.X, e_) @ net.readout(e_)).double())
        r = [c["id"], c["role"], c["Ra"], c["Pr"], d.min(),
             eproj(dw, sc, bd.Vb[1]), eproj(dw, sc, bd.VS[1]),
             eproj(dw, sc, cat(bd.Vb, space.complement(Phi, nc.whiten, bd.pod)[:2])[1]),
             eproj(dw, sc, cat(bd.VS, extend(bd.VS, t[:, None], nc.whiten)[:2])[1])]
        r += [eproj(dw, sc, cat(bd.VS, extend_k(bd.VS, Phi, nc.whiten, k))[1]) for k in KS]
        Pr_ = dirs(rnd, q)
        r += [eproj(dw, sc, cat(bd.VS, extend_k(bd.VS, Pr_, nc.whiten, k))[1]) for k in KS]
        r += [eproj(dw, sc, cat(bd.VS, extend(bd.VS, dirs(net, E[d.argmin()]), nc.whiten)[:2])[1]),
              eproj(dw, sc, cat(bd.VS, Zu128)[1]), eproj(dw, sc, cat(bd.VS, Zu[:2])[1])]
        wr.writerow(r)
print(f"→ {OUT / f'stage1_n{budget}.csv'}")
