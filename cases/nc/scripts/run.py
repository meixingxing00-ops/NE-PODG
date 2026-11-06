"""NE-PODG 在测试工况上的求解：快照预算 nB 的完整 POD 与已训练网络（models/n<B>.pt），role 为 test 或 extrap。
每个工况记录 n_V、n_Z、速度误差 e、增广空间与 POD 空间的最佳逼近误差、e_real、温度误差、平均 Nu 误差、收敛与耗时，
结果 results/n<B>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.run B role [前 k 个工况]"""
import csv
import sys
import time

import numpy as np
import torch

from cases.nc.case import NC, ROOT, cases, eta
from nepodg import candidates, closure, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
nc = NC()
train = [nc.load(c["id"]) for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([s[0] for s in train], 1), nc.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
net, _ = candidates.load(ROOT / "models" / f"n{budget}.pt", nc.X.device)
todo = [c for c in cases() if c["role"] == role][:int(sys.argv[3]) if len(sys.argv) > 3 else None]
Mp = nc.cav.Mp
out = ROOT / "results" / f"n{budget}_{role}.csv"
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Ra", "Pr", "nV", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_T", "e_Nu", "converged", "iters", "time"])
    for c in todo:
        psi_r, T_r, mu = nc.load(c["id"])
        t0 = time.perf_counter()
        Phi = candidates.directions(net, nc.X, eta(mu), nc.admissible)
        Z, Zw, _ = space.complement(Phi, nc.whiten, pod)
        B, Bw = torch.cat([pod.V, Z], 1), torch.cat([pod.Vw, Zw], 1)
        y, T, st = closure.solve(nc.reduced, pod.mean, B, mu, np.concatenate([init(eta(mu)), np.zeros(Z.shape[1])]))
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(y, psi_r, pod.mean, Bw, nc.whiten, nc.lift)
        ep_pod = metrics.errors(y, psi_r, pod.mean, pod.Vw, nc.whiten, nc.lift)[1]
        T = T.cpu().numpy()
        eT = np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2))
        nu, nu_r = nc.cav.nusselt(T)[2], nc.cav.nusselt(T_r)[2]
        wr.writerow([c["id"], c["Ra"], c["Pr"], pod.V.shape[1], Z.shape[1], e, ep, er, ep_pod, eT, abs(nu / nu_r - 1),
                     st["converged"], st["iters"], t])
        fh.flush()
print(f"{len(todo)} 个工况 → {out}")
