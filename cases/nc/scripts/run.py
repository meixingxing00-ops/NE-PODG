"""降阶求解（方法设计 §5）：快照预算 N<B>，role 为 interp（内插）或 extrap（外推）。space：
  pod           V：快照空间
  sapodg        W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>   在基 V 或 VS（= W）上加查询点的网络方向（去基的分量）：
                v        训练的网络 models/v/n<B>.pt（种子 1234）
                v<种子>  以该种子训练的网络 models/v<种子>/n<B>.pt
                r<种子>  同结构、以该种子随机初始化的未训练网络
  主方法为 VS_v（默认）。
温度都在全网格上求解。每个工况记录 n_V、n_S、n_Z、速度误差 e、所用空间与 POD 空间的最佳逼近误差、e_real、
温度误差、平均 Nu 误差、收敛与耗时；有网络方向时另记其主角正弦谱 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数。
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.run B role [space]"""
import csv
import sys
import time

import numpy as np
import torch

from cases.nc.case import MODELS, NC, ROOT, cases, eta
from nepodg import candidates, closure, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "VS_v"
nc = NC()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
train = [nc.load(i) for i in ids]
pod = space.POD(torch.stack([s[0] for s in train], 1), nc.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
base, ns = pod, 0
if mode == "sapodg" or mode.startswith("VS"):
    base, ns, _ = space.sensitivity_space(pod, nc.sens(ids).reshape(nc.n, -1), nc.whiten)
net = None
if "_" in mode:
    kind = mode.split("_")[1]
    if kind[0] == "r":
        torch.manual_seed(int(kind[1:]))
        net = candidates.Net().to(nc.X.device).eval()
    else:
        net = candidates.load(MODELS / kind / f"n{budget}.pt", nc.X.device)
Mp = nc.cav.Mp
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Ra", "Pr", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_T", "e_Nu", "converged", "iters",
                 "time", "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        psi_r, T_r, mu = nc.load(c["id"])
        t0 = time.perf_counter()
        B, Bw, sst = base.V, base.Vw, [""] * 4
        if net is not None:
            Z, Zw, s = space.complement(candidates.directions(net, nc.X, eta(mu), nc.admissible), nc.whiten, base)
            B, Bw = torch.cat([B, Z], 1), torch.cat([Bw, Zw], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        nz = B.shape[1] - base.V.shape[1]
        c0 = np.concatenate([init(eta(mu)), np.zeros(B.shape[1] - pod.V.shape[1])])
        y, T, st = closure.solve(nc.reduced, pod.mean, B, mu, c0)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(y, psi_r, pod.mean, Bw, nc.whiten, nc.lift)
        ep_pod = metrics.errors(y, psi_r, pod.mean, pod.Vw, nc.whiten, nc.lift)[1]
        T = T.cpu().numpy()
        eT = np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2))
        nu, nu_r = nc.cav.nusselt(T)[2], nc.cav.nusselt(T_r)[2]
        wr.writerow([c["id"], c["Ra"], c["Pr"], pod.V.shape[1], ns, nz, e, ep, er, ep_pod, eT, abs(nu / nu_r - 1),
                     st["converged"], st["iters"], t, *sst])
        fh.flush()
print(f"→ {out}")
