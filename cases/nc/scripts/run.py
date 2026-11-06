"""降阶求解：快照预算 nB 的完整 POD，role 为 interp（内插）或 extrap（外推）。space：
  ne（默认）  NE-PODG，已训练网络 models/n<B>.pt 给出候选方向
  pod         只用 POD 空间
  rand<种子>  候选方向由同结构、以该种子随机初始化、未训练的网络给出，其余与 NE-PODG 相同
  sapodg      SA-PODG，候选方向为训练快照处的稳态灵敏度（cases/nc/sapodg），与查询参数无关
各方式闭合相同，温度都在全网格上求解。每个工况记录 n_V、n_Z、速度误差 e、所用空间与 POD 空间的最佳逼近误差、e_real、
温度误差、平均 Nu 误差、收敛与耗时，有候选方向时另记主角正弦谱 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数，
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.run B role [space]"""
import csv
import sys
import time

import numpy as np
import torch

from cases.nc.case import NC, ROOT, cases, eta
from cases.nc.sapodg.directions import directions as sapodg_directions
from nepodg import candidates, closure, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "ne"
nc = NC()
train = [nc.load(c["id"]) for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([s[0] for s in train], 1), nc.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
if mode == "pod":
    augment = None
elif mode == "sapodg":
    fixed = space.complement(sapodg_directions(budget), nc.whiten, pod)
    augment = lambda mu: fixed
else:
    net = candidates.load(ROOT / "models" / f"n{budget}.pt", nc.X.device)[0]
    if mode.startswith("rand"):
        torch.manual_seed(int(mode[4:]))
        net = candidates.Net(**net.cfg).to(nc.X.device).eval()
    augment = lambda mu: space.complement(candidates.directions(net, nc.X, eta(mu), nc.admissible), nc.whiten, pod)
Mp = nc.cav.Mp
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Ra", "Pr", "nV", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_T", "e_Nu", "converged", "iters", "time",
                 "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        psi_r, T_r, mu = nc.load(c["id"])
        t0 = time.perf_counter()
        sst = [""] * 4
        if augment is not None:
            Z, Zw, s = augment(mu)
            B, Bw = torch.cat([pod.V, Z], 1), torch.cat([pod.Vw, Zw], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        else:
            B, Bw = pod.V, pod.Vw
        nz = B.shape[1] - pod.V.shape[1]
        y, T, st = closure.solve(nc.reduced, pod.mean, B, mu, np.concatenate([init(eta(mu)), np.zeros(nz)]))
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(y, psi_r, pod.mean, Bw, nc.whiten, nc.lift)
        ep_pod = metrics.errors(y, psi_r, pod.mean, pod.Vw, nc.whiten, nc.lift)[1]
        T = T.cpu().numpy()
        eT = np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2))
        nu, nu_r = nc.cav.nusselt(T)[2], nc.cav.nusselt(T_r)[2]
        wr.writerow([c["id"], c["Ra"], c["Pr"], pod.V.shape[1], nz, e, ep, er, ep_pod, eT, abs(nu / nu_r - 1),
                     st["converged"], st["iters"], t, *sst])
        fh.flush()
print(f"→ {out}")
