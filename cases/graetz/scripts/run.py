"""降阶求解（方法设计 §5、§8）：快照预算 N<B>，role 为 pool（训练池）、interp（内插）或 extrap（外推）。space：
  pod           V：快照空间
  sapodg        W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>   在基 V 或 VS（= W）上加查询点的网络方向（去基的分量）：
                d        值与导数监督的网络 models/d/n<B>.pt
                v        只用值监督的网络 models/v/n<B>.pt
                r<种子>  同结构、以该种子随机初始化的未训练网络
  主方法为 VS_d（默认）；VS_v、V_d、V_v、VS_r<种子> 为消融。
闭合为线性 Galerkin 直接求解（case.Graetz.solve）。每个工况记录 n_V、n_S、n_Z、温度误差 e、所用空间与 POD 空间的最佳逼近误差、
e_real、T_av 误差、局部 Nu 沿受热壁的最大误差与耗时；有网络方向时另记其主角正弦谱 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数。
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.run B role [space]"""
import csv
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.graetz.case import MODELS, ROOT, Graetz, cases, eta
from nepodg import candidates, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "VS_d"
g = Graetz()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([g.load(i)[0] for i in ids], 1), g.whiten)
base, ns = pod, 0
if mode == "sapodg" or mode.startswith("VS"):
    base, ns, _ = space.sensitivity_space(pod, g.sens(ids).reshape(g.n, -1), g.whiten)
net = None
if "_" in mode:
    kind = mode.split("_")[1]
    if kind[0] == "r":
        torch.manual_seed(int(kind[1:]))
        net = candidates.Net().to(g.X.device).eval()
    else:
        net = candidates.load(MODELS / kind / f"n{budget}.pt", g.X.device)
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "L", "Pe", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_Tav", "e_Nu", "time",
                 "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        T_r, mu = g.load(c["id"])
        t0 = time.perf_counter()
        B, Bw, sst = base.V, base.Vw, [""] * 4
        if net is not None:
            Z, Zw, s = space.complement(candidates.directions(net, g.X, eta(mu), g.admissible), g.whiten, base)
            B, Bw = torch.cat([B, Z], 1), torch.cat([Bw, Zw], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        T = g.solve(pod.mean, B, mu)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(T, T_r, pod.mean, Bw, g.whiten, g.lift)
        ep_pod = metrics.errors(T, T_r, pod.mean, pod.Vw, g.whiten, g.lift)[1]
        wr.writerow([c["id"], c["L"], c["Pe"], pod.V.shape[1], ns, B.shape[1] - base.V.shape[1], e, ep, er, ep_pod,
                     abs(g.tav(T, mu) / g.tav(T_r, mu) - 1), np.abs(g.nusselt(T, mu) / g.nusselt(T_r, mu) - 1).max(), t, *sst])
print(f"→ {out}")
