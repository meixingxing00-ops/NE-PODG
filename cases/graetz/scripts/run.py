"""降阶求解（方法设计 §5）：快照预算 N<B>，role 为 pool（训练池）、interp（内插）或 extrap（外推）。space：
  pod                         V：快照空间
  sapodg                      W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>[_fix<编号>][_z<r>]
                              在基 V 或 VS（= W）上加网络方向（去基的分量）。网络名见 case.kind：
                                v、v<种子>、v…m<m>  训练的网络 models/<网络名>/n<B>.pt
                                r<种子>、r…m<m>     同结构、以该种子随机初始化的未训练网络
                              fix<编号>：方向取自 cases.csv 中该工况的参数，所有查询相同（默认取查询参数）
                              z<r>：候选空间只取 Φ(η) 的前 r 个奇异方向（M 范数）再去基的分量，n_Z ≤ r
  主方法为 VS_v（默认）。例：VS_r1、V_v、VS_v_fixP018、VS_v_z32、VS_vm64、VS_v_fixP018_z32。
闭合为线性 Galerkin 直接求解（case.Graetz.solve）。每个工况记录 n_V、n_S、n_Z、温度误差 e、所用空间与 POD 空间的最佳逼近误差
（按闭合所用的状态基，metrics.basis）、e_real、T_av 误差、局部 Nu 沿受热壁的最大误差、单次查询耗时 time 及其中网络方向与去投影的耗时 time_basis；
有网络方向时另记保留方向的主角正弦 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数。
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.run B role [space]"""
import csv
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.graetz.case import MODELS, ROOT, Graetz, cases, eta, kind
from nepodg import candidates, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "VS_v"
g = Graetz()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([g.load(i)[0] for i in ids], 1), g.whiten)
base, ns = pod, 0
if mode == "sapodg" or mode.startswith("VS"):
    base, ns, _ = space.sensitivity_space(pod, g.sens(ids).reshape(g.n, -1), g.whiten)
net, fix, zmax = None, None, None
if "_" in mode:
    parts = mode.split("_")
    trained, seed, m = kind(parts[1])
    if trained:
        net = candidates.load(MODELS / parts[1] / f"n{budget}.pt", g.X.device)
    else:
        torch.manual_seed(seed)
        net = candidates.Net(m=m).to(g.X.device).eval()
    for p in parts[2:]:
        if p.startswith("fix"):
            fix = next(eta((float(c["L"]), float(c["Pe"]))) for c in cases() if c["id"] == p[3:])
        else:
            zmax = int(p[1:])
Vw_pod = metrics.basis(pod.V, g.whiten)
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "L", "Pe", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_Tav", "e_Nu", "time",
                 "time_basis", "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        T_r, mu = g.load(c["id"])
        t0 = time.perf_counter()
        B, sst = base.V, [""] * 4
        if net is not None:
            Z, _, s = space.complement(candidates.directions(net, g.X, eta(mu) if fix is None else fix, g.admissible),
                                       g.whiten, base, r=zmax)
            B = torch.cat([B, Z], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        torch.cuda.synchronize()
        tb = time.perf_counter() - t0
        T = g.solve(pod.mean, B, mu)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(T, T_r, pod.mean, metrics.basis(B, g.whiten), g.whiten, g.lift)
        ep_pod = metrics.errors(T, T_r, pod.mean, Vw_pod, g.whiten, g.lift)[1]
        wr.writerow([c["id"], c["L"], c["Pe"], pod.V.shape[1], ns, B.shape[1] - base.V.shape[1], e, ep, er, ep_pod,
                     abs(g.tav(T, mu) / g.tav(T_r, mu) - 1), np.abs(g.nusselt(T, mu) / g.nusselt(T_r, mu) - 1).max(), t, tb,
                     *sst])
print(f"→ {out}")
