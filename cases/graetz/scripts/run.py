"""降阶求解：快照预算 nB 的完整 POD，role 为 pool、interp 或 extrap。space：
  pod     只用 POD 空间
  sapodg  SA-PODG，候选方向为预算内训练点的 ∂T/∂η₁、∂T/∂η₂，与查询参数无关
闭合为线性 Galerkin 直接求解（case.Graetz.solve）。每个工况记录 n_V、n_Z、温度误差 e、所用空间与 POD 空间的最佳逼近误差、
e_real、T_av 相对误差与耗时，有候选方向时另记主角正弦谱 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数，
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录）：python -m cases.graetz.scripts.run B role [space]"""
import csv
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.graetz.case import ROOT, Graetz, cases
from nepodg import metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "pod"
g = Graetz()
train = [g.load(c["id"]) for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([s[0] for s in train], 1), g.whiten)
if mode == "pod":
    B, Bw, nz, sst = pod.V, pod.Vw, 0, [""] * 4
else:
    Z, Zw, s = space.complement(g.sens_directions(budget), g.whiten, pod)
    B, Bw, nz = torch.cat([pod.V, Z], 1), torch.cat([pod.Vw, Zw], 1), Z.shape[1]
    s = s[:nz].cpu().numpy()
    sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "L", "Pe", "nV", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_Tav", "time",
                 "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        T_r, mu = g.load(c["id"])
        t0 = time.perf_counter()
        T = g.solve(pod.mean, B, mu)
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(T, T_r, pod.mean, Bw, g.whiten, g.lift)
        ep_pod = metrics.errors(T, T_r, pod.mean, pod.Vw, g.whiten, g.lift)[1]
        wr.writerow([c["id"], c["L"], c["Pe"], pod.V.shape[1], nz, e, ep, er, ep_pod, abs(g.tav(T, mu) / g.tav(T_r, mu) - 1), t, *sst])
print(f"→ {out}")
