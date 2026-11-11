"""降阶求解（方法设计 §5）：快照预算 N<B>，role 为 pool（训练池）、interp（内插）或 extrap（外推）。space：
  pod                         V：快照空间
  sapodg                      W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>[_z<r>]          在基 V 或 VS（= W）上加网络方向（去基的分量），网络名见 case.kind；z<r>：只取 Φ(η) 的前 r 个奇异方向
闭合为线性 Galerkin 直接求解（case.Conjugate.solve）。每个工况记录 n_V、n_S、n_Z、主场误差 e（温度增量 u 的参考域 L²）、
所用空间与 POD 空间的最佳逼近误差、e_real、温度场误差 e_T、界面总热流 Q、平均界面温度 Tw、出口混合温度 Tout 的相对误差、
四个位置局部 Nu 的最大相对误差、单次查询耗时 time 及其中网络方向与去投影的耗时 time_basis，以及保留方向主角正弦 s 的统计。
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.conjugate.scripts.run B role [space]"""
import csv
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.conjugate.case import MODELS, ROOT, Conjugate, cases, eta, kind
from nepodg import candidates, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "VS_v"
g = Conjugate()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
pod = space.POD(torch.stack([g.load(i)[0] for i in ids], 1), g.whiten)
base, ns = pod, 0
if mode == "sapodg" or mode.startswith("VS"):
    base, ns, _ = space.sensitivity_space(pod, g.sens(ids).reshape(g.n, -1), g.whiten)
net, zmax = None, None
if "_" in mode:
    parts = mode.split("_")
    trained, seed, m = kind(parts[1])
    if trained:
        net = candidates.load(MODELS / parts[1] / f"n{budget}.pt", g.X.device)
    else:
        torch.manual_seed(seed)
        net = candidates.Net(m=m).to(g.X.device).eval()
    for p in parts[2:]:
        zmax = int(p[1:])
Vw_pod = metrics.basis(pod.V, g.whiten)
Mh = g.ch.Mh
rel = lambda a, b: abs(a / b - 1)
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Pe", "ratio", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_T", "e_Q", "e_Tw", "e_Tout",
                 "e_Nu", "time", "time_basis", "s_min", "s_med", "n_s01", "n_s001"])
    for c in [c for c in cases() if c["role"] == role]:
        u_r, mu = g.load(c["id"])
        t0 = time.perf_counter()
        B, sst = base.V, [""] * 4
        if net is not None:
            Z, _, s = space.complement(candidates.directions(net, g.X, eta(mu), g.admissible), g.whiten, base, r=zmax)
            B = torch.cat([B, Z], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        torch.cuda.synchronize()
        tb = time.perf_counter() - t0
        u = g.solve(pod.mean, B, mu)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(u, u_r, pod.mean, metrics.basis(B, g.whiten), g.whiten, g.lift)
        ep_pod = metrics.errors(u, u_r, pod.mean, Vw_pod, g.whiten, g.lift)[1]
        T, T_r = g.temperature(u), g.temperature(u_r)
        o, o_r = g.outputs(u, mu), g.outputs(u_r, mu)
        wr.writerow([c["id"], c["Pe"], c["ratio"], pod.V.shape[1], ns, B.shape[1] - base.V.shape[1], e, ep, er, ep_pod,
                     np.sqrt(Mh @ (T - T_r) ** 2 / (Mh @ T_r ** 2)), rel(o["Q"], o_r["Q"]), rel(o["Tw"], o_r["Tw"]),
                     rel(o["Tout"], o_r["Tout"]), np.abs(o["Nu"] / o_r["Nu"] - 1).max(), t, tb, *sst])
print(f"→ {out}")
