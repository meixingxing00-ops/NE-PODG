"""降阶求解（方法设计 §5）：快照预算 N<B>，role 为 interp（内插）或 extrap（外推）。space：
  pod                         V：快照空间
  sapodg                      W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>[_z<r>]          在基 V 或 VS（= W）上加网络方向（去基的分量），网络名见 case.kind；主方法 VS_v
闭合用 LM（nepodg.closure），初值为训练点 POD 系数的薄板样条插值；降阶解求出后在全网格上解一次温度。每个工况记录 n_V、n_S、n_Z、
速度误差 e（参考域速度 L²，按容许增量归一）、所用空间与 POD 空间的最佳逼近误差、e_real、温度误差 e_T、主再附着长度 x_r、
受热壁最大 Nu 与平均 Nu 的相对误差、x/S = 1、2、5、10、20 处局部 Nu 的最大相对误差、收敛、迭代数与耗时。
结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.bfs.scripts.run B role [space]"""
import csv
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.bfs.case import BFS, MODELS, ROOT, cases, eta, kind
from nepodg import candidates, closure, metrics, space

budget, role = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "VS_v"
g = BFS()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
train = [g.load(i) for i in ids]
pod = space.POD(torch.stack([s[0] for s in train], 1), g.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
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
Mp, Vw_pod = g.m.Mphat, metrics.basis(pod.V, g.whiten)
NUX = ["Nu_1S", "Nu_2S", "Nu_5S", "Nu_10S", "Nu_20S"]
rel = lambda a, b: abs(a / b - 1)
out = ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Re", "ER", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "e_T", "e_xr", "e_Numax", "e_Nuavg",
                 "e_Nu", "converged", "iters", "time", "time_basis"])
    for c in [c for c in cases() if c["role"] == role]:
        q_r, T_r, mu = g.load(c["id"])
        t0 = time.perf_counter()
        B = base.V
        if net is not None:
            Z = space.complement(candidates.directions(net, g.X, eta(mu), g.admissible), g.whiten, base, r=zmax)[0]
            B = torch.cat([B, Z], 1)
        torch.cuda.synchronize()
        tb = time.perf_counter() - t0
        c0 = np.concatenate([init(eta(mu)), np.zeros(B.shape[1] - pod.V.shape[1])])
        q, _, st = closure.solve(g.reduced, pod.mean, B, mu, c0)
        T = g.temperature(q, mu)
        t = time.perf_counter() - t0
        e, ep, er = metrics.errors(q, q_r, pod.mean, metrics.basis(B, g.whiten), g.whiten, g.lift)
        ep_pod = metrics.errors(q, q_r, pod.mean, Vw_pod, g.whiten, g.lift)[1]
        o, o_r = g.outputs(q, T, mu), g.outputs(q_r, T_r, mu)
        wr.writerow([c["id"], c["Re"], c["ER"], pod.V.shape[1], ns, B.shape[1] - base.V.shape[1], e, ep, er, ep_pod,
                     np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2)), rel(o["xr_S"], o_r["xr_S"]),
                     rel(o["Nu_max"], o_r["Nu_max"]), rel(o["Nu_avg"], o_r["Nu_avg"]),
                     max(rel(o[k], o_r[k]) for k in NUX), st["converged"], st["iters"], t, tb])
        fh.flush()
print(f"→ {out}")
