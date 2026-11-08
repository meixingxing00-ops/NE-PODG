"""网络直接输出 T̂ = T̄ + Φ w 的误差：预算 N<B> 的网络（kind 为 d 或 v，同 train.py）在 pool、interp 或 extrap 工况上，
温度误差 e（参考域 L²）、T_av 误差与局部 Nu 沿受热壁的最大误差。结果 results/n<B>/teacher_<kind>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.teacher B role [kind]"""
import csv
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

from cases.graetz.case import MODELS, ROOT, Graetz, cases, eta
from nepodg import candidates

budget, role = int(sys.argv[1]), sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "d"
g = Graetz()
net, mean = candidates.load(MODELS / kind / f"n{budget}.pt", g.X.device)
out = ROOT / "results" / f"n{budget}" / f"teacher_{kind}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "L", "Pe", "e", "e_Tav", "e_Nu"])
    for c in [c for c in cases() if c["role"] == role]:
        T_r, mu = g.load(c["id"])
        e_ = torch.as_tensor(eta(mu), dtype=torch.float32, device=g.X.device)
        with torch.no_grad():
            T = mean + g.admissible((net.fields(g.X, e_) @ net.readout(e_)).double())
        e = float(torch.linalg.norm(g.whiten(T - T_r)) / torch.linalg.norm(g.whiten(T_r)))
        wr.writerow([c["id"], c["L"], c["Pe"], e, abs(g.tav(T, mu) / g.tav(T_r, mu) - 1),
                     np.abs(g.nusselt(T, mu) / g.nusselt(T_r, mu) - 1).max()])
print(f"→ {out}")
