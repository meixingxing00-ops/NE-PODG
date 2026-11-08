"""网络直接输出 û = ū + Φ w 的误差：预算 N<B> 的网络（kind 为 d 或 v，同 train.py）在 interp（内插）或 extrap（外推）工况上，
速度误差 e（M 范数）；另以 û 的速度在全网格上解能量方程得温度，记温度误差与平均 Nu 误差。结果 results/n<B>/teacher_<kind>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.teacher B role [kind]"""
import csv
import sys

import numpy as np
import torch

from cases.nc.case import MODELS, NC, ROOT, cases, eta, nu_kap
from nepodg import candidates

budget, role = int(sys.argv[1]), sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "d"
nc = NC()
net, mean = candidates.load(MODELS / kind / f"n{budget}.pt", nc.X.device)
Mp = nc.cav.Mp
out = ROOT / "results" / f"n{budget}" / f"teacher_{kind}_{role}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "Ra", "Pr", "e", "e_T", "e_Nu"])
    for c in [c for c in cases() if c["role"] == role]:
        psi_r, T_r, mu = nc.load(c["id"])
        e_ = torch.as_tensor(eta(mu), dtype=torch.float32, device=nc.X.device)
        with torch.no_grad():
            y = mean + nc.admissible((net.fields(nc.X, e_) @ net.readout(e_)).double())
        e = float(torch.linalg.norm(nc.whiten(y - psi_r)) / torch.linalg.norm(nc.whiten(psi_r)))
        T = nc.op.temperature(y, nu_kap(mu)[1]).cpu().numpy()
        eT = np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2))
        nu, nu_r = nc.cav.nusselt(T)[2], nc.cav.nusselt(T_r)[2]
        wr.writerow([c["id"], c["Ra"], c["Pr"], e, eT, abs(nu / nu_r - 1)])
print(f"→ {out}")
