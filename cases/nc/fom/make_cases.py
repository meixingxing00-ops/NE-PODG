"""工况表 cases/nc/cases.csv（实例化设计 §2–3）。参数取 3 位有效数字。
- pool：训练池 50 点，log10 Ra = 1, 1.5, …, 5.5，log10 Pr = −1, −0.5, …, 1；n12、n20、n30、n50 标记所属快照预算。
- test：训练区内 100 点，(log10 Ra, log10 Pr) ∈ [1, 5.5] × [−1, 1] 上的拉丁超立方取样；与任一训练池点在两个对数坐标上
  都相差不到 0.05 的取样作废，按种子从 0 起依次试，取第一个全部满足的种子。
- extrap：外推区 27 点，log10 Ra = 5.5 + 0.125、0.25、0.5（4.22×10⁵、5.62×10⁵、1.00×10⁶），log10 Pr = −1, −0.75, …, 1。
用法（仓库根目录）：python -m cases.nc.fom.make_cases"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

r3 = lambda v: float(f"{v:.3g}")
lra, lpr = np.arange(1, 5.51, 0.5), np.arange(-1, 1.01, 0.5)
rows = []
for j, b in enumerate(lpr):
    for i, a in enumerate(lra):
        sr, sp = i % 3 == 0, j % 2 == 0
        rows.append(dict(role="pool", Ra=r3(10 ** a), Pr=r3(10 ** b), n12=int(sr and sp), n20=int(sr), n30=int(sp), n50=1))

grid = np.array([(a, b) for a in lra for b in lpr])
for seed in range(100000):
    z = qmc.LatinHypercube(d=2, seed=seed).random(100) * [4.5, 2.0] + [1.0, -1.0]
    z = np.log10([[r3(10 ** a), r3(10 ** b)] for a, b in z])
    if np.abs(z[:, None, :] - grid[None]).max(2).min() >= 0.05:
        break
for a, b in z[np.lexsort((z[:, 1], z[:, 0]))]:
    rows.append(dict(role="test", Ra=r3(10 ** a), Pr=r3(10 ** b), n12=0, n20=0, n30=0, n50=0))

for d in (0.125, 0.25, 0.5):
    for b in np.arange(-1, 1.01, 0.25):
        rows.append(dict(role="extrap", Ra=r3(10 ** (5.5 + d)), Pr=r3(10 ** b), n12=0, n20=0, n30=0, n50=0))

prefix, count = {"pool": "P", "test": "T", "extrap": "E"}, {}
for r in rows:
    count[r["role"]] = count.get(r["role"], 0) + 1
    r["id"] = f"{prefix[r['role']]}{count[r['role']]:03d}"
    r["Ra"], r["Pr"] = f"{r['Ra']:g}", f"{r['Pr']:g}"
out = Path(__file__).resolve().parents[1] / "cases.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, ["id", "role", "Ra", "Pr", "n12", "n20", "n30", "n50"])
    wr.writeheader()
    wr.writerows(rows)
print(f"LHS seed {seed}; {count}")
