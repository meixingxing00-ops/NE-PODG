"""工况表 cases/graetz/cases.csv（实例化设计 §3–4）。参数取 3 位有效数字。
- pool：训练池 35 点，log10 L = 0, 0.25, …, 1，log10 Pe = −1, −0.5, …, 2；n4 标记快照预算 N4（四角 L ∈ {1, 10} × Pe ∈ {0.1, 100}）。
- interp：训练区内 100 点，(log10 L, log10 Pe) ∈ [0, 1] × [−1, 2] 上的拉丁超立方取样；与任一训练池点在两个对数坐标上
  都相差不到 0.05 的取样作废，按种子从 0 起依次试，取第一个全部满足的种子。
- extrap：外推区 27 点，log10 Pe = 2 + 0.125、0.25、0.5（133、178、316），log10 L = 0, 0.125, …, 1。
用法（仓库根目录）：python -m cases.graetz.fom.make_cases"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

r3 = lambda v: float(f"{v:.3g}")
lL, lPe = np.arange(0, 1.01, 0.25), np.arange(-1, 2.01, 0.5)
B = ["n4"]
rows = []
for j, b in enumerate(lPe):
    for i, a in enumerate(lL):
        rows.append(dict(role="pool", L=r3(10 ** a), Pe=r3(10 ** b), n4=int(i in (0, 4) and j in (0, 6))))

grid = np.array([(a, b) for a in lL for b in lPe])
for seed in range(100000):
    z = qmc.LatinHypercube(d=2, seed=seed).random(100) * [1.0, 3.0] + [0.0, -1.0]
    z = np.log10([[r3(10 ** a), r3(10 ** b)] for a, b in z])
    if np.abs(z[:, None, :] - grid[None]).max(2).min() >= 0.05:
        break
for a, b in z[np.lexsort((z[:, 1], z[:, 0]))]:
    rows.append(dict(role="interp", L=r3(10 ** a), Pe=r3(10 ** b), **dict.fromkeys(B, 0)))

for d in (0.125, 0.25, 0.5):
    for a in np.arange(0, 1.01, 0.125):
        rows.append(dict(role="extrap", L=r3(10 ** a), Pe=r3(10 ** (2 + d)), **dict.fromkeys(B, 0)))

prefix, count = {"pool": "P", "interp": "I", "extrap": "E"}, {}
for r in rows:
    count[r["role"]] = count.get(r["role"], 0) + 1
    r["id"] = f"{prefix[r['role']]}{count[r['role']]:03d}"
    r["L"], r["Pe"] = f"{r['L']:g}", f"{r['Pe']:g}"
out = Path(__file__).resolve().parents[1] / "cases.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, ["id", "role", "L", "Pe"] + B)
    wr.writeheader()
    wr.writerows(rows)
print(f"LHS seed {seed}; {count}; " + ", ".join(f"{b} {sum(int(r[b]) for r in rows)}" for b in B))
