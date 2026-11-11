"""后台阶工况表 cases/bfs/cases.csv（实例化设计 §2–4）；不求解。参数取 3 位有效数字。
- pool：训练池 7×5，log10 Re 在 [2, log10 400] 上等分 7 档，ER = 1.5, 1.625, …, 2；n4/n9/n15/n21/n35 标记预算。
- interp：训练区 (log10 Re, ER) 上的 100 点拉丁超立方；在两个单位化坐标上都与某一训练池点相差不到 0.02 的取样作废，
  按种子从 0 起取第一个全部满足的种子。
- extrap：Re = 450、520、600 × ER = 1.5, 1.5625, …, 2（9 档），共 27 点。
用法（仓库根目录）：python -m cases.bfs.fom.make_cases"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

r3 = lambda v: float(f"{v:.3g}")


def make():
    lo, hi = np.array([2., 1.5]), np.array([np.log10(400), 2.])
    unit = np.array([(a, b) for b in np.linspace(0, 1, 5) for a in np.linspace(0, 1, 7)])
    rows = []
    for j in range(5):
        for i in range(7):
            z = lo + unit[len(rows)] * (hi - lo)
            rows.append(dict(role="pool", Re=r3(10 ** z[0]), ER=r3(z[1]), n4=int(i % 6 == 0 and j % 4 == 0),
                             n9=int(i % 3 == 0 and j % 2 == 0), n15=int(i % 3 == 0), n21=int(j % 2 == 0), n35=1))
    for seed in range(1000000):
        u = qmc.LatinHypercube(2, seed=seed).random(100)
        if np.abs(u[:, None] - unit).max(2).min() >= 0.02:
            break
    for z in (lo + u[np.argsort(u[:, 0])] * (hi - lo)):
        rows.append(dict(role="interp", Re=r3(10 ** z[0]), ER=r3(z[1]), n4=0, n9=0, n15=0, n21=0, n35=0))
    for re in (450, 520, 600):
        for er in np.linspace(1.5, 2, 9):
            rows.append(dict(role="extrap", Re=float(re), ER=r3(er), n4=0, n9=0, n15=0, n21=0, n35=0))
    counts = {}
    for row in rows:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
        row["id"] = dict(pool="P", interp="I", extrap="E")[row["role"]] + f"{counts[row['role']]:03d}"
    with (Path(__file__).resolve().parents[1] / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["id", "role", "Re", "ER", "n4", "n9", "n15", "n21", "n35"])
        w.writeheader()
        w.writerows(rows)
    print(f"LHS seed={seed}; {counts}")


if __name__ == "__main__":
    make()
