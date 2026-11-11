"""生成 Brinkman 方腔训练池与独立测试工况；不求解、不写快照。参数 (Ra*, Da)，Ra* = Ra·Da，取 3 位有效数字。
内插测试与任一训练池点在 log10 Ra*、log10 Da 上都相差不到 0.05 的取样作废，按种子从 0 起取第一个全部满足的种子。"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

r3 = lambda v: float(f"{v:.3g}")


def make():
    lo, hi = np.array([1.5, -4.0]), np.array([3.0, -2.0])
    pool = np.array([(a, b) for b in np.linspace(lo[1], hi[1], 5) for a in np.linspace(lo[0], hi[0], 7)])
    rows = []
    for j in range(5):
        for i in range(7):
            p = 10 ** pool[len(rows)]
            rows.append(dict(role="pool", Ras=r3(p[0]), Da=r3(p[1]), n4=int(i % 6 == 0 and j % 4 == 0), n9=int(i % 3 == 0 and j % 2 == 0),
                             n15=int(i % 3 == 0), n21=int(j % 2 == 0), n35=1))
    for seed in range(10000000):
        z = lo + qmc.LatinHypercube(2, seed=seed).random(100) * (hi - lo)
        if np.abs(z[:, None] - pool).max(2).min() >= 0.05:
            break
    for z1 in z[np.argsort(z[:, 0])]:
        p = 10 ** z1
        rows.append(dict(role="interp", Ras=r3(p[0]), Da=r3(p[1]), n4=0, n9=0, n15=0, n21=0, n35=0))
    for a in (3.125, 3.25, 3.5):
        for b in np.linspace(-4, -2, 9):
            rows.append(dict(role="extrap", Ras=r3(10 ** a), Da=r3(10 ** b), n4=0, n9=0, n15=0, n21=0, n35=0))
    counts = {}
    for row in rows:
        role = row["role"]
        counts[role] = counts.get(role, 0) + 1
        row["id"] = dict(pool="P", interp="I", extrap="E")[role] + f"{counts[role]:03d}"
    path = Path(__file__).resolve().parents[1] / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, ["id", "role", "Ras", "Da", "n4", "n9", "n15", "n21", "n35"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"LHS seed={seed}; {counts}")


if __name__ == "__main__":
    make()
