"""生成 Brinkman 方腔训练池与独立测试工况；不求解、不写快照。"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def make():
    lo, hi = np.array([3.0, -5.0]), np.array([5.5, -3.0])
    grid = np.array([(a, b) for b in np.linspace(0, 1, 5) for a in np.linspace(0, 1, 7)])
    rows = []
    for j in range(5):
        for i in range(7):
            p = 10 ** (lo + grid[len(rows)] * (hi - lo))
            rows.append(dict(role="pool", Ra=p[0], Da=p[1], n9=int(i % 3 == 0 and j % 2 == 0),
                             n15=int(i % 3 == 0), n21=int(j % 2 == 0), n35=1))
    for seed in range(100000):
        z = qmc.LatinHypercube(2, seed=seed).random(100)
        if np.abs(z[:, None] - grid).max(2).min() >= 0.015:
            break
    else:
        raise RuntimeError("no separated Latin hypercube")
    for z1 in z[np.argsort(z[:, 0])]:
        p = 10 ** (lo + z1 * (hi - lo))
        rows.append(dict(role="interp", Ra=p[0], Da=p[1], n9=0, n15=0, n21=0, n35=0))
    for a in (5.625, 5.75, 6.0):
        for b in np.linspace(-5, -3, 9):
            rows.append(dict(role="extrap", Ra=10 ** a, Da=10 ** b, n9=0, n15=0, n21=0, n35=0))
    counts = {}
    for row in rows:
        role = row["role"]
        counts[role] = counts.get(role, 0) + 1
        row["id"] = dict(pool="P", interp="I", extrap="E")[role] + f"{counts[role]:03d}"
    path = Path(__file__).resolve().parents[1] / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, ["id", "role", "Ra", "Da", "n9", "n15", "n21", "n35"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"LHS seed={seed}; {counts}")


if __name__ == "__main__":
    make()
