"""只生成参数表；不求解、不生成快照。参数取 3 位有效数字，eta 由取整后的参数计算。运行：python -m cases.conjugate.fom.make_cases"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

PE_LOG = np.linspace(1, 3, 5)
RATIO_LOG = np.linspace(-1, 2, 7)
ROOT = Path(__file__).resolve().parents[1]
r3 = lambda v: float(f"{v:.3g}")


def parameters():
    pool = np.array([(p, k) for p in PE_LOG for k in RATIO_LOG])
    for seed in range(10000):
        test = qmc.scale(qmc.LatinHypercube(2, seed=seed).random(100), [1, -1], [3, 2])
        near = np.max(np.abs(test[:, None, :] - pool[None, :, :]), axis=2) < 0.05
        if not near.any():
            break
    else:
        raise RuntimeError("no separated Latin hypercube found")
    extrap = [(p, k) for p in (3.125, 3.25, 3.5) for k in RATIO_LOG]
    extrap += [(p, k) for k in (-1.125, -1.25, -1.5, 2.125, 2.25, 2.5) for p in PE_LOG]
    rows = []
    for role, prefix, points in (("pool", "P", pool), ("interp", "I", test), ("extrap", "E", extrap)):
        for index, (p, k) in enumerate(points, 1):
            n4 = role == "pool" and p in (1, 3) and k in (-1, 2)
            n9 = role == "pool" and p in (1, 2, 3) and k in (-1, 0.5, 2)
            Pe, ratio = r3(10**p), r3(10**k)
            rows.append(dict(id=f"{prefix}{index:03d}", role=role, Pe=Pe, ratio=ratio,
                             eta1=np.log10(Pe) - 2, eta2=(np.log10(ratio) - 0.5) / 1.5,
                             n4=int(n4), n9=int(n9), lhs_seed=seed if role == "interp" else ""))
    return rows


def run():
    rows = parameters()
    assert sum(r["n4"] for r in rows) == 4 and sum(r["n9"] for r in rows) == 9
    path = ROOT / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases; LHS seed={next(r['lhs_seed'] for r in rows if r['role'] == 'interp')}; {path}")


if __name__ == "__main__":
    run()
