"""在候选正式网格上扫描参数、核对坐标灵敏度；只保存标量诊断。

运行：python -m cases.conjugate.fom.parameter_checks
"""
import csv
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from cases.conjugate.fom.config import mesh, outputs
from cases.conjugate.fom.make_cases import parameters

ROOT = Path(__file__).resolve().parents[1] / "results" / "fom"


def run():
    ch = mesh(2)
    rows = []
    for r in parameters():
        Pe, ratio = r["Pe"], r["ratio"]
        A, f = ch.matrix(Pe, ratio)
        lu = spla.splu(A)
        T = lu.solve(f)
        O = outputs(ch, T, Pe, ratio)
        backward = np.linalg.norm(A @ T - f) / (spla.norm(A) * np.linalg.norm(T) + np.linalg.norm(f))
        defect = abs(ch.balance(T, Pe, ratio)["defect"] / O["Q"])
        assert T.min() > -1e-9 and T.max() < 1 + 1e-9
        assert backward < 1e-12 and defect < 1e-8
        rows.append(dict(id=r["id"], role=r["role"], Pe=Pe, ratio=ratio,
                         Tmin=float(T.min()), Tmax=float(T.max()), backward_error=backward,
                         balance=defect, Q=O["Q"], Tw=O["Tw"], Tout=O["Tout"]))
    path = ROOT / "parameter_checks.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(f"parameter scan: {len(rows)} points, cells={ch.n}", flush=True)

    rows = []
    for Pe, ratio in ((10, 0.1), (10, 100), (1000, 0.1), (1000, 100), (100, 10**0.5)):
        _, sPe, sRatio = ch.solve(Pe, ratio, True)
        analytical = [np.log(10) * Pe * sPe, 1.5 * np.log(10) * ratio * sRatio]
        for j, scale in enumerate((1, 1.5)):
            h = 1e-5
            up = (Pe * 10**(scale * h), ratio) if j == 0 else (Pe, ratio * 10**(scale * h))
            down = (Pe * 10**(-scale * h), ratio) if j == 0 else (Pe, ratio * 10**(-scale * h))
            fd = (ch.solve(*up) - ch.solve(*down)) / (2 * h)
            s = analytical[j]
            e = np.sqrt(ch.Mh @ (s - fd)**2 / (ch.Mh @ s**2))
            assert e < 1e-5
            rows.append(dict(Pe=Pe, ratio=ratio, coordinate=f"eta{j+1}", relative_error=e,
                             norm_sensitivity=float(np.sqrt(ch.Mh @ s**2))))
    path = ROOT / "sensitivity_checks.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(f"coordinate sensitivities: max relative difference={max(r['relative_error'] for r in rows):.3e}", flush=True)


if __name__ == "__main__":
    run()
