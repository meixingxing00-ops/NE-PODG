"""共轭传热网格收敛；只保存标量结果，不生成快照。

运行：D:/Anaconda/python.exe -m cases.conjugate.fom.grid_study
ref=8 (327680 单元)；level=0.5,1,2,4。分区插值，不跨界面。
"""
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cases.conjugate.fom.config import mesh, outputs

CASES = [(10, 0.1), (10, 100), (1000, 0.1), (1000, 100), (100, 10**0.5),
         (10, 10**-1.5), (1000, 10**2.5), (10**3.5, 0.1), (10**3.5, 100)]
LEVELS = (0.5, 1, 2, 4)
ROOT = Path(__file__).resolve().parents[1] / "results" / "fom"


def select_grid(rows):
    """以全部验证工况的误差选取网格族中最粗的合格网格。"""
    accepted = []
    for level in LEVELS:
        subset = [r for r in rows if float(r["level"]) == level]
        assert len(subset) == len(CASES)
        if (max(float(r[k]) for r in subset for k in ("e_Tf", "e_Ts", "e_Q", "e_Tw", "e_Tout")) <= 5e-4
                and max(float(r["e_Nu"]) for r in subset) <= 2e-3):
            accepted.append(level)
    if not accepted:
        raise RuntimeError("no grid satisfies field/output tolerances")
    print(f"coarsest accepted level: {min(accepted):g}", flush=True)
    return min(accepted)


def run():
    rows = []
    ref = mesh(8)
    for Pe, ratio in CASES:
        Tref = ref.solve(Pe, ratio)
        Oref = outputs(ref, Tref, Pe, ratio)
        mask_ref = ref.yc < 0
        interpolators = [RegularGridInterpolator((ref.xc, ref.yc[mask]),
                            Tref.reshape(ref.nx, ref.ny)[:, mask], bounds_error=False, fill_value=None)
                         for mask in (mask_ref, ~mask_ref)]
        print(f"reference Pe={Pe:.6g}, ratio={ratio:.6g}, cells={ref.n}", flush=True)
        for level in LEVELS:
            ch = mesh(level)
            T = ch.solve(Pe, ratio)
            O = outputs(ch, T, Pe, ratio)
            tm = T.reshape(ch.nx, ch.ny)
            errors = []
            for mask, interpolate in zip((ch.yc < 0, ch.yc > 0), interpolators):
                points = np.stack(np.meshgrid(ch.xc, ch.yc[mask], indexing="ij"), -1).reshape(-1, 2)
                tr = interpolate(points)
                weights = (ch.dx[:, None] * ch.dy[mask]).ravel()
                errors.append(np.sqrt(weights @ (tm[:, mask].ravel() - tr)**2 / (weights @ tr**2)))
            rows.append(dict(Pe=Pe, ratio=ratio, level=level, cells=ch.n, reference_cells=ref.n,
                             e_Ts=errors[0], e_Tf=errors[1], Q=O["Q"], Tw=O["Tw"], Tout=O["Tout"],
                             e_Q=abs(O["Q"] / Oref["Q"] - 1), e_Tw=abs(O["Tw"] / Oref["Tw"] - 1),
                             e_Tout=abs(O["Tout"] / Oref["Tout"] - 1),
                             e_Nu=float(np.max(np.abs(O["Nu"] / Oref["Nu"] - 1))),
                             Tmin=float(T.min()), Tmax=float(T.max()),
                             balance=abs(ch.balance(T, Pe, ratio)["defect"] / O["Q"])))
            print(f" level={level:g}, eTf={errors[1]:.3e}, eTs={errors[0]:.3e}, "
                  f"eQ={rows[-1]['e_Q']:.3e}, eNu={rows[-1]['e_Nu']:.3e}", flush=True)
        path = ROOT / "grid_study.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    select_grid(rows)


if __name__ == "__main__":
    run()
