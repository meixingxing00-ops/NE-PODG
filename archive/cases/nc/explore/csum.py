"""闭合结果 explore/results/n<B>/<space>_{interp,extrap}.csv（或主线 cases/nc/results）按区段取中位数（%）与未收敛数。
用法：python -m cases.nc.explore.csum B 列 space1 space2 ...（space 以 main/ 开头时读主线结果）"""
import csv
import sys
from pathlib import Path

import numpy as np

from cases.nc.explore.common import SEG

HERE = Path(__file__).resolve().parent
budget, col, spaces = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(sp):
    d = HERE.parent / "results" if sp.startswith("main/") else HERE / "results"
    sp = sp.removeprefix("main/")
    rows = []
    for role in ("interp", "extrap"):
        for r in csv.DictReader(open(d / f"n{budget}" / f"{sp}_{role}.csv", encoding="utf-8")):
            r["role"] = role
            rows.append(r)
    return rows


data = {sp: load(sp) for sp in spaces}
print(f"| 区段 | n | " + " | ".join(spaces) + " |")
for role, f, name in SEG:
    cells = []
    for sp in spaces:
        sub = [r for r in data[sp] if r["role"] == role and f(float(r["Ra"]))]
        bad = sum(r["converged"] != "True" for r in sub)
        cells.append(f"{np.median([float(r[col]) for r in sub]) * 100:.3g}" + (f"（{bad} 未收敛）" if bad else ""))
    print(f"| {name} | {len(sub)} | " + " | ".join(cells) + " |")
