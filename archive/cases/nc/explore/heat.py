"""闭合结果汇总：各预算下 POD、SA-PODG、NE-PODG、SA + 网络方向的速度误差 e、温度误差 e_T、平均 Nu 误差 e_Nu 与单次查询时间，
按 common.SEG 分段取中位数（%，未收敛工况不计入并注明个数）。
来源：主线 cases/nc/results/n<B>/{pod,sapodg,V_v}_<role>.csv；explore/results/n<B>/sane_<role>.csv（N6 为 sa、sane_n6sup）。
用法：python -m cases.nc.explore.heat"""
import csv
from pathlib import Path

import numpy as np

from cases.nc.explore.common import SEG

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "results"
SRC = {6: {"SA-PODG": HERE / "results/n6/sa", "SA+网络": HERE / "results/n6/sane_n6sup"}}
for b in (12, 20, 30, 50):
    SRC[b] = {"POD": MAIN / f"n{b}/pod", "SA-PODG": MAIN / f"n{b}/sapodg", "NE-PODG": MAIN / f"n{b}/V_v",
              "SA+网络": HERE / f"results/n{b}/sane"}


def load(stem):
    rows = []
    for role in ("interp", "extrap"):
        p = Path(f"{stem}_{role}.csv")
        if p.exists():
            rows += [dict(r, role=role) for r in csv.DictReader(open(p))]
    return rows


for col in ("e", "e_T", "e_Nu", "time"):
    print(f"\n### {col}\n")
    for b, methods in SRC.items():
        data = {m: load(s) for m, s in methods.items()}
        data = {m: r for m, r in data.items() if r}
        print(f"N{b} | " + " | ".join(data) + " |")
        for role, f, name in SEG:
            out = []
            for m, rows in data.items():
                sub = [r for r in rows if r["role"] == role and f(float(r["Ra"]))]
                ok = [float(r[col]) for r in sub if r["converged"] == "True"]
                v = np.median(ok) * (1 if col == "time" else 100)
                out.append(f"{v:.3g}" + (f"（{len(sub) - len(ok)} 未收敛）" if len(ok) < len(sub) else ""))
            print(f"| {name} | " + " | ".join(out) + " |")
