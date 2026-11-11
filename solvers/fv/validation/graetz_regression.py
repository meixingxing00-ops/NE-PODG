"""逐项比较推广前的 git 版本与当前 Channel；不读取或生成正式快照。

运行：D:/Anaconda/python.exe -m solvers.fv.validation.graetz_regression
参考源码从固定提交读入内存，保证回归可重复，不依赖工作区中的旧数组。
"""
import csv
import subprocess
import types
from pathlib import Path

import numpy as np

from solvers.fv.fv import Channel, grid

BASE = "38d1e0fdbb1b9b0d833c439de458e2b2ea5d74b4"
ROOT = Path(__file__).resolve().parents[3]


def run():
    source = subprocess.check_output(["git", "show", f"{BASE}:solvers/fv/fv.py"], cwd=ROOT).decode("utf-8")
    old = types.ModuleType("graetz_before")
    exec(compile(source, f"git:{BASE}:fv.py", "exec"), old.__dict__)
    rows = []
    for scheme in ("cds", "luds", "quick"):
        before = old.Channel(*old.grid(40, 1.5, 160, 2.0, 80, 2.0), scheme=scheme)
        after = Channel(*grid(40, 1.5, 160, 2.0, 80, 2.0), scheme=scheme)
        for L, Pe in ((1, 0.1), (10, 0.1), (1, 316), (10, 316), (3.16, 10)):
            a, b = before.solve(L, Pe, True), after.solve(L, Pe, True)
            d = [float(np.max(np.abs(x - y))) for x, y in zip(a, b)]
            opdiff = []
            for x, y in zip(before.operators(L), after.operators(L)):
                delta = x - y
                opdiff.append(float(np.max(np.abs(delta.data), initial=0)) if hasattr(delta, "tocsc")
                              else float(np.max(np.abs(delta))))
            post = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in (
                (before.wall(a[0], Pe), after.wall(b[0], Pe)),
                (before.tav(a[0], Pe), after.tav(b[0], Pe)),
                (before.bulk(a[0]), after.bulk(b[0])),
                (before.nusselt(a[0], Pe), after.nusselt(b[0], Pe)),
                (before.balance(a[0], L, Pe), after.balance(b[0], L, Pe))))
            assert max(*d, *opdiff, post) == 0
            rows.append(dict(base=BASE, scheme=scheme, L=L, Pe=Pe, cells=after.n,
                             max_T=d[0], max_eta1=d[1], max_eta2=d[2],
                             max_operators=max(opdiff), max_outputs=post))
    path = Path(__file__).with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(f"Graetz: {len(rows)} full-grid comparisons, all differences zero; {path}", flush=True)


if __name__ == "__main__":
    run()
