"""顶盖混合对流的参数延拓与网格研究，只写标量 CSV。

PINNGPU，仓库根目录：python -X utf8 -m cases.mixed.fom.prepare scan|grid|audit|sensitivities
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from solvers.mac.mac import Cavity, nodes
from solvers.mac.study import advance, audit_cases, field_errors, interpolate_state, jacobian_audit, scalars, sensitivity_check

OUT = Path(__file__).resolve().parents[1] / "results" / "fom"
PR = 0.71


def mapping(z):
    re, ri = 10.0 ** np.asarray(z)
    return 1 / re, 1 / (re * PR), dict(buoyancy=ri)


def cavity(n, s):
    x = nodes(n, s)
    return Cavity(x, x, lid=lambda x: np.ones_like(x), thermal=True)


def write_rows(path, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def scan(n, s):
    cav, rows = cavity(n, s), []
    # 各固定 Re 沿 Ri 升降扫描，检查所追踪分支的路径一致性。
    re_values = np.geomspace(20, 400, 9)
    ri_values = np.geomspace(0.03, 10, 11)
    forward = {}
    for direction in ("up", "down"):
        for re in re_values:
            psi = T = None
            current = np.log10([20, 0.03])
            targets = ri_values if direction == "up" else ri_values[::-1]
            for ri in targets:
                t0 = time.perf_counter()
                target = np.log10([re, ri])
                try:
                    psi, T, st = advance(cav, current, target, mapping, psi, T)
                    nu, kap, kw = mapping(target)
                    row = dict(direction=direction, N=n, s=s, Re=re, Ri=ri, success=True,
                               **st, **scalars(cav, psi, T), **jacobian_audit(cav, psi, T, nu, kap, **kw))
                    key = (float(re), float(ri))
                    if direction == "up":
                        forward[key] = psi.copy()
                        row["path_difference_M"] = 0.0
                    else:
                        p0 = forward[key]
                        d = psi - p0
                        row["path_difference_M"] = float(np.sqrt(d @ (cav.Mpsi @ d) / (p0 @ (cav.Mpsi @ p0))))
                    current = target
                    row["time"] = time.perf_counter() - t0
                    rows.append(row)
                    write_rows("scan.csv", rows)
                    print(f"{direction} N={n} Re={re:.5g} Ri={ri:.5g} Nu={row['Nu_cold']:.7g} rcond={row['rcond_est']:.2e}", flush=True)
                except RuntimeError as error:
                    print(f"FAILED {direction} {re=} {ri=}: {error}", flush=True)
                    raise
    return rows


def grid():
    rows, states = [], {}
    # 参数四角以及 (100,1)、(200,1)。
    targets = [(20, 0.1), (20, 5), (200, 0.1), (200, 5), (100, 1), (200, 1)]
    for n, s in ((64, 2.0), (128, 2.0), (256, 2.0)):
        cav = cavity(n, s)
        for re, ri in targets:
            target = np.log10([re, ri])
            if n > 64:
                psi, T = interpolate_state(*states[n // 2, re, ri], cav)
                nu, kap, kw = mapping(target)
                psi, T, st = cav.solve(nu, kap, psi, T, tol=1e-10, **kw)
                assert st["converged"], (n, re, ri, st)
                st.update(total_iters=st["iters"], retries=0)
            else:
                psi, T, st = advance(cav, np.log10([20, 0.03]), target, mapping)
            row = dict(N=n, s=s, Re=re, Ri=ri, **st, **scalars(cav, psi, T))
            states[n, re, ri] = (cav, psi, T)
            if n > 64:
                prev, p0, t0 = states[n // 2, re, ri]
                row.update(field_errors(prev, p0, t0, cav, psi, T))
            else:
                row.update(psi_M_error=0.0, T_L2_error=0.0)
            rows.append(row)
            write_rows("grid.csv", rows)
            print(f"grid N={n} Re={re} Ri={ri} Nu={row['Nu_cold']:.9g}", flush=True)
    return rows


def audit(n, s):
    with (OUT.parents[1] / "cases.csv").open(encoding="utf-8") as fh:
        cases = list(csv.DictReader(fh))
    return audit_cases(cavity(n, s), cases, ("Re", "Ri"), mapping, write_rows)


def sensitivities(n, s):
    def chain(nu, kap, kw, q):
        return [-np.log(10) * (nu * q["nu"] + kap * q["kap"]),
                np.log(10) * kw["buoyancy"] * q["buoyancy"]]
    rows = sensitivity_check(cavity(n, s), [(20, .1), (100, 1), (200, 5)], ("Re", "Ri"), mapping, chain, np.log10(20))
    write_rows("sensitivities.csv", rows)
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("scan", "grid", "audit", "sensitivities"))
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--s", type=float, default=2.0)
    args = parser.parse_args()
    dict(scan=scan, grid=lambda n, s: grid(), audit=audit, sensitivities=sensitivities)[args.task](args.N, args.s)
