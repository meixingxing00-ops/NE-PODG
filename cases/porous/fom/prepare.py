"""Brinkman 方腔的参数延拓、网格研究和基准对照，只写标量 CSV。

PINNGPU，仓库根目录：python -X utf8 -m cases.porous.fom.prepare scan|grid|audit|sensitivities
固定有效黏度比 1，忽略动量输运，保留能量输运；参数坐标 z=(log10 Ra*, log10 Da)，Ra*=Ra·Da，Ra 为流体 Rayleigh 数，Da=K/L²。
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
    ras, da = 10.0 ** np.asarray(z)
    ra = ras / da
    nu, kap = np.sqrt(PR / ra), 1 / np.sqrt(ra * PR)
    return nu, kap, dict(drag=nu / da)


def cavity(n, s):
    x = nodes(n, s)
    return Cavity(x, x, thermal=True, inertia=0.0)


def write_rows(path, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def scan(n, s):
    cav, rows, forward = cavity(n, s), [], {}
    ra_values, da_values = 10.0 ** np.arange(1, 4.01, 0.25), 10.0 ** np.arange(-4.5, -1.49, 0.5)
    for direction in ("up", "down"):
        for da in da_values:
            psi = T = None
            current = np.array([1.0, np.log10(da)])
            targets = ra_values if direction == "up" else ra_values[::-1]
            for ra in targets:
                t0 = time.perf_counter()
                target = np.log10([ra, da])
                psi, T, st = advance(cav, current, target, mapping, psi, T)
                nu, kap, kw = mapping(target)
                row = dict(direction=direction, N=n, s=s, Ras=ra, Da=da, success=True, **st,
                           **scalars(cav, psi, T), **jacobian_audit(cav, psi, T, nu, kap, **kw))
                key = (float(ra), float(da))
                if direction == "up":
                    forward[key] = psi.copy()
                    row["path_difference_M"] = 0.0
                else:
                    p0, d = forward[key], psi - forward[key]
                    row["path_difference_M"] = float(np.sqrt(d @ (cav.Mpsi @ d) / (p0 @ (cav.Mpsi @ p0))))
                current = target
                row["time"] = time.perf_counter() - t0
                rows.append(row)
                write_rows("scan.csv", rows)
                print(f"{direction} N={n} Ras={ra:.5g} Da={da:.5g} Nu={row['Nu_cold']:.7g} rcond={row['rcond_est']:.2e}", flush=True)
    return rows


def grid():
    rows, states = [], {}
    targets = [(31.6, 1e-4), (31.6, 1e-2), (3160, 1e-4), (3160, 1e-2), (178, 1e-3), (1000, 1e-4)]
    for n, s in ((64, 2.5), (128, 2.5), (256, 2.5)):
        cav = cavity(n, s)
        for ra, da in targets:
            target = np.log10([ra, da])
            if n > 64:
                psi, T = interpolate_state(*states[n // 2, ra, da], cav)
                nu, kap, kw = mapping(target)
                psi, T, st = cav.solve(nu, kap, psi, T, tol=1e-10, **kw)
                assert st["converged"], (n, ra, da, st)
                st.update(total_iters=st["iters"], retries=0)
            else:
                psi, T, st = advance(cav, [1, np.log10(da)], target, mapping)
            row = dict(N=n, s=s, Ras=ra, Da=da, **st, **scalars(cav, psi, T))
            states[n, ra, da] = (cav, psi, T)
            if n > 64:
                prev, p0, t0 = states[n // 2, ra, da]
                row.update(field_errors(prev, p0, t0, cav, psi, T))
            else:
                row.update(psi_M_error=0.0, T_L2_error=0.0)
            rows.append(row)
            write_rows("grid.csv", rows)
            print(f"grid N={n} Ras={ra:g} Da={da:g} Nu={row['Nu_cold']:.9g}", flush=True)
    return rows


def audit(n, s):
    with (OUT.parents[1] / "cases.csv").open(encoding="utf-8") as fh:
        cases = list(csv.DictReader(fh))
    return audit_cases(cavity(n, s), cases, ("Ras", "Da"), mapping, write_rows)


def sensitivities(n, s):
    def chain(nu, kap, kw, q):
        # Ra = Ra*/Da；nu、kap ∝ Ra^-1/2，drag = nu/Da
        a = np.log(10) / 2 * (nu * q["nu"] + kap * q["kap"] + kw["drag"] * q["drag"])
        return [-a, a - np.log(10) * kw["drag"] * q["drag"]]
    rows = sensitivity_check(cavity(n, s), [(31.6, 1e-4), (178, 1e-3), (3160, 1e-2)], ("Ras", "Da"), mapping, chain, 1.0)
    write_rows("sensitivities.csv", rows)
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("scan", "grid", "audit", "sensitivities"))
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--s", type=float, default=2.5)
    args = parser.parse_args()
    dict(scan=scan, grid=lambda n, s: grid(), audit=audit, sensitivities=sensitivities)[args.task](args.N, args.s)
