"""后台阶的参数延拓扫描、全部工况检查与灵敏度核对；只写标量 CSV（results/fom/）。
用法（仓库根目录，PINNGPU）：python -m cases.bfs.fom.prepare scan|audit|sensitivities"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from cases.bfs.fom.config import deta, eta, model
from cases.bfs.fom.grid_study import outputs
from solvers.openmac.openmac import condition_estimate

OUT = Path(__file__).resolve().parents[1] / "results" / "fom"
TOL = 1e-9                                         # n=48 时残差的舍入下限约 1.5e-10


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def reach(m, mu, y, mu_prev):
    """从上一个解直接 Newton；不收敛时在 (log Re, ER) 中自适应延拓。"""
    y1, T, st = m.solve(mu, state=y, tol=TOL)
    retries = 0
    if not st["converged"] and y is not None:
        y1, T, hist = m.continue_to(mu_prev, mu, state=y, tol=TOL)
        st, retries = {**hist[-1], "converged": bool(hist[-1]["converged"] and hist[-1]["fraction"] == 1.)}, len(hist)
    return y1, T, st, retries


def diagnostics(m, y, T, mu):
    return dict(cond_J=condition_estimate(m.flow_jacobian(y, mu), m.M.diagonal()),
                cond_AT=condition_estimate(m._temperature(y, mu).A, m.Mphat), **outputs(m, y, T, mu))


def scan():
    m, rows, up = model(), [], {}
    res = 10 ** np.arange(np.log10(50), np.log10(800) + 1e-9, 0.125)
    for direction in ("up", "down"):
        for er in (1.4, 1.5, 1.625, 1.75, 1.875, 2., 2.1):
            y, prev = (None, None) if direction == "up" else (up[res[-1], er].copy(), np.array([res[-1], er]))
            for re in (res if direction == "up" else res[::-1]):
                t0, mu = time.perf_counter(), np.array([re, er])
                y, T, st, retries = reach(m, mu, y, prev)
                row = dict(direction=direction, Re=re, ER=er, converged=st["converged"], retries=retries,
                           backward=st.get("linear_backward_error"))
                if st["converged"]:
                    row.update(diagnostics(m, y, T, mu))
                    if direction == "up":
                        up[re, er] = y.copy()
                    else:
                        d = m.whiten(y - up[re, er])
                        row["path_difference_M"] = float(np.linalg.norm(d) / np.linalg.norm(m.whiten(up[re, er] - m.ell)))
                row["time"] = time.perf_counter() - t0
                rows.append(row)
                write("scan.csv", rows)
                print(direction, f"Re={re:.4g} ER={er} conv={st['converged']} xr={row.get('xr_S', np.nan):.4f} "
                      f"condJ={row.get('cond_J', np.nan):.2e}", flush=True)
                assert st["converged"], row
                prev = mu


def audit():
    with (OUT.parents[1] / "cases.csv").open(encoding="utf-8") as fh:
        cases = list(csv.DictReader(fh))
    m, rows, y, prev = model(), [], None, None
    for c in cases:
        t0, mu = time.perf_counter(), np.array([float(c["Re"]), float(c["ER"])])
        y, T, st, retries = reach(m, mu, y, prev)
        assert st["converged"], c
        row = dict(id=c["id"], role=c["role"], Re=mu[0], ER=mu[1], retries=retries, iterations=st.get("iterations"),
                   backward=st.get("linear_backward_error"), **diagnostics(m, y, T, mu))
        row["time"] = time.perf_counter() - t0
        rows.append(row)
        write("audit.csv", rows)
        print(c["id"], f"xr={row['xr_S']:.4f} Numax={row['Nu_max']:.4f} condJ={row['cond_J']:.2e}", flush=True)
        prev = mu


def sensitivities():
    m, rows = model(), []
    for mu in ((100., 1.5), (200., 1.75), (400., 2.)):
        mu = np.array(mu)
        y, T, st = m.solve(np.array([100., mu[1]]), tol=TOL)
        y, T, st, _ = reach(m, mu, y, np.array([100., mu[1]]))
        sy, sT, _ = m.sensitivity(y, T, mu)
        exact_y, exact_T = sy * deta(mu), sT * deta(mu)
        for k in range(2):
            h = 1e-4
            st_ = []
            for sign in (1, -1):
                e = eta(mu)
                e[k] += sign * h
                mu1 = np.array([200 * 2 ** e[0], 1.75 + e[1] / 4])
                y1, T1, s1 = m.solve(mu1, state=y, tol=TOL)
                assert s1["converged"], s1
                st_.append((y1, T1))
            fy, fT = (st_[0][0] - st_[1][0]) / (2 * h), (st_[0][1] - st_[1][1]) / (2 * h)
            rows.append(dict(Re=mu[0], ER=mu[1], derivative=f"eta{k + 1}",
                             psi_M_error=float(np.linalg.norm(m.whiten(exact_y[:, k] - fy)) / np.linalg.norm(m.whiten(fy))),
                             T_L2_error=float(np.linalg.norm(np.sqrt(m.Mphat) * (exact_T[:, k] - fT))
                                              / np.linalg.norm(np.sqrt(m.Mphat) * fT))))
            print(rows[-1], flush=True)
    write("sensitivities.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=("scan", "audit", "sensitivities"))
    dict(scan=scan, audit=audit, sensitivities=sensitivities)[p.parse_args().task]()
