"""后台阶网格研究：n（每入口高度单元数）、ER、Re 上的输出量，只写标量 CSV。

几何与边界取 solvers/openmac/benchmarks.bfs：入口高 h=1、抛物线入口（平均速度 1）、台阶在 x=0，
台阶下游下壁定温 T=1，其余壁绝热，入口 T=0；Re=2U_b h/ν，Pr=0.71。
用法（仓库根目录，PINNGPU）：python -m cases.bfs.fom.grid_study name --n 24 32 48 --ER 1.5 2 --Re 100 200 400 600 [--Lout 40]
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from solvers.openmac.benchmarks import bfs

OUT = Path(__file__).resolve().parents[1] / "results" / "fom"
XS = (1., 2., 5., 10., 20.)                       # 局部 Nu 的采样位置 x/S


def outputs(m, y, T, mu):
    S = mu[1] - 1
    op, Q, w = m.fields(y, mu)
    rec, top = m.recirculation(y, mu), m.recirculation(y, mu, 'upper')
    wall = m.wall_fields(y, T, mu, 'heated')
    nu = m.nusselt(y, T, mu, 'heated', S, 0.)['Nu']
    interior = ~m.grid.boundary
    spacing = np.where(m.grid.axis == 0, op.dx[np.minimum(m.grid.fi, m.grid.nx - 1)],
                       op.dy[np.minimum(m.grid.fj, m.grid.ny - 1)])
    k = int(np.argmax(nu))
    row = dict(xr_S=rec['primary_reattachment'] / S, n_lower_roots=len(rec['reattachment']),
               upper_sep_S=top['separation'][0] / S if len(top['separation']) else np.nan,
               upper_reat_S=top['reattachment'][0] / S if len(top['reattachment']) else np.nan,
               Nu_max=float(nu[k]), Nu_peak_x_S=float(wall['x'][k] / S),
               Nu_avg=float(np.sum(nu * wall['area']) / np.sum(wall['area'])),
               mesh_Re=float(np.max(abs(w[interior]) * spacing[interior] / op.nu)),
               outlet_backflow=float(-np.sum(np.minimum(Q[m.grid.tags == 'outlet'], 0.))),
               T_min=float(T.min()), T_max=float(T.max()))
    for xs in XS:
        row[f"Nu_{xs:g}S"] = float(np.interp(xs * S, wall['x'], nu))
    return row


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.n:
        m = bfs(n, Lin=a.Lin, Lout=a.Lout, extension=a.extension)
        for er in a.ER:
            y = None
            for re in a.Re:
                t0, mu = time.perf_counter(), np.array([re, er])
                prev = y
                y, T, st = m.solve(mu, state=y, tol=a.tol)
                if not st['converged'] and prev is not None:
                    y, T, hist = m.continue_to([rows[-1]['Re'], er], mu, state=prev, tol=a.tol)
                    st = {**hist[-1], 'converged': bool(hist[-1]['converged'] and hist[-1]['fraction'] == 1.)}
                row = dict(n=n, ny=m.grid.ny_state, ER=er, Re=re, Lout=a.Lout + a.extension,
                           converged=st['converged'], iterations=st.get('iterations'), residual=st['residual'])
                if st['converged']:
                    row.update(outputs(m, y, T, mu))
                row['time'] = time.perf_counter() - t0
                rows.append(row)
                with (OUT / f"{a.name}.csv").open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, list(dict.fromkeys(k for r in rows for k in r)))
                    w.writeheader()
                    w.writerows(rows)
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                if not st['converged']:
                    break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--n", type=int, nargs="+", default=[24, 32, 48])
    p.add_argument("--ER", type=float, nargs="+", default=[1.5, 2.])
    p.add_argument("--Re", type=float, nargs="+", default=[100., 200., 400., 600.])
    p.add_argument("--Lin", type=float, default=4.)
    p.add_argument("--Lout", type=float, default=40.)
    p.add_argument("--extension", type=float, default=0.)
    p.add_argument("--tol", type=float, default=1e-10)          # n=64 时残差舍入下限约 2e-10
    run(p.parse_args())
