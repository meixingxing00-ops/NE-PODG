"""Literature-specific validation runs; save scalar diagnostics and wall profiles."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from solvers.openmac.benchmarks import bfs
from solvers.openmac.openmac import condition_estimate


def run(args):
    m = bfs(args.n, Lin=args.Lin, Lout=args.Lout, Pr=args.Pr, extension=args.extension)
    state, rows = None, []
    for re in args.Re:
        mu = np.array([re, args.ER])
        previous = state
        state, T, status = m.solve(mu, state=state, tol=1.e-10, verbose=True)
        continuation = []
        if not status['converged'] and previous is not None:
            state, T, continuation = m.continue_to([rows[-1]['Re'], args.ER], mu,
                state=previous, tol=1.e-10, verbose=True)
            status = {**continuation[-1], 'converged': bool(
                continuation[-1]['converged'] and continuation[-1]['fraction']==1.)}
        row = dict(Re=re, ER=args.ER, n=args.n, n_y=m.grid.ny_state, n_T=m.grid.np,
                   Lin=args.Lin, Lout=args.Lout+args.extension, base_Lout=args.Lout,
                   extension=args.extension, Pr=args.Pr, status=status, continuation=continuation)
        if not status['converged']:
            rows.append(row)
            print(row, flush=True)
            break
        op, Q, w = m.fields(state, mu)
        xr, x, shear = m.reattachment(state, mu)
        recirculation = m.recirculation(state, mu)
        wall = m.wall_fields(state, T, mu, 'heated')
        nu = m.nusselt(state, T, mu, 'heated', args.ER-1, 0.)['Nu']
        peak = int(np.argmax(nu))
        interior = ~m.grid.boundary
        # Directional mesh Reynolds/Peclet numbers at primal internal faces.
        spacing = np.where(m.grid.axis==0, op.dx[np.minimum(m.grid.fi, m.grid.nx-1)],
                            op.dy[np.minimum(m.grid.fj, m.grid.ny-1)])
        row.update(xr_h=xr.tolist(), xr_S=(xr/(args.ER-1)).tolist(),
                   primary_xr_S=recirculation['primary_reattachment']/(args.ER-1),
                   separation_h=recirculation['separation'].tolist(),
                   Nu_max=float(nu[peak]), Nu_peak_x_S=float(wall['x'][peak]/(args.ER-1)),
                   Nu_average=float(np.sum(nu*wall['area'])/np.sum(wall['area'])),
                   max_mesh_Re=float(np.max(abs(w[interior])*spacing[interior]/op.nu)),
                   max_mesh_Pe=float(np.max(abs(w[interior])*spacing[interior]/op.kappa)),
                   outlet_backflow=float(-np.sum(np.minimum(Q[m.grid.tags=='outlet'], 0.))),
                   pressure_recovery=m.pressure(state, mu)[1],
                   condition_J=condition_estimate(m.flow_jacobian(state, mu), m.M.diagonal()),
                   condition_AT=condition_estimate(m._temperature(state, mu).A, m.Mphat),
                   balances=m.balances(state, T, mu))
        u, v = m.faces(state, mu)
        top = m.recirculation(state, mu, 'upper')
        row['upper_wall_separation_h'] = top['separation'].tolist()
        row['upper_wall_reattachment_h'] = top['reattachment'].tolist()
        downstream = op.xc>0
        uu = .5*(u[:-1]+u[1:])
        vv = .5*(v[:,:-1]+v[:,1:])
        # Only the rectangular downstream subdomain is interpolated.
        us = RegularGridInterpolator((op.xc[downstream], op.yc), uu[downstream])
        vs = RegularGridInterpolator((op.xc[downstream], op.yc), vv[downstream])
        locations = np.array([7.,15.]) * args.ER
        ysample = np.linspace(-(args.ER-1), 1., 11)[1:-1]
        row['velocity_profiles'] = [dict(x_h=float(xx), y_h=ysample.tolist(),
            u=us(np.column_stack([np.full(len(ysample),xx),ysample])).tolist(),
            v=vs(np.column_stack([np.full(len(ysample),xx),ysample])).tolist())
            for xx in locations if xx < op.xc[-1]]
        selected = np.linspace(0, len(nu)-1, min(180,len(nu)), dtype=int)
        row['wall_profile'] = dict(x_S=(wall['x'][selected]/(args.ER-1)).tolist(),
                                   Nu=nu[selected].tolist(), shear=shear[selected].tolist())
        rows.append(row)
        print({k:v for k,v in row.items() if k not in ('wall_profile','velocity_profiles')}, flush=True)
        result = dict(date=datetime.now().astimezone().isoformat(), interpreter=sys.executable,
                      result=rows)
        (Path(__file__).with_name('results')/(args.name+'.json')).write_text(
            json.dumps(result, indent=2), encoding='utf-8')
    result = dict(date=datetime.now().astimezone().isoformat(), interpreter=sys.executable, result=rows)
    (Path(__file__).with_name('results')/(args.name+'.json')).write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--Lin', type=float, default=4.)
    p.add_argument('--Lout', type=float, default=30.)
    p.add_argument('--extension', type=float, default=0.)
    p.add_argument('--ER', type=float, default=101/52)
    p.add_argument('--Re', type=float, nargs='+', default=[50.,100.,150.,300.])
    p.add_argument('--Pr', type=float, default=.71)
    run(p.parse_args())
