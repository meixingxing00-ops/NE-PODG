"""Discrete identities, independent differences and manufactured-solution checks.

Run with the PINNGPU interpreter from the repository root. Only JSON scalars are
saved; no full-order snapshots are written.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from solvers.openmac.benchmarks import bfs, channel, inlet_psi
from solvers.openmac.openmac import condition_estimate


def relative(a, b, weights=None):
    norm = lambda v: np.linalg.norm(v) if weights is None else np.sqrt(np.sum(weights * v**2))
    return float(norm(a - b) / max(norm(b), 1.e-30))


def ops():
    rng = np.random.default_rng(717)
    rows = []
    for name, model in [('channel', channel(4)),
                        ('step_uniform', bfs(3, Lin=2, Lout=4, uniform=True)),
                        ('step_stretched', bfs(3, Lin=2, Lout=4))]:
        g = model.grid
        metric = model.M.copy()
        for er in (1.5, 1.77, 2.):
            mu = np.array([80., er])
            op = model.operators(mu)
            y = model.lift(mu) + .1 * rng.normal(size=g.ny_state)
            _, Q, w = model.fields(y, mu)
            DU, B = op.DU.toarray(), op.B.toarray()
            null = la.null_space(DU)
            test = null @ rng.normal(size=null.shape[1])
            reconstructed = B @ la.lstsq(B, test)[0]
            p = rng.normal(size=g.np)
            D = sp.diags(1 / op.Mp) @ op.Dvol
            raw = D @ w
            scaled_div = np.max(abs(raw)) / max(1., np.max(np.asarray(abs(D).sum(axis=1))) * max(abs(w)))
            altered = y.copy()
            outlet_node = np.intersect1d(g.free, g.ends[g.tags == 'outlet'])
            altered[np.flatnonzero(g.free == outlet_node[len(outlet_node) // 2])] += .2
            q2 = model.fields(altered, mu)[1]
            rows.append(dict(configuration=name, ER=er, n_y=g.ny_state, n_U=g.nu, n_p=g.np,
                             DfK=float(abs(g.Df @ g.K).max()), DU_B=float(abs(op.DU @ op.B).max()),
                             scaled_divergence=float(scaled_div), rank_DU=int(np.linalg.matrix_rank(DU)),
                             rank_B=int(np.linalg.matrix_rank(B)), dimension_identity=bool(g.ny_state == g.nu - g.np),
                             nullspace_representation=relative(reconstructed, test),
                             pressure_cancellation=float(np.max(abs(op.B.T @ op.Gweak @ p))),
                             piola_flux=float(abs(sp.diags(op.area) @ op.C - g.K).max()),
                             metric_change=float(abs(model.M - metric).max()),
                             metric_min_eigenvalue=float(la.eigvalsh(model.M.toarray())[0]),
                             inflow=float(-np.sum(g.sign[g.tags == 'inlet'] * Q[g.tags == 'inlet'])),
                             outflow=float(np.sum(g.sign[g.tags == 'outlet'] * Q[g.tags == 'outlet'])),
                             outlet_profile_change=float(np.linalg.norm((q2-Q)[g.tags == 'outlet'])),
                             outlet_total_change=float(np.sum((q2-Q)[g.tags == 'outlet']))))
    return rows


def derivatives():
    rng = np.random.default_rng(81)
    out = {}
    traction = lambda x, y, mu: (.02 * (1 + .1 * y) * mu[1] / mu[0],
                                .03 * np.sin(np.pi * y) / mu[0])
    mu = np.array([137., 1.77])
    for scheme in ('cds', 'luds'):
        model = bfs(6, Lin=2, Lout=8, scheme=scheme, traction=traction)
        g = model.grid
        state = model.lift(mu) + .01 * rng.normal(size=g.ny_state)
        T = rng.normal(size=g.np)
        v, t = rng.normal(size=g.ny_state), rng.normal(size=g.np)
        v /= np.linalg.norm(v)
        t /= np.linalg.norm(t)
        jy, _, jt, att = model.jacobian(state, T, mu)
        jp, tp = model.parameter_derivatives(state, T, mu)
        jrows, prows = [], []
        for h in (1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7):
            plus = model.residual(state + h*v, T + h*t, mu)
            minus = model.residual(state - h*v, T - h*t, mu)
            qa, qb = model.fields(state + h*v, mu)[1], model.fields(state - h*v, mu)[1]
            changes = int(np.count_nonzero((qa.real >= 0) != (qb.real >= 0)))
            jrows.append(dict(step=h, switches=changes,
                              flow=relative((plus[0]-minus[0])/(2*h), jy @ v),
                              temperature=relative((plus[1]-minus[1])/(2*h), jt @ v + att @ t)))
            for k, name in enumerate(('Re', 'ER')):
                dz = np.zeros(2)
                dz[k] = mu[k] * h
                plus, minus = model.residual(state, T, mu+dz), model.residual(state, T, mu-dz)
                prows.append(dict(parameter=name, relative_step=h,
                                  flow=relative((plus[0]-minus[0])/(2*dz[k]), jp[:, k]),
                                  temperature=relative((plus[1]-minus[1])/(2*dz[k]), tp[:, k])))
        T0, sv, smu = model.temperature(state, mu, v[:, None], parameters=True)
        auxrows = []
        for h in (1.e-4, 1.e-5, 1.e-6):
            finite = (model.temperature(state+h*v, mu)-model.temperature(state-h*v, mu))/(2*h)
            qa, qb = model.fields(state + h*v, mu)[1], model.fields(state - h*v, mu)[1]
            changes = int(np.count_nonzero((qa.real >= 0) != (qb.real >= 0)))
            auxrows.append(dict(step=h, switches=changes, error=relative(finite, sv[:, 0])))
        auxparams = []
        for k, name in enumerate(('Re', 'ER')):
            dz = np.zeros(2)
            dz[k] = mu[k]*1.e-5
            finite = (model.temperature(state, mu+dz)-model.temperature(state, mu-dz))/(2*dz[k])
            auxparams.append(dict(parameter=name, error=relative(finite, smu[:, k])))
        out[scheme] = dict(state_jacobian=jrows, parameter_partials=prows,
                           auxiliary_direction=auxrows, auxiliary_parameter=auxparams)
    model = bfs(10, Lin=3, Lout=12)
    state, T, status = model.solve(mu, tol=2.e-12)
    assert status['converged'], status
    sy, st, sw = model.sensitivity(state, T, mu)
    op, Q, w = model.fields(state, mu)
    rows = []
    for h in (1.e-3, 3.e-4, 1.e-4, 3.e-5, 1.e-5):
        for k, name in enumerate(('Re', 'ER')):
            delta = np.zeros(2)
            delta[k] = h * mu[k]
            yp, tp, sp_ = model.solve(mu+delta, state=state, tol=2.e-12)
            ym, tm, sm_ = model.solve(mu-delta, state=state, tol=2.e-12)
            assert sp_['converged'] and sm_['converged'], (sp_, sm_)
            wp, wm = model.fields(yp, mu+delta)[2], model.fields(ym, mu-delta)[2]
            qp, qm = model.fields(yp, mu+delta)[1], model.fields(ym, mu-delta)[1]
            changes = int(np.count_nonzero((qp.real >= 0) != (qm.real >= 0)))
            rows.append(dict(parameter=name, relative_step=h, switches=changes,
                             flow=relative(model.whiten((yp-ym)/(2*delta[k])), model.whiten(sy[:, k])),
                             temperature=relative((tp-tm)/(2*delta[k]), st[:, k], model.Mphat),
                             physical_velocity=relative((wp-wm)/(2*delta[k]), sw[:, k], op.Mu),
                             max_residual=max(sp_['residual'], sm_['residual'])))
    logrows = []
    for k, name in enumerate(('log10_Re', 'log10_ER')):
        h = 1.e-5
        mup, mum = mu.copy(), mu.copy()
        mup[k] *= 10**h
        mum[k] *= 10**(-h)
        yp, _, a = model.solve(mup, state=state, tol=2.e-12)
        ym, _, b = model.solve(mum, state=state, tol=2.e-12)
        assert a['converged'] and b['converged']
        exact = sy[:, k] * mu[k] * np.log(10.)
        logrows.append(dict(parameter=name, error=relative(model.whiten((yp-ym)/(2*h)), model.whiten(exact))))
    out['steady'] = dict(status=status, differences=rows, logarithmic_coordinates=logrows,
                         pressure_recovery=model.pressure(state, mu)[1],
                         condition_J=condition_estimate(model.flow_jacobian(state, mu), model.M.diagonal()),
                         condition_AT=condition_estimate(model._temperature(state, mu).A, model.Mphat),
                         balances=model.balances(state, T, mu))
    return out


def steady_blocks():
    rng = np.random.default_rng(120)
    model = bfs(8, Lin=2, Lout=10)
    mu = np.array([137., 1.77])
    y, T, status = model.solve(mu, tol=2.e-12)
    assert status['converged'], status
    v, t = rng.normal(size=y.shape), rng.normal(size=T.shape)
    v /= np.linalg.norm(v)
    t /= np.linalg.norm(t)
    jy, zero, jt, at = model.jacobian(y, T, mu)
    rows = []
    for h in (1.e-4, 1.e-5, 1.e-6, 1.e-7):
        a, b = model.residual(y+h*v, T+h*t, mu), model.residual(y-h*v, T-h*t, mu)
        qa, qb = model.fields(y+h*v, mu)[1], model.fields(y-h*v, mu)[1]
        rows.append(dict(step=h, switches=int(np.count_nonzero((qa>=0) != (qb>=0))),
                         flow=relative((a[0]-b[0])/(2*h), jy@v),
                         temperature=relative((a[1]-b[1])/(2*h), jt@v+at@t)))
    # Exact T=1-y: lower wall unit inward gradient, upper wall unit outward
    # gradient. This independently checks the signed Neumann load and wall output.
    model = channel(12)
    model.thermal_bc = lambda x, y, tag, mu: (('dirichlet', 1-y) if tag=='inlet'
                          else ('neumann', 0. if tag=='outlet' else
                                model.coefficients(mu)[1] * np.where(y<.5, 1., -1.)))
    model.source = lambda x, y, mu: 0.
    op = model.operators(mu)
    exact = inlet_psi(model.grid.xv.ravel()[model.grid.free],
                      model.grid.yv.ravel()[model.grid.free], mu)
    tn = model.temperature(exact, mu)
    wall = model.wall_fields(exact, tn, mu, 'wall')
    return dict(state_jacobian=rows, passive_zero_block=float(abs(zero).max()),
                neumann_temperature_error=relative(tn, 1-op.yc[model.grid.cj], model.Mphat),
                neumann_wall_gradient_error=float(np.max(abs(wall['gradient']-np.where(wall['y']<.5,1.,-1.)))),
                neumann_wall_temperature_error=float(np.max(abs(wall['Tw']-(1-wall['y'])))),
                neumann_balances=model.balances(exact, tn, mu))


def continuation():
    model = bfs(4, Lin=2, Lout=8)
    target = np.array([80., 2.])
    y, T, history = model.continue_to([40., 1.5], target, step=.5, tol=1.e-11)
    assert history[-1]['fraction']==1. and history[-1]['converged'], history
    direct, _, status = model.solve(target, tol=1.e-11)
    assert status['converged'], status
    lower = model.recirculation(y, target)
    upper = model.recirculation(y, target, 'upper')
    return dict(history=history, direct_status=status,
                path_difference=relative(model.whiten(y), model.whiten(direct)),
                lower_roots=lower['reattachment'].tolist(),
                primary_reattachment=lower['primary_reattachment'],
                upper_separation=upper['separation'].tolist(),
                upper_reattachment=upper['reattachment'].tolist())


def manufactured():
    rows = []
    mu = np.array([20., 2.])
    for scheme in ('cds', 'luds'):
        for n in (8, 16, 32, 64):
            model = channel(n, scheme=scheme)
            g = model.grid
            state, T, status = model.solve(mu, tol=2.e-11)
            assert status['converged'], status
            exact_state = inlet_psi(g.xv.ravel()[g.free], g.yv.ravel()[g.free], mu)
            op = model.operators(mu)
            exact_T = 1 + np.cos(np.pi * op.xc[g.ci] / 2) * np.sin(np.pi * op.yc[g.cj])
            T_isolated = model.temperature(exact_state, mu)
            rows.append(dict(scheme=scheme, n=n, n_y=g.ny_state,
                             velocity_error=relative(model.whiten(state), model.whiten(exact_state)),
                             temperature_error=relative(T, exact_T, model.Mphat),
                             isolated_temperature_error=relative(T_isolated, exact_T, model.Mphat),
                             pressure_recovery=model.pressure(state, mu)[1], status=status,
                             balances=model.balances(state, T, mu)))
            print(scheme, n, rows[-1]['velocity_error'], rows[-1]['isolated_temperature_error'], flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=('ops', 'derivatives', 'manufactured', 'steady_blocks','continuation'))
    args = parser.parse_args()
    result = dict(date=datetime.now().astimezone().isoformat(),
                  interpreter=sys.executable, result=globals()[args.part]())
    path = Path(__file__).with_name('results')
    path.mkdir(exist_ok=True)
    (path / (args.part + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
