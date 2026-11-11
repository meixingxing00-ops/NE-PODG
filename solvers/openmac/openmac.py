"""Pressure-free steady MAC flow on a simply connected, orthogonal open channel.

All residuals are control-volume integrals. Geometry and boundary callbacks may
receive complex parameters for complex-step parameter derivatives. Arrays use
[x, y] indexing; temperature is numbered on active cells only.
"""
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from solvers.mac.mac import _Direct, _keys


def _matrix(rows, cols, values, shape):
    return sp.csr_matrix((np.concatenate(values),
                          (np.concatenate(rows), np.concatenate(cols))), shape=shape)


class _Factor:
    """cuDSS factorization (solvers.mac._Direct). Re-plans on the union pattern when a matrix leaves the
    analysed nonzero structure (e.g. LUDS stencil switches) or needs more right-hand-side columns."""

    def __init__(self):
        self.d, self.cols = None, 0

    @staticmethod
    def _pattern(A):
        return sp.csr_matrix((np.ones(A.nnz), A.indices.copy(), A.indptr.copy()), shape=A.shape)

    def factor(self, A, cols=1):
        A = sp.csr_matrix(A)
        A.sort_indices()
        if self.d is None or cols > self.cols or not np.isin(_keys(A), self.d.key).all():
            P = self._pattern(A) if self.d is None else self._pattern(A) + self._pattern(self.d.A)
            self.cols = max(cols, self.cols)
            self.d = _Direct(P.tocsr(), self.cols)
        self.d.factor(A)
        return self.solve

    def solve(self, B):
        B = np.asarray(B, dtype=float)
        X = self.d.solve(B)
        return X[:, 0] if B.ndim == 1 else X


def _put(store, r, c, v):
    r, c = np.broadcast_arrays(r, c)
    store[0].append(r.ravel())
    store[1].append(c.ravel())
    store[2].append(np.broadcast_to(v, r.shape).ravel())


class Grid:
    """Fixed reference topology. tag(axis, i, j, outward_sign) labels boundary faces.

    Only 'outlet' frees a normal boundary velocity. All other labels prescribe
    velocity and streamfunction; their thermal conditions are supplied separately.
    """

    def __init__(self, x, y, active, tag):
        self.x, self.y, self.active = np.array(x), np.array(y), np.array(active)
        nx, ny = self.active.shape
        self.nx, self.ny = nx, ny
        self.tid = -np.ones((nx, ny), int)
        self.ci, self.cj = np.where(self.active)
        self.np = len(self.ci)
        self.tid[self.active] = np.arange(self.np)
        ua, va = np.zeros((nx + 1, ny), bool), np.zeros((nx, ny + 1), bool)
        ua[:-1] |= active
        ua[1:] |= active
        va[:, :-1] |= active
        va[:, 1:] |= active
        ui, uj = np.where(ua)
        vi, vj = np.where(va)
        self.uid, self.vid = -np.ones(ua.shape, int), -np.ones(va.shape, int)
        self.nf = len(ui) + len(vi)
        self.uid[ua] = np.arange(len(ui))
        self.vid[va] = len(ui) + np.arange(len(vi))
        self.axis = np.r_[np.zeros(len(ui), int), np.ones(len(vi), int)]
        self.fi, self.fj = np.r_[ui, vi], np.r_[uj, vj]
        tc = np.pad(self.tid, 1, constant_values=-1)
        self.p = np.r_[tc[ui, uj + 1], tc[vi + 1, vj]]
        self.n = np.r_[tc[ui + 1, uj + 1], tc[vi + 1, vj + 1]]
        self.boundary = (self.p < 0) | (self.n < 0)
        self.sign = np.where(self.p >= 0, 1, -1)
        self.tags = np.full(self.nf, '', object)
        for f in np.flatnonzero(self.boundary):
            self.tags[f] = tag(self.axis[f], self.fi[f], self.fj[f], self.sign[f])
        self.U = np.flatnonzero(~self.boundary | (self.tags == 'outlet'))
        self.nu = len(self.U)
        nv = (nx + 1) * (ny + 1)
        a = self.fi * (ny + 1) + self.fj
        b = a + np.where(self.axis == 0, 1, ny + 1)
        s = np.where(self.axis == 0, 1, -1)
        self.ends = np.column_stack([a, b])
        self.K = sp.csr_matrix((np.r_[-s, s],
                               (np.r_[np.arange(self.nf), np.arange(self.nf)], np.r_[a, b])),
                              shape=(self.nf, nv))
        ii, jj, ids = self.ci, self.cj, np.arange(self.np)
        self.Df = sp.csr_matrix((np.r_[np.ones(self.np), -np.ones(self.np),
                                       np.ones(self.np), -np.ones(self.np)],
                                (np.tile(ids, 4), np.r_[self.uid[ii + 1, jj], self.uid[ii, jj],
                                                       self.vid[ii, jj + 1], self.vid[ii, jj]])),
                               shape=(self.np, self.nf))
        fixed = np.unique(self.ends[self.boundary & (self.tags != 'outlet')])
        all_boundary = np.unique(self.ends[self.boundary])
        vertices = np.unique(self.ends)
        self.fixed = fixed
        self.free = np.setdiff1d(vertices, fixed)
        self.inside = np.setdiff1d(vertices, all_boundary)
        self.ny_state = len(self.free)
        self.E = sp.csr_matrix((np.ones(len(self.free)), (self.free, np.arange(len(self.free)))),
                              shape=(nv, len(self.free)))
        self.xv, self.yv = np.meshgrid(self.x, self.y, indexing='ij')
        self.outlet_paths = []
        neighbors = {}
        for a, b in self.ends[self.tags == 'outlet']:
            neighbors.setdefault(a, []).append(b)
            neighbors.setdefault(b, []).append(a)
        while neighbors:
            start = next(k for k, v in neighbors.items() if len(v) == 1)
            path, previous, node = [], -1, start
            while True:
                path.append(node)
                nxt = [k for k in neighbors.pop(node) if k != previous]
                if not nxt:
                    break
                previous, node = node, nxt[0]
            path = np.array(path)
            xx, yy = self.xv.ravel()[path], self.yv.ravel()[path]
            distance = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
            self.outlet_paths.append((path, distance / distance[-1]))


class _Momentum:
    def __init__(self, nf, dtype):
        self.nf, self.m = nf, 0
        self.S, self.A, self.H, self.K = [([], [], []) for _ in range(4)]
        self.h, self.bd = [], np.zeros(nf, dtype=dtype)
        self.links, self.walls, self.exterior = [], [], []

    def link(self, left, right, flux_terms, conductance):
        left, right = np.ravel(left), np.ravel(right)
        ids = self.m + np.arange(len(left))
        self.m += len(ids)
        _put(self.S, left, ids, 1.)
        _put(self.S, right, ids, -1.)
        _put(self.H, ids, left, .5)
        _put(self.H, ids, right, .5)
        self.h.append(np.zeros(len(ids)))
        for face, weight in flux_terms:
            _put(self.A, ids, face, weight)
        for r, c, sign in ((left, left, 1), (right, right, 1),
                            (left, right, -1), (right, left, -1)):
            _put(self.K, r, c, sign * conductance)
        self.links.append((left, right, conductance))

    def boundary(self, row, face, flow_weight, trace, constant, conductance=0., wall=0.):
        row, face = np.ravel(row), np.ravel(face)
        ids = self.m + np.arange(len(row))
        self.m += len(ids)
        _put(self.S, row, ids, 1.)
        _put(self.A, ids, face, flow_weight)
        trace = np.broadcast_to(trace, row.shape)
        selected = trace >= 0
        _put(self.H, ids[selected], trace[selected], 1.)
        self.h.append(np.broadcast_to(constant, row.shape))
        conductance = np.broadcast_to(conductance, row.shape)
        _put(self.K, row, row, conductance)
        np.add.at(self.bd, row, conductance * wall)
        self.walls.append((row, conductance, np.broadcast_to(wall, row.shape)))
        self.exterior.append((ids, row))

    def build(self):
        self.S = _matrix(*self.S, (self.nf, self.m))
        self.A = _matrix(*self.A, (self.m, self.nf))
        self.H = _matrix(*self.H, (self.m, self.nf))
        self.K = _matrix(*self.K, (self.nf, self.nf))
        self.h = np.concatenate(self.h)
        return self

    def conv(self, Q, w):
        return self.S @ ((self.A @ Q) * (self.H @ w + self.h))


class OpenMAC:
    """Generic core. Callbacks use (coordinates, tag, mu), geometry uses mu.

    geometry(mu) -> x,y; coefficients(mu) -> nu,kappa;
    psi_bc(x_reference,y_reference,mu) -> streamfunction;
    velocity(x,y,tag,mu) -> u,v; thermal(x,y,tag,mu) -> kind,value;
    traction(x,y,mu) -> tx,ty; force(x,y,mu) -> fx,fy; source(x,y,mu) -> sT.
    A Neumann thermal value is kappa*partial_n T, positive into the fluid.
    """

    def __init__(self, grid, geometry, coefficients, psi_bc, mu_reference,
                 velocity=None, thermal=None, traction=None, force=None,
                 source=None, scheme='luds', backend='gpu'):
        self.grid, self.geometry, self.coefficients = grid, geometry, coefficients
        self.psi_bc, self.thermal_bc = psi_bc, thermal
        self.velocity_bc = velocity or (lambda x, y, tag, mu: (0., 0.))
        self.traction = traction or (lambda x, y, mu: (0., 0.))
        self.force = force or (lambda x, y, mu: (0., 0.))
        self.source = source or (lambda x, y, mu: 0.)
        self.scheme, self.backend, self._gpu = scheme, backend, {}
        self._mu, self._op = None, None
        ref = self._mesh(grid.x, grid.y)
        self.Chat = ref.C @ grid.E
        self.Muhat, self.Mphat = ref.Mu, ref.Mp
        self.M = (self.Chat.T @ sp.diags(ref.Mu) @ self.Chat).tocsr()
        self._Mlu = sla.splu(self.M.tocsc())
        laplace = (ref.C.T @ sp.diags(ref.Mu) @ ref.C).tocsr()
        self._lift_boundary = laplace[grid.inside]
        self._lift_lu = sla.splu(laplace[grid.inside][:, grid.inside].tocsc()) if len(grid.inside) else None
        self.ell = self.lift(mu_reference)

    def _lu(self, name, A, cols=1):
        """Solve function of a factorization of A: cuDSS (one planned solver per name) or SciPy SuperLU."""
        if self.backend == 'cpu':
            return sla.splu(sp.csc_matrix(A), permc_spec='MMD_AT_PLUS_A').solve
        return self._gpu.setdefault(name, _Factor()).factor(A, cols)

    def _mesh(self, x, y):
        g = self.grid
        x, y = np.asarray(x), np.asarray(y)
        dx, dy = np.diff(x), np.diff(y)
        xc, yc = .5 * (x[1:] + x[:-1]), .5 * (y[1:] + y[:-1])
        area = np.r_[dy[g.fj[g.axis == 0]], dx[g.fi[g.axis == 1]]]
        mp = dx[g.ci] * dy[g.cj]
        mu = .5 * (np.where(g.p >= 0, mp[np.maximum(g.p, 0)], 0.)
                   + np.where(g.n >= 0, mp[np.maximum(g.n, 0)], 0.))
        C = sp.diags(1 / area) @ g.K
        Dvol = g.Df @ sp.diags(area)
        return SimpleNamespace(x=x, y=y, dx=dx, dy=dy, xc=xc, yc=yc, area=area,
                               Mp=mp, Mu=mu, C=C, Cv=C @ g.E,
                               DU=Dvol[:, g.U].tocsr(), Dvol=Dvol)

    def prescribed(self, mu):
        g = self.grid
        values = np.zeros(g.E.shape[0], dtype=np.result_type(mu, float))
        values[g.fixed] = self.psi_bc(g.xv.ravel()[g.fixed], g.yv.ravel()[g.fixed], mu)
        return values

    def lift(self, mu):
        g = self.grid
        full = self.prescribed(mu)
        for path, t in g.outlet_paths:
            full[path] = (1 - t) * full[path[0]] + t * full[path[-1]]
        if self._lift_lu is not None:
            full[g.inside] = self._lift_lu.solve(-(self._lift_boundary @ full))
        return full[g.free]

    def full(self, state, mu):
        return self.prescribed(mu) + self.grid.E @ state

    def admissible(self, raw):
        return np.asarray(raw).reshape(self.grid.E.shape[0], -1)[self.grid.free].squeeze()

    def whiten(self, state):
        out = self.Chat @ state
        return (np.sqrt(self.Muhat) if out.ndim == 1 else np.sqrt(self.Muhat)[:, None]) * out

    def operators(self, mu):
        mu = np.asarray(mu)
        if np.iscomplexobj(mu):
            return self._assemble(mu)
        key = tuple(mu)
        if key != self._mu:
            self._op, self._mu = self._assemble(mu), key
        return self._op

    def _assemble(self, mu):
        g = self.grid
        op = self._mesh(*self.geometry(mu))
        op.nu, op.kappa = self.coefficients(mu)
        dtype = np.result_type(op.x, op.y, op.nu, op.kappa, mu, float)
        mo = _Momentum(g.nf, dtype)
        i, j = g.ci, g.cj
        l, r = g.uid[i, j], g.uid[i + 1, j]
        mo.link(l, r, [(l, .5), (r, .5)], op.dy[j] / op.dx[i])
        l, r = g.vid[i, j], g.vid[i, j + 1]
        mo.link(l, r, [(l, .5), (r, .5)], op.dx[i] / op.dy[j])
        op.load = np.zeros(g.nf, dtype=dtype)
        for axis in (0, 1):
            faces = np.flatnonzero((g.axis == axis) & ~g.boundary)
            i, j = g.fi[faces], g.fj[faces]
            for endpoint in (0, 1):
                if axis == 0:
                    l, r = g.vid[i - 1, j + endpoint], g.vid[i, j + endpoint]
                    area, distance = .5 * op.dy[j], op.xc[i] - op.xc[i - 1]
                else:
                    l, r = g.uid[i + endpoint, j - 1], g.uid[i + endpoint, j]
                    area, distance = .5 * op.dx[i], op.yc[j] - op.yc[j - 1]
                mo.link(l, r, [(faces, .5)], area / distance)
        boundary = np.flatnonzero(g.boundary)
        mo.boundary(boundary, boundary, g.sign[boundary], boundary, 0.)
        for tag in np.unique(g.tags[boundary]):
            for axis in (0, 1):
                faces = np.flatnonzero(g.boundary & (g.tags == tag) & (g.axis == axis))
                i, j, sign = g.fi[faces], g.fj[faces], g.sign[faces]
                cell = np.maximum(g.p[faces], g.n[faces])
                ci, cj = g.ci[cell], g.cj[cell]
                xb = op.x[i] if axis == 0 else op.xc[i]
                yb = op.yc[j] if axis == 0 else op.y[j]
                tr = self.traction(xb, yb, mu)
                if tag == 'outlet':
                    op.load[faces] += op.area[faces] * np.broadcast_to(tr[axis], faces.shape)
                for endpoint in (0, 1):
                    if axis == 0:
                        row = g.vid[ci, cj + endpoint]
                        area, distance = .5 * op.dy[cj], sign * (op.x[i] - op.xc[ci])
                        xt, yt = xb, .5 * (op.y[cj + endpoint] + op.yc[cj])
                    else:
                        row = g.uid[ci + endpoint, cj]
                        area, distance = .5 * op.dx[ci], sign * (op.y[j] - op.yc[cj])
                        xt, yt = .5 * (op.x[ci + endpoint] + op.xc[ci]), yb
                    if tag == 'outlet':
                        tangential = np.broadcast_to(self.traction(xt, yt, mu)[1 - axis], faces.shape)
                        mo.boundary(row, faces, .5 * sign, row, distance * tangential / op.nu)
                        np.add.at(op.load, row, area * tangential)
                    else:
                        value = np.broadcast_to(self.velocity_bc(xt, yt, tag, mu)[1 - axis], faces.shape)
                        mo.boundary(row, faces, .5 * sign, -1, value, area / distance, value)
        op.mo = mo.build()
        op.B = op.Cv[g.U].tocsr()
        op.Gweak = -op.DU.T.tocsr()
        fx = np.r_[op.x[g.fi[g.axis == 0]], op.xc[g.fi[g.axis == 1]]]
        fy = np.r_[op.yc[g.fj[g.axis == 0]], op.y[g.fj[g.axis == 1]]]
        force = self.force(fx, fy, mu)
        op.load += op.Mu * np.where(g.axis == 0, np.broadcast_to(force[0], fx.shape),
                                   np.broadcast_to(force[1], fy.shape))
        return op

    def fields(self, state, mu):
        op = self.operators(mu)
        psi = self.full(state, mu)
        return op, self.grid.K @ psi, op.C @ psi

    def flow_residual(self, state, mu, convection=1.):
        op, Q, w = self.fields(state, mu)
        R = convection * op.mo.conv(Q, w) + op.nu * (op.mo.K @ w - op.mo.bd) - op.load
        return op.Cv.T @ R

    def flow_jacobian(self, state, mu, convection=1.):
        op, Q, w = self.fields(state, mu)
        mo = op.mo
        J = convection * (mo.S @ sp.diags(mo.H @ w + mo.h) @ mo.A @ self.grid.K @ self.grid.E
                          + mo.S @ sp.diags(mo.A @ Q) @ mo.H @ op.Cv)
        return (op.Cv.T @ (J + op.nu * mo.K @ op.Cv)).tocsr()

    def _temperature(self, state, mu):
        g = self.grid
        op, Q, _ = self.fields(state, mu)
        dtype = np.result_type(Q, op.kappa, op.x, op.y, float)
        hs, ks = ([], [], []), ([], [], [])
        h0, bd, bn = np.zeros(g.nf, dtype=dtype), np.zeros(g.np, dtype=dtype), np.zeros(g.np, dtype=dtype)
        for axis in (0, 1):
            faces = np.flatnonzero((g.axis == axis) & ~g.boundary)
            i, j, p, n = g.fi[faces], g.fj[faces], g.p[faces], g.n[faces]
            coord, center = (op.x, op.xc) if axis == 0 else (op.y, op.yc)
            k = i if axis == 0 else j
            conductance = op.area[faces] / (center[k] - center[k - 1])
            for r, c, sign in ((p, p, 1), (n, n, 1), (p, n, -1), (n, p, -1)):
                _put(ks, r, c, sign * conductance)
            if self.scheme == 'cds':
                _put(hs, faces, p, .5)
                _put(hs, faces, n, .5)
                continue
            forward = Q[faces].real >= 0
            donor = np.where(forward, p, n)
            di, dj = g.ci[donor], g.cj[donor]
            si, sj = di.copy(), dj.copy()
            step = np.where(forward, -1, 1)
            if axis == 0:
                si += step
            else:
                sj += step
            padded = np.pad(g.tid, 1, constant_values=-1)
            second = padded[si + 1, sj + 1]
            inside = second >= 0
            dc = center[di if axis == 0 else dj]
            sc = center[np.clip(si if axis == 0 else sj, 0, len(center) - 1)]
            weight = np.ones(len(faces), dtype=dtype)
            weight[inside] = (coord[k[inside]] - sc[inside]) / (dc[inside] - sc[inside])
            _put(hs, faces[inside], second[inside], 1 - weight[inside])
            for local in np.flatnonzero(~inside):
                ci, cj, direction = di[local], dj[local], step[local]
                bf = (g.uid[ci + (direction > 0), cj] if axis == 0
                      else g.vid[ci, cj + (direction > 0)])
                xb = op.x[g.fi[bf]] if axis == 0 else op.xc[g.fi[bf]]
                yb = op.yc[g.fj[bf]] if axis == 0 else op.y[g.fj[bf]]
                kind, value = self.thermal_bc(xb, yb, g.tags[bf], mu)
                if kind == 'dirichlet':
                    xb_axis = xb if axis == 0 else yb
                    wd = (coord[k[local]] - xb_axis) / (dc[local] - xb_axis)
                    weight[local] = wd
                    h0[faces[local]] = (1 - wd) * value
            _put(hs, faces, donor, weight)
        boundary = np.flatnonzero(g.boundary)
        for tag in np.unique(g.tags[boundary]):
            for axis in (0, 1):
                faces = np.flatnonzero(g.boundary & (g.tags == tag) & (g.axis == axis))
                i, j, sign = g.fi[faces], g.fj[faces], g.sign[faces]
                cell = np.maximum(g.p[faces], g.n[faces])
                xb = op.x[i] if axis == 0 else op.xc[i]
                yb = op.yc[j] if axis == 0 else op.y[j]
                distance = (sign * (xb - op.xc[g.ci[cell]]) if axis == 0
                            else sign * (yb - op.yc[g.cj[cell]]))
                kind, value = self.thermal_bc(xb, yb, tag, mu)
                value = np.broadcast_to(value, faces.shape)
                inflow = (sign * Q[faces].real < 0) & (kind == 'dirichlet')
                h0[faces[inflow]] = value[inflow]
                _put(hs, faces[~inflow], cell[~inflow], 1.)
                if kind == 'dirichlet':
                    conductance = op.area[faces] / distance
                    _put(ks, cell, cell, conductance)
                    np.add.at(bd, cell, conductance * value)
                else:
                    np.add.at(bn, cell, op.area[faces] * value)
        H = _matrix(*hs, (g.nf, g.np))
        K = _matrix(*ks, (g.np, g.np))
        src = np.broadcast_to(self.source(op.xc[g.ci], op.yc[g.cj], mu), (g.np,)) * op.Mp
        A = (g.Df @ sp.diags(Q) @ H + op.kappa * K).tocsc()
        rhs = op.kappa * bd + bn + src - g.Df @ (Q * h0)
        return SimpleNamespace(A=A, rhs=rhs, H=H, h0=h0, K=K, bd=bd, bn=bn, source=src, Q=Q)

    def residual(self, state, T, mu):
        rt = None
        if T is not None:
            en = self._temperature(state, mu)
            rt = en.A @ T - en.rhs
        return self.flow_residual(state, mu), rt

    def jacobian(self, state, T, mu):
        jyy = self.flow_jacobian(state, mu)
        if T is None:
            return jyy, None, None, None
        en = self._temperature(state, mu)
        jty = self.grid.Df @ sp.diags(en.H @ T + en.h0) @ self.grid.K @ self.grid.E
        return jyy, sp.csr_matrix((self.grid.ny_state, self.grid.np)), jty.tocsr(), en.A

    def parameter_derivatives(self, state, T, mu):
        mu = np.asarray(mu, float)
        ry, rt = [], []
        for j in range(len(mu)):
            z = mu.astype(complex)
            z[j] += 1.e-30j
            a, b = self.residual(state, T, z)
            ry.append(a.imag / 1.e-30)
            if b is not None:
                rt.append(b.imag / 1.e-30)
        return np.column_stack(ry), (np.column_stack(rt) if rt else None)

    def temperature(self, state, mu, directions=None, parameters=False):
        """T, optional state directions, and optional fixed-state parameter partials.

        With parameters=True return (T, SyV or None, Smu), sharing one factorization.
        """
        en = self._temperature(state, mu)
        cols = max(1, 0 if directions is None else directions.shape[1], len(mu) if parameters else 0)
        lu = self._lu('temperature', en.A, cols)
        T = lu(en.rhs)
        derivative = None
        if directions is not None:
            dQ = self.grid.K @ self.grid.E @ directions
            rhs = -(self.grid.Df @ ((en.H @ T + en.h0)[:, None] * dQ))
            derivative = lu(rhs)
        if parameters:
            partial = self.parameter_derivatives(state, T, mu)[1]
            return T, derivative, lu(-partial)
        return T if directions is None else (T, derivative)

    def sensitivity(self, state, T, mu):
        jyy, _, jty, att = self.jacobian(state, T, mu)
        rp, tp = self.parameter_derivatives(state, T, mu)
        sy = self._lu('flow', jyy, rp.shape[1])(-rp)
        st = None if T is None else self._lu('temperature', att, rp.shape[1])(-tp - jty @ sy)
        op = self.operators(mu)
        sw = op.Cv @ sy
        for j in range(len(mu)):
            z = np.array(mu, dtype=complex)
            z[j] += 1.e-30j
            sw[:, j] += self.fields(state, z)[2].imag / 1.e-30
        return sy, st, sw

    def solve(self, mu, state=None, tol=1.e-10, maxit=35, thermal=True, verbose=False):
        op = self.operators(mu)
        if state is None:
            zero = np.zeros(self.grid.ny_state)
            J = self.flow_jacobian(zero, mu, convection=0.)
            state = self._lu('flow', J)(-self.flow_residual(zero, mu, convection=0.))
        else:
            state = state.copy()
        norm = lambda r: float(np.sqrt(max(0., r @ self._Mlu.solve(r))))
        scale = max(1., norm(self.flow_residual(state, mu)))
        converged, backward, step_norm = False, 0., np.inf
        for it in range(maxit):
            r = self.flow_residual(state, mu)
            J = self.flow_jacobian(state, mu)
            delta = self._lu('flow', J)(-r)
            backward = max(backward, float(np.linalg.norm(J @ delta + r, np.inf)
                           / max(sla.norm(J, np.inf)*np.linalg.norm(delta, np.inf)
                                 + np.linalg.norm(r, np.inf), np.finfo(float).tiny)))
            step_norm = np.linalg.norm(self.whiten(delta)) / max(1., np.linalg.norm(self.whiten(state - self.lift(mu))))
            res = norm(r) / scale
            if verbose:
                print(f'  {it:2d} residual={res:.3e} correction={step_norm:.3e}', flush=True)
            if res <= tol and step_norm <= tol:
                converged = True
                break
            alpha, old = 1., norm(r)
            for _ in range(22):
                candidate = state + alpha * delta
                if norm(self.flow_residual(candidate, mu)) <= (1 - 1.e-4 * alpha) * old:
                    state = candidate
                    break
                alpha *= .5
            else:
                break
        final = self.flow_residual(state, mu)
        status = dict(converged=converged, iterations=it + 1, residual=norm(final) / scale,
                      residual_l2=float(np.linalg.norm(final)), correction=float(step_norm),
                      linear_backward_error=backward)
        T = self.temperature(state, mu) if converged and thermal and self.thermal_bc is not None else None
        return state, T, status

    def continue_to(self, mu_start, mu_target, state=None, log_coordinates=(0,),
                    step=.25, min_step=1.e-4, **solve_options):
        """Adaptive continuation between two parameter points on fixed topology."""
        start, target = np.array(mu_start, float), np.array(mu_target, float)
        a, b = start.copy(), target.copy()
        for j in log_coordinates:
            a[j], b[j] = np.log(a[j]), np.log(b[j])
        state, T, status = self.solve(start, state=state, **solve_options)
        history = [dict(mu=start.tolist(), fraction=0., **status)]
        if not status['converged']:
            return state, T, history
        fraction = 0.
        while fraction < 1.:
            trial = min(1., fraction+step)
            mu = a + trial*(b-a)
            for j in log_coordinates:
                mu[j] = np.exp(mu[j])
            candidate, temp, status = self.solve(mu, state=state, **solve_options)
            history.append(dict(mu=mu.tolist(), fraction=trial, **status))
            if status['converged']:
                state, T, fraction = candidate, temp, trial
                step = min(.5, 1.5*step)
            else:
                step *= .5
                if step < min_step:
                    break
        return state, T, history

    def pressure(self, state, mu):
        op, Q, w = self.fields(state, mu)
        R = (op.mo.conv(Q, w) + op.nu * (op.mo.K @ w - op.mo.bd) - op.load)[self.grid.U]
        p = sla.spsolve((op.DU @ op.DU.T).tocsc(), op.DU @ R)
        return p, float(np.linalg.norm(R - op.DU.T @ p) / max(1., np.linalg.norm(R)))

    def balances(self, state, T, mu):
        g = self.grid
        op, Q, w = self.fields(state, mu)
        mo = op.mo
        F, trace = mo.A @ Q, mo.H @ w + mo.h
        kinetic, boundary_correction = 0., 0.
        for ids, rows in mo.exterior:
            kinetic += .5 * np.sum(F[ids] * trace[ids]**2)
            boundary_correction -= .5 * np.sum(F[ids] * (w[rows] - trace[ids])**2)
        convection = float(w @ mo.conv(Q, w))
        dissipation = sum(np.sum(c * (w[l] - w[r])**2) for l, r, c in mo.links)
        dissipation += sum(np.sum(c * (w[r] - v)**2) for r, c, v in mo.walls)
        wall_work = sum(np.sum(c * (w[r] - v) * v) for r, c, v in mo.walls)
        viscous = w @ (mo.K @ w - mo.bd)
        out = dict(kinetic_flux=float(kinetic), convection_work=convection,
                   boundary_convection_correction=float(boundary_correction),
                   convection_defect=float(convection - kinetic - boundary_correction),
                   viscous_dissipation=float(op.nu * dissipation),
                   viscous_defect=float(op.nu * (viscous - dissipation - wall_work)),
                   dual_volume_flux_defect=float(np.max(np.abs(mo.S @ F))))
        if T is not None:
            en = self._temperature(state, mu)
            adv = g.Df @ (Q * (en.H @ T + en.h0))
            diffusion = op.kappa * (en.K @ T - en.bd) - en.bn
            out.update(heat_advection=float(adv.sum()), heat_diffusion=float(diffusion.sum()),
                       heat_source=float(en.source.sum()),
                       heat_defect=float((adv + diffusion - en.source).sum()),
                       temperature_min=float(T.min()), temperature_max=float(T.max()))
        return out

    def faces(self, state, mu):
        g = self.grid
        _, _, w = self.fields(state, mu)
        u, v = np.full(g.uid.shape, np.nan), np.full(g.vid.shape, np.nan)
        u[g.uid >= 0] = w[g.uid[g.uid >= 0]]
        v[g.vid >= 0] = w[g.vid[g.vid >= 0]]
        return u, v

    def bulk(self, state, T, mu):
        g = self.grid
        op, Q, _ = self.fields(state, mu)
        flow = .5 * (Q[g.uid[g.ci, g.cj]] + Q[g.uid[g.ci + 1, g.cj]])
        numerator, denominator = np.zeros(g.nx), np.zeros(g.nx)
        np.add.at(numerator, g.ci, flow * T)
        np.add.at(denominator, g.ci, flow)
        return op.xc, numerator / denominator

    def wall_fields(self, state, T, mu, tag):
        g = self.grid
        op, _, _ = self.fields(state, mu)
        faces = np.flatnonzero(g.boundary & (g.tags == tag))
        cell = np.maximum(g.p[faces], g.n[faces])
        ci, cj, axis, sign = g.ci[cell], g.cj[cell], g.axis[faces], g.sign[faces]
        si, sj = ci - (axis == 0) * sign, cj - (axis == 1) * sign
        second = g.tid[si, sj]
        xb = np.where(axis == 0, op.x[g.fi[faces]], op.xc[np.minimum(g.fi[faces], g.nx - 1)])
        yb = np.where(axis == 1, op.y[g.fj[faces]], op.yc[np.minimum(g.fj[faces], g.ny - 1)])
        d0 = np.where(axis == 0, sign * (xb - op.xc[ci]), sign * (yb - op.yc[cj]))
        d1 = np.where(axis == 0, sign * (xb - op.xc[si]), sign * (yb - op.yc[sj]))
        kind, value = self.thermal_bc(xb, yb, tag, mu)
        if kind == 'dirichlet':
            tw = np.broadcast_to(value, faces.shape)
            grad = -((T[cell] - tw) * d1**2 - (T[second] - tw) * d0**2) / (d0 * d1 * (d1 - d0))
        else:
            grad = np.broadcast_to(value, faces.shape) / op.kappa
            curvature = (T[second] - T[cell] + grad * (d1 - d0)) / (d1**2 - d0**2)
            tw = T[cell] + grad * d0 - curvature * d0**2
        return dict(faces=faces, x=xb, y=yb, area=op.area[faces], Tw=tw, gradient=grad)

    def nusselt(self, state, T, mu, tag, length, reference):
        wall = self.wall_fields(state, T, mu, tag)
        difference = wall['Tw'] - np.broadcast_to(reference, wall['Tw'].shape)
        nu = np.divide(length*wall['gradient'], difference,
                       out=np.full(difference.shape, np.nan), where=difference!=0)
        return {**wall, 'Nu': nu, 'average': np.sum(nu*wall['area'])/np.sum(wall['area'])}

    def reattachment(self, state, mu, tag='heated'):
        """All lower-wall negative-to-positive crossings, x coordinates and shear."""
        data = self.recirculation(state, mu, tag)
        return data['reattachment'], data['x'], data['shear']

    def recirculation(self, state, mu, tag='heated'):
        """Horizontal-wall crossings; identify the strongest reverse-shear region."""
        g = self.grid
        op = self.operators(mu)
        faces = np.flatnonzero(g.boundary & (g.tags == tag) & (g.axis == 1))
        cell = np.maximum(g.p[faces], g.n[faces])
        ci, cj = g.ci[cell], g.cj[cell]
        sign = g.sign[faces]
        second_j = cj - sign
        u, _ = self.faces(state, mu)
        u0 = .5 * (u[ci, cj] + u[ci + 1, cj])
        u1 = .5 * (u[ci, second_j] + u[ci + 1, second_j])
        a = -sign * (op.yc[cj] - op.y[g.fj[faces]])
        b = -sign * (op.yc[second_j] - op.y[g.fj[faces]])
        shear = op.nu * (u0 * b**2 - u1 * a**2) / (a * b * (b - a))
        x = op.xc[ci]
        crossings = np.flatnonzero((shear[:-1] < 0) & (shear[1:] >= 0))
        xr = x[crossings] - shear[crossings] * np.diff(x)[crossings] / np.diff(shear)[crossings]
        splits = np.flatnonzero((shear[:-1] > 0) & (shear[1:] <= 0))
        xs = x[splits] - shear[splits] * np.diff(x)[splits] / np.diff(shear)[splits]
        primary = xr[xr > x[np.argmin(shear)]] if np.min(shear) < 0 else np.array([])
        return dict(x=x, shear=shear, separation=xs, reattachment=xr,
                    primary_reattachment=float(primary[0]) if len(primary) else None)


def condition_estimate(A, metric_diagonal=None, backend='gpu'):
    """One-norm condition estimate; optional symmetric diagonal coordinate scaling."""
    s = np.ones(A.shape[0]) if metric_diagonal is None else 1 / np.sqrt(metric_diagonal)
    A = (sp.diags(s) @ A @ sp.diags(s)).tocsc()
    if backend == 'cpu':
        lu = sla.splu(A, permc_spec='MMD_AT_PLUS_A')
        solve, tsolve = lu.solve, lambda v: lu.solve(v, trans='T')
    else:
        solve, tsolve = _Factor().factor(A), _Factor().factor(A.T)
    inverse = sla.LinearOperator(A.shape, matvec=solve, rmatvec=tsolve)
    return float(sla.norm(A, 1) * sla.onenormest(inverse))
