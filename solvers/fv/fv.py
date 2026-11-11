"""给定速度、分区导热通道的有限体积求解器（求解器说明.md）。

TransportChannel: 通用通道，A = K(ratio)/Pe + C，解析物理参数灵敏度。
Channel: Couette–Graetz 兼容配置，保留几何映射与对 η 的灵敏度。
单元编号 k = i·ny + j，残差为不除以单元面积的积分守恒形式。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LN10 = np.log(10.0)
# 竖直面上的线性插值格式：面 (i, i+1) 的模板单元相对 i 的偏移；单元 −1 为入口边界点 x = 0（T = 0）
SCHEMES = {"cds": (0, 1), "luds": (-1, 0), "quick": (-1, 0, 1)}


def stretch(n, s, side):
    """[0, 1] 上 n 个单元的节点，按 tanh 加密：side = 'lo' 向 0、'hi' 向 1、'both' 向两端；s = 0 为均匀。"""
    xi = np.linspace(0, 1, n + 1)
    if s == 0:
        return xi
    if side == "lo":
        return 1 + np.tanh(s * (xi - 1)) / np.tanh(s)
    if side == "hi":
        return np.tanh(s * xi) / np.tanh(s)
    return 0.5 * (1 + np.tanh(s * (2 * xi - 1)) / np.tanh(s))


def grid(n1, s1, n2, s2, ny, sy):
    """参考节点：入口段 n1 个单元向两端加密，受热段 n2 个单元向 x̂ = 1 加密，y 向 ny 个单元向底壁加密。"""
    xh = np.concatenate([stretch(n1, s1, "both"), 1 + stretch(n2, s2, "lo")[1:]])
    return xh, stretch(ny, sy, "lo")


def velocity_fluxes(x, y, velocity):
    """四点 Gauss 积分给定无散速度 velocity(X,Y)->(u,v)，返回 Fx、Fy。"""
    x, y = np.asarray(x), np.asarray(y)
    z, w = np.polynomial.legendre.leggauss(4)
    xc, yc = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
    yy = yc[:, None] + np.diff(y)[:, None] * z / 2
    xx = xc[:, None] + np.diff(x)[:, None] * z / 2
    u, _ = velocity(x[:, None, None], yy[None, :, :])
    _, v = velocity(xx[:, None, :], y[None, :, None])
    fx = np.broadcast_to(u, (len(x), len(y) - 1, 4)) @ w * np.diff(y) / 2
    fy = np.broadcast_to(v, (len(x) - 1, len(y), 4)) @ w * np.diff(x)[:, None] / 2
    return fx, fy


def poiseuille_fluxes(x, y, lower=0.0, upper=1.0):
    """精确积分平面 Poiseuille 剖面 6z(1-z)，流体区间外速度为零。"""
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    z = np.clip((np.asarray(y) - lower) / (upper - lower), 0, 1)
    primitive = (upper - lower) * (3 * z**2 - 2 * z**3)
    return (np.broadcast_to(np.diff(primitive), (len(x), len(y) - 1)).copy(),
            np.zeros((len(x) - 1, len(y))))


class TransportChannel:
    """给定守恒面通量、分区系数与逐面边界的线性稳态传热。

    bc 的键为 left/right/bottom/top，值为 (类型,值)，可逐面给数组。
    neumann 值为输入导热通量 (k/Pe)∂nT。solid 单元系数为 ratio*conductivity。
    所有输入在构造时复制；改变速度、边界或几何须重新构造。
    """

    def __init__(self, x, y, fluxes=None, solid=None, conductivity=1.0,
                 bc=None, source=0.0, scheme="luds", reference=None):
        def nodes(a):
            a = np.array(a, dtype=float, copy=True)
            if a.ndim != 1 or len(a) < 3 or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0):
                raise ValueError("each coordinate needs at least two positive-width cells")
            return a

        self.x, self.y = nodes(x), nodes(y)
        self.dx, self.dy = np.diff(self.x), np.diff(self.y)
        self.xc, self.yc = (self.x[1:] + self.x[:-1]) / 2, (self.y[1:] + self.y[:-1]) / 2
        self.nx, self.ny = len(self.dx), len(self.dy)
        shape = self.nx, self.ny
        self.n = self.nx * self.ny
        self.tid = np.arange(self.n).reshape(shape)
        self.solid = np.broadcast_to(False if solid is None else solid, shape).astype(bool).copy()
        self.k0 = np.broadcast_to(conductivity, shape).astype(float).copy()
        self.source = np.broadcast_to(source, shape).astype(float).copy()
        if not np.all(np.isfinite(self.k0)) or np.any(self.k0 <= 0) or not np.all(np.isfinite(self.source)):
            raise ValueError("conductivity must be positive and source finite")
        if scheme not in ("cds", "luds", "upwind"):
            raise ValueError("scheme must be cds, luds or upwind")
        self.scheme = scheme
        xh, yh = (self.x, self.y) if reference is None else tuple(map(nodes, reference))
        if len(xh) != len(self.x) or len(yh) != len(self.y):
            raise ValueError("reference and physical grids must have the same topology")
        self.Mh = (np.diff(xh)[:, None] * np.diff(yh)).ravel()
        fx, fy = (np.zeros((self.nx + 1, self.ny)), np.zeros((self.nx, self.ny + 1))) if fluxes is None else fluxes
        self.fx, self.fy = np.array(fx, dtype=float, copy=True), np.array(fy, dtype=float, copy=True)
        if self.fx.shape != (self.nx + 1, self.ny) or self.fy.shape != (self.nx, self.ny + 1):
            raise ValueError("incorrect face flux shapes")
        if not np.all(np.isfinite(self.fx)) or not np.all(np.isfinite(self.fy)):
            raise ValueError("face fluxes must be finite")
        scale = max(np.max(np.abs(self.fx)), np.max(np.abs(self.fy)), 1e-30)
        div = np.diff(self.fx, axis=0) + np.diff(self.fy, axis=1)
        if np.max(np.abs(div)) > 1e-10 * scale:
            raise ValueError("face velocity must be discretely divergence-free")
        touching_solid = (np.abs(self.fx[:-1]) + np.abs(self.fx[1:])
                          + np.abs(self.fy[:, :-1]) + np.abs(self.fy[:, 1:]))[self.solid]
        if np.any(touching_solid > 1e-12 * scale):
            raise ValueError("all faces touching a solid cell must have zero advection")
        self._faces = []
        for ids, coord, center, width in ((self.tid, self.x, self.xc, self.dy),
                                          (self.tid.T, self.y, self.yc, self.dx)):
            p, q = ids[:-1].ravel(), ids[1:].ravel()
            dp = np.broadcast_to((coord[1:-1] - center[:-1])[:, None], ids[:-1].shape).ravel()
            dn = np.broadcast_to((center[1:] - coord[1:-1])[:, None], ids[:-1].shape).ravel()
            area = np.broadcast_to(width, ids[:-1].shape).ravel()
            self._faces.append((p, q, dp, dn, area))
        bc = {} if bc is None else bc
        if set(bc) - {"left", "right", "bottom", "top"}:
            raise ValueError("unknown boundary name")
        self._boundaries = {}
        specs = (("left", self.tid[0], self.xc[0] - self.x[0], self.dy, -self.fx[0]),
                 ("right", self.tid[-1], self.x[-1] - self.xc[-1], self.dy, self.fx[-1]),
                 ("bottom", self.tid[:, 0], self.yc[0] - self.y[0], self.dx, -self.fy[:, 0]),
                 ("top", self.tid[:, -1], self.y[-1] - self.yc[-1], self.dx, self.fy[:, -1]))
        for name, ids, distance, area, flow in specs:
            kind, value = bc.get(name, ("neumann", 0.0))
            kinds = np.broadcast_to(kind, ids.shape)
            value = np.broadcast_to(value, ids.shape).astype(float).copy()
            if np.any(~np.isin(kinds, ("dirichlet", "neumann"))) or not np.all(np.isfinite(value)):
                raise ValueError("invalid boundary condition")
            fixed = (kinds == "dirichlet").copy()
            if np.any((flow < 0) & ~fixed):
                raise ValueError("every advective inflow face needs a Dirichlet temperature")
            self._boundaries[name] = ids, distance, area, flow, fixed, value
        self.C, self._bc_adv = self._advection()
        self._f0 = (self.source * self.dx[:, None] * self.dy).ravel() + self._bc_adv
        for ids, _, area, _, fixed, value in self._boundaries.values():
            np.add.at(self._f0, ids, np.where(fixed, 0, area * value))
        # 固定构造数据，避免后续修改令预组装的对流与载荷失效。
        for a in (self.x, self.y, self.dx, self.dy, self.xc, self.yc, self.tid,
                  self.solid, self.k0, self.source, self.Mh, self.fx, self.fy):
            a.setflags(write=False)

    def _advection(self):
        rows, cols, vals = [], [], []
        rhs = np.zeros(self.n)

        def put(r, c, v):
            rows.append(np.ravel(r))
            cols.append(np.ravel(c))
            vals.append(np.broadcast_to(v, np.shape(r)).ravel())

        for ids, coord, center, flux, lo, hi in (
                (self.tid, self.x, self.xc, self.fx, "left", "right"),
                (self.tid.T, self.y, self.yc, self.fy.T, "bottom", "top")):
            p, q, flow = ids[:-1], ids[1:], flux[1:-1]
            if self.scheme == "cds":
                wp = ((center[1:] - coord[1:-1]) / np.diff(center))[:, None]
                for donor, weight in ((p, wp), (q, 1 - wp)):
                    put(p, donor, flow * weight)
                    put(q, donor, -flow * weight)
                continue
            i = np.broadcast_to(np.arange(len(center) - 1)[:, None], p.shape)
            donor_i = np.where(flow >= 0, i, i + 1)
            donor = np.take_along_axis(ids, donor_i, axis=0)
            weight = np.ones_like(flow)
            if self.scheme == "luds":
                second_i = np.where(flow >= 0, i - 1, i + 2)
                inside = (second_i >= 0) & (second_i < len(center))
                clipped = np.clip(second_i, 0, len(center) - 1)
                second = np.take_along_axis(ids, clipped, axis=0)
                bf0, bv0 = self._boundaries[lo][-2:]
                bf1, bv1 = self._boundaries[hi][-2:]
                boundary_fixed = np.where(second_i < 0, bf0, bf1)
                valid = np.where(inside, ~self.solid.ravel()[second], boundary_fixed)
                x2 = np.where(second_i < 0, coord[0],
                              np.where(second_i >= len(center), coord[-1], center[clipped]))
                wd = (coord[1:-1, None] - x2) / (center[donor_i] - x2)
                weight = np.where(valid, wd, 1.0)
                v2 = flow * (1 - weight)
                ok = inside & valid
                put(p[ok], second[ok], v2[ok])
                put(q[ok], second[ok], -v2[ok])
                known = np.where(second_i < 0, bv0, bv1)
                load = np.where(~inside & valid, v2 * known, 0)
                np.add.at(rhs, p.ravel(), -load.ravel())
                np.add.at(rhs, q.ravel(), load.ravel())
            put(p, donor, flow * weight)
            put(q, donor, -flow * weight)
        for ids, _, _, flow, fixed, value in self._boundaries.values():
            put(ids, ids, np.maximum(flow, 0))
            np.add.at(rhs, ids, -np.minimum(flow, 0) * value)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.n, self.n)), rhs

    def _diffusion(self, ratio):
        k = (self.k0 * np.where(self.solid, ratio, 1.0)).ravel()
        dk = (self.k0 * self.solid).ravel()
        rows, cols, vals, dvals = [], [], [], []
        bd, dbd = np.zeros(self.n), np.zeros(self.n)
        for p, q, dp, dn, area in self._faces:
            resistance = dp / k[p] + dn / k[q]
            a = area / resistance
            da = area / resistance**2 * (dp * dk[p] / k[p]**2 + dn * dk[q] / k[q]**2)
            for r, c, sign in ((p, p, 1), (q, q, 1), (p, q, -1), (q, p, -1)):
                rows.append(r), cols.append(c), vals.append(sign * a), dvals.append(sign * da)
        for ids, distance, area, _, fixed, value in self._boundaries.values():
            a = np.where(fixed, area * k[ids] / distance, 0)
            da = np.where(fixed, area * dk[ids] / distance, 0)
            rows.append(ids), cols.append(ids), vals.append(a), dvals.append(da)
            np.add.at(bd, ids, a * value)
            np.add.at(dbd, ids, da * value)
        rc = np.concatenate(rows), np.concatenate(cols)
        return (sp.csr_matrix((np.concatenate(vals), rc), shape=(self.n, self.n)),
                sp.csr_matrix((np.concatenate(dvals), rc), shape=(self.n, self.n)), bd, dbd)

    def matrix(self, Pe, ratio, derivatives=False):
        """A,f；derivatives=True 时另返回物理参数偏导字典。"""
        if not np.isfinite(Pe) or not np.isfinite(ratio) or Pe <= 0 or ratio <= 0:
            raise ValueError("Pe and ratio must be finite and positive")
        K, Kr, bd, bdr = self._diffusion(ratio)
        A, f = (K / Pe + self.C).tocsc(), bd / Pe + self._f0
        if derivatives:
            return A, f, {"Pe": (-K / Pe**2, -bd / Pe**2), "ratio": (Kr / Pe, bdr / Pe)}
        return A, f

    def residual(self, T, Pe, ratio):
        A, f = self.matrix(Pe, ratio)
        return A @ T - f

    def solve(self, Pe, ratio, sens=False):
        """返回 T，或 (T, ∂Pe T, ∂ratio T)，复用同一 LU。"""
        A, f, derivatives = self.matrix(Pe, ratio, derivatives=True)
        lu = spla.splu(A)
        T = lu.solve(f)
        if not sens:
            return T
        S = lu.solve(np.column_stack([fp - Ap @ T for Ap, fp in derivatives.values()]))
        return T, S[:, 0], S[:, 1]

    def face_fields(self, T, Pe, ratio):
        """返回共享面温和按正坐标方向的积分扩散通量；边界按指定条件重构。"""
        if Pe <= 0 or ratio <= 0 or not np.isfinite(Pe + ratio):
            raise ValueError("Pe and ratio must be finite and positive")
        T = np.asarray(T).reshape(self.n)
        k = (self.k0 * np.where(self.solid, ratio, 1.0)).ravel()
        tx, ty = np.empty_like(self.fx), np.empty_like(self.fy)
        qx, qy = np.empty_like(self.fx), np.empty_like(self.fy)
        for face, tf, qf in zip(self._faces, (tx[1:-1], ty.T[1:-1]), (qx[1:-1], qy.T[1:-1])):
            p, q, dp, dn, area = face
            g1, g2 = k[p] / dp, k[q] / dn
            tf[:] = ((g1 * T[p] + g2 * T[q]) / (g1 + g2)).reshape(tf.shape)
            qf[:] = (area / (dp / k[p] + dn / k[q]) / Pe * (T[p] - T[q])).reshape(qf.shape)
        for name, tf, qf, sign in (("left", tx[0], qx[0], -1), ("right", tx[-1], qx[-1], 1),
                                    ("bottom", ty[:, 0], qy[:, 0], -1), ("top", ty[:, -1], qy[:, -1], 1)):
            ids, d, area, _, fixed, value = self._boundaries[name]
            tf[:] = np.where(fixed, value, T[ids] + Pe * d / k[ids] * value)
            qf[:] = sign * np.where(fixed, k[ids] * area / (Pe * d) * (T[ids] - value), -area * value)
        return dict(tx=tx, ty=ty, qx=qx, qy=qy)

    def boundary_fluxes(self, T, Pe, ratio):
        fields = self.face_fields(T, Pe, ratio)
        diffusive = (-fields["qx"][0], fields["qx"][-1], -fields["qy"][:, 0], fields["qy"][:, -1])
        T = np.asarray(T).reshape(self.n)
        out = {}
        for (name, spec), diff in zip(self._boundaries.items(), diffusive):
            ids, _, _, flow, _, value = spec
            out[name] = dict(diffusion=diff, advection=flow * np.where(flow < 0, value, T[ids]))
        return out

    def balance(self, T, Pe, ratio):
        flux = self.boundary_fluxes(T, Pe, ratio)
        diff = sum(float(v["diffusion"].sum()) for v in flux.values())
        adv = sum(float(v["advection"].sum()) for v in flux.values())
        source = float(np.sum(self.source * self.dx[:, None] * self.dy))
        return dict(diffusion=diff, advection=adv, source=source, defect=diff + adv - source)


class Channel:
    def __init__(self, xh, y, scheme="cds"):
        self.xh, self.y, self.scheme = xh, y, scheme
        self.nx, self.ny = len(xh) - 1, len(y) - 1
        self.n = self.nx * self.ny
        self.i1 = int(np.argmin(np.abs(xh - 1)))               # x̂ = 1 的节点编号，受热单元为 i ≥ i1
        self.dxh, self.dy = np.diff(xh), np.diff(y)
        self.yc = 0.5 * (y[1:] + y[:-1])
        self.xhc = 0.5 * (xh[1:] + xh[:-1])
        self.Mh = (self.dxh[:, None] * self.dy[None, :]).ravel()  # 参考单元面积（度量）
        self.F = 0.5 * (y[1:] ** 2 - y[:-1] ** 2)                # 竖直面体积通量 ∫ y dy
        self.tid = np.arange(self.n).reshape(self.nx, self.ny)
        self._L = None

    def coords(self, L):
        """物理节点 x 及 dx/dL。"""
        d = np.where(self.xh > 1, self.xh - 1, 0.0)
        return np.where(self.xh > 1, 1 + L * d, self.xh), d

    def operators(self, L):
        """K、C、f 及其对 L 的导数 K_L、C_L、f_L。"""
        nx, ny, tid, dy, yc = self.nx, self.ny, self.tid, self.dy, self.yc
        x, dX = self.coords(L)
        dx, ddx = np.diff(x), np.diff(dX)
        xc, dxc = 0.5 * (x[1:] + x[:-1]), 0.5 * (dX[1:] + dX[:-1])
        r, c, v, dv = [], [], [], []

        def put(rr, cc, vv, dd):
            rr, cc = np.ravel(rr), np.ravel(cc)
            r.append(rr), c.append(cc)
            v.append(np.broadcast_to(np.ravel(vv), rr.shape)), dv.append(np.broadcast_to(np.ravel(dd), rr.shape))

        # 扩散：内部面两点通量，系数 = 面长 / 两侧中心距离
        h = xc[1:] - xc[:-1]
        dh = dxc[1:] - dxc[:-1]
        a = np.broadcast_to(dy[None, :] / h[:, None], (nx - 1, ny))
        da = np.broadcast_to(-dy[None, :] * dh[:, None] / h[:, None] ** 2, (nx - 1, ny))
        faces = [(tid[:-1], tid[1:], a, da)]
        k = yc[1:] - yc[:-1]
        faces.append((tid[:, :-1], tid[:, 1:], np.broadcast_to(dx[:, None] / k[None, :], (nx, ny - 1)),
                      np.broadcast_to(ddx[:, None] / k[None, :], (nx, ny - 1))))
        for P, N, a, da in faces:
            for rr, cc, s in ((P, P, 1), (N, N, 1), (P, N, -1), (N, P, -1)):
                put(rr, cc, s * a.ravel(), s * da.ravel())
        # Dirichlet：入口 x = 0、顶壁 y = 1，系数 = 面长 / 中心到边界距离
        put(tid[0], tid[0], dy / xc[0], -dy * dxc[0] / xc[0] ** 2)
        put(tid[:, -1], tid[:, -1], dx / (1 - yc[-1]), ddx / (1 - yc[-1]))
        R, Cc = np.concatenate(r), np.concatenate(c)
        K = sp.csr_matrix((np.concatenate(v), (R, Cc)), shape=(self.n, self.n))
        KL = sp.csr_matrix((np.concatenate(dv), (R, Cc)), shape=(self.n, self.n))

        # 对流：竖直面 (i, i+1) 上 T_f = Σ w_m T_m（模板点的 Lagrange 插值，入口边界点 x = 0 处 T = 0）
        off = SCHEMES[self.scheme]
        xp, dxp = np.concatenate([[0.0], xc]), np.concatenate([[0.0], dxc])  # 位置表，下标 +1；−1 → 边界点
        i = np.arange(nx - 1)
        xf, dxf = x[1:-1], dX[1:-1]
        r, c, v, dv = [], [], [], []
        for m in off:
            km = np.clip(i + m, -1, None)
            w, dlw = np.ones(nx - 1), np.zeros(nx - 1)
            for q in off:
                if q == m:
                    continue
                kq = np.clip(i + q, -1, None)
                num, den = xf - xp[kq + 1], xp[km + 1] - xp[kq + 1]
                w *= num / den
                dlw += (dxf - dxp[kq + 1]) / num - (dxp[km + 1] - dxp[kq + 1]) / den
            dw = w * dlw
            ok = km >= 0
            W, E, M = tid[i[ok]], tid[i[ok] + 1], tid[km[ok]]
            Fw = self.F[None, :] * w[ok, None]
            Fd = self.F[None, :] * dw[ok, None]
            put(W, M, Fw.ravel(), Fd.ravel())
            put(E, M, -Fw.ravel(), -Fd.ravel())
        put(tid[-1], tid[-1], self.F, 0.0)                     # 出口：对流通量取出口单元温度
        R, Cc = np.concatenate(r), np.concatenate(c)
        C = sp.csr_matrix((np.concatenate(v), (R, Cc)), shape=(self.n, self.n))
        CL = sp.csr_matrix((np.concatenate(dv), (R, Cc)), shape=(self.n, self.n))

        # 受热段底壁单位热流
        f, fL = np.zeros(self.n), np.zeros(self.n)
        f[tid[self.i1:, 0]], fL[tid[self.i1:, 0]] = dx[self.i1:], ddx[self.i1:]
        return K, C, f, KL, CL, fL

    def _ops(self, L):
        if self._L != L:
            self._L, self._op = L, self.operators(L)
        return self._op

    def matrix(self, L, Pe):
        K, C, f = self._ops(L)[:3]
        return (K / Pe + C).tocsc(), f

    def solve(self, L, Pe, sens=False):
        """T；sens 时另返回 ∂T/∂η₁、∂T/∂η₂（η₁ = 2 log10 L − 1，η₂ = (log10 Pe − 0.5)/1.5），复用 A 的 LU 分解。"""
        K, C, f, KL, CL, fL = self._ops(L)
        lu = spla.splu((K / Pe + C).tocsc())
        T = lu.solve(f)
        if not sens:
            return T
        c1, c2 = LN10 / 2 * L, 1.5 * LN10
        b = np.stack([c1 * (fL - (KL / Pe + CL) @ T), c2 / Pe * (K @ T)], 1)
        S = lu.solve(b)
        return T, S[:, 0], S[:, 1]

    # ---------- 后处理 ----------
    def wall(self, T, Pe):
        """受热段壁温 T_w = T_P + Pe Δy_P（各受热单元列）。"""
        return T.reshape(self.nx, self.ny)[self.i1:, 0] + Pe * self.yc[0]

    def tav(self, T, Pe):
        """受热段平均温度 T_av = ∫₁² T̂_w dx̂。"""
        return float(self.dxh[self.i1:] @ self.wall(T, Pe))

    def bulk(self, T):
        """混合平均温度 T_b = 2 ∫ y T dy（各单元列）。"""
        return 2 * T.reshape(self.nx, self.ny) @ self.F

    def nusselt(self, T, Pe):
        """受热段局部 Nu = Pe / (T_w − T_b)。"""
        return Pe / (self.wall(T, Pe) - self.bulk(T)[self.i1:])

    def balance(self, T, L, Pe):
        """全局热量收支：(Γ2 输入, 顶壁扩散, 入口扩散, 出口对流) 的出流量。"""
        x, _ = self.coords(L)
        dx = np.diff(x)
        Tm = T.reshape(self.nx, self.ny)
        top = (dx / (1 - self.yc[-1])) @ Tm[:, -1] / Pe
        inlet = (self.dy / (0.5 * x[1])) @ Tm[0] / Pe
        return float(dx[self.i1:].sum()), float(top), float(inlet), float(self.F @ Tm[-1])
