"""MAC 交错网格稳态求解器（solvers/mac/求解器说明.md）。

单位方腔，张量积网格（节点 x、y）。ψ 在角点（边界为 0），u 在竖直面，v 在水平面，T 在单元中心。
数组下标为 [i, j]（x 方向在前），按 C 顺序展平。速度坐标 w = [u 内部面, v 内部面]，w = C ψ。
残差均为控制体积分形式（对偶量）：R_w 为面控制体上的动量残差（不含压力），r_ψ = Cᵀ R_w；R_T 为单元上的能量残差。
对流为保对称形式：面通量取相邻两面体积通量的平均，面上输运量取两侧算术平均；w 离散无散时对流矩阵反对称。
Newton 的线性方程用 cuDSS（nvmath，PINNGPU 环境）求解，符号分析每个网格只做一次。
"""
import logging

import numpy as np
import nvmath.sparse.advanced as nsa
import scipy.sparse as sp

_LOG = logging.getLogger("solvers.mac.cudss")
_LOG.setLevel(logging.ERROR)              # 不输出"未指定多线程库""重置 LHS 需重新分解"的提示


def _keys(A):
    A.sort_indices()
    return np.repeat(np.arange(A.shape[0], dtype=np.int64), np.diff(A.indptr)) * A.shape[1] + A.indices


class _Direct:
    """非零结构固定的稀疏方程组的 cuDSS 直接法（float64）。SciPy 乘积会丢掉数值为零的元素，故以 template 的结构为
    固定模板，每次把矩阵的值放进模板；符号分析只在建立时做一次。右端项列数上限 cols，不足的列补零。"""

    def __init__(self, template, cols):
        self.A, self.key = template, _keys(template)
        self.b = np.zeros((template.shape[0], cols), order="F")
        self.s = nsa.DirectSolver(self.A, self.b, options={"logger": _LOG})
        self.s.plan()

    def factor(self, M):
        M = M.tocsr()
        M.sort_indices()
        if M.nnz == self.A.nnz:                 # 结构包含于模板且非零数相同，即与模板结构相同
            self.A.data[:] = M.data
        else:
            self.A.data[:] = 0.0
            self.A.data[np.searchsorted(self.key, _keys(M))] = M.data
        self.s.reset_operands(a=self.A)
        self.s.factorize()

    def solve(self, B):
        B = B.reshape(len(B), -1)
        k = B.shape[1]
        self.b[:, :k] = B
        self.b[:, k:] = 0.0
        self.s.reset_operands(b=self.b)
        return self.s.solve()[:, :k]                   # 每次 solve 返回新数组，切片无需复制


def nodes(n, s=0.0):
    """[0, 1] 上 n 个单元的节点；s > 0 时按 tanh 向两壁加密。"""
    xi = np.linspace(0, 1, n + 1)
    return xi if s == 0 else 0.5 * (1 + np.tanh(s * (2 * xi - 1)) / np.tanh(s))


class _Faces:
    """一组控制体上的面。面通量 F = A w，面上输运量 φ_f = H φ，对流 = S (F ⊙ φ_f)；扩散 K φ − b。"""

    def __init__(self, n, nw):
        self.n, self.nw, self.m = n, nw, 0
        self.S, self.A, self.H, self.K = ([], [], []), ([], [], []), ([], [], []), ([], [], [])
        self.b = np.zeros(n)

    def add(self, L, R, flux, c, ub=0.0):
        """L、R：面两侧控制体编号（−1 为边界）；flux：[(w 编号, 系数), …]（w 编号 −1 为零）；c = 面积/距离；ub：边界值。"""
        L, R = np.ravel(L), np.ravel(R)
        k = len(L)
        f = self.m + np.arange(k)
        self.m += k
        c, ub = np.broadcast_to(np.ravel(c), (k,)), np.broadcast_to(np.ravel(ub), (k,))
        for idx, s in ((L, 1.0), (R, -1.0)):
            ok = idx >= 0
            self._put(self.S, idx[ok], f[ok], np.full(ok.sum(), s))
            self._put(self.H, f[ok], idx[ok], np.full(ok.sum(), 0.5))
        for wi, a in flux:
            wi, a = np.ravel(wi), np.broadcast_to(np.ravel(a), (k,))
            ok = wi >= 0
            self._put(self.A, f[ok], wi[ok], a[ok])
        both, onlyL, onlyR = (L >= 0) & (R >= 0), (L >= 0) & (R < 0), (L < 0) & (R >= 0)
        self._put(self.K, L[both], L[both], c[both])
        self._put(self.K, R[both], R[both], c[both])
        self._put(self.K, L[both], R[both], -c[both])
        self._put(self.K, R[both], L[both], -c[both])
        for idx, sel in ((L, onlyL), (R, onlyR)):
            self._put(self.K, idx[sel], idx[sel], c[sel])
            np.add.at(self.b, idx[sel], c[sel] * ub[sel])

    @staticmethod
    def _put(t, r, c, v):
        t[0].append(r), t[1].append(c), t[2].append(v)

    def build(self, nphi):
        mk = lambda t, shape: sp.csr_matrix((np.concatenate(t[2]), (np.concatenate(t[0]), np.concatenate(t[1]))), shape=shape)
        self.S, self.A = mk(self.S, (self.n, self.m)), mk(self.A, (self.m, self.nw))
        self.H, self.K = mk(self.H, (self.m, nphi)), mk(self.K, (self.n, nphi))
        return self

    def conv(self, w, phi):
        return self.S @ ((self.A @ w) * (self.H @ phi))

    def dconv(self, w, phi):
        """对流对 w、对 φ 的雅可比。"""
        return self.S @ sp.diags(self.H @ phi) @ self.A, self.S @ sp.diags(self.A @ w) @ self.H


class Cavity:
    """lid(x)：顶盖切向速度（None 为静止壁）；thermal：左壁 T = −1/2、右壁 +1/2、上下绝热，浮力 +T e_y。"""

    def __init__(self, x, y, lid=None, thermal=False):
        nx, ny = len(x) - 1, len(y) - 1
        dx, dy = np.diff(x), np.diff(y)
        xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
        self.x, self.y, self.xc, self.yc, self.dx, self.dy, self.thermal = x, y, xc, yc, dx, dy, thermal

        pid = -np.ones((nx + 1, ny + 1), int)
        pid[1:-1, 1:-1] = np.arange((nx - 1) * (ny - 1)).reshape(nx - 1, ny - 1)
        uid = -np.ones((nx + 1, ny), int)
        uid[1:-1] = np.arange((nx - 1) * ny).reshape(nx - 1, ny)
        nu_ = (nx - 1) * ny
        vid = -np.ones((nx, ny + 1), int)
        vid[:, 1:-1] = nu_ + np.arange(nx * (ny - 1)).reshape(nx, ny - 1)
        nw, npsi = nu_ + nx * (ny - 1), (nx - 1) * (ny - 1)
        tid = np.arange(nx * ny).reshape(nx, ny)
        self.pid, self.uid, self.vid, self.tid, self.nw, self.npsi, self.nT = pid, uid, vid, tid, nw, npsi, nx * ny
        I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")

        # 离散旋度 C：u_ij = (ψ_i,j+1 − ψ_ij)/dy_j，v_ij = −(ψ_i+1,j − ψ_ij)/dx_i
        r, c, v = [], [], []
        for rows, a, b, h, s in ((uid[1:-1], pid[1:-1, 1:], pid[1:-1, :-1], dy[None, :], 1.0),
                                 (vid[:, 1:-1], pid[1:, 1:-1], pid[:-1, 1:-1], dx[:, None], -1.0)):
            h = np.broadcast_to(h, rows.shape)
            for p, sg in ((a, s), (b, -s)):
                ok = p >= 0
                r.append(rows[ok]), c.append(p[ok]), v.append(sg / h[ok])
        self.C = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(nw, npsi))

        # 离散散度（体积通量形式，未除单元面积）D_vol：单元 = Σ 出流通量
        r, c, v = [], [], []
        for k, sg in ((1, 1.0), (0, -1.0)):
            uu = uid[k:nx + k]
            ok = uu >= 0
            r.append(tid[ok]), c.append(uu[ok]), v.append(sg * np.broadcast_to(dy[None, :], uu.shape)[ok])
            vv = vid[:, k:ny + k]
            ok = vv >= 0
            r.append(tid[ok]), c.append(vv[ok]), v.append(sg * np.broadcast_to(dx[:, None], vv.shape)[ok])
        self.Dvol = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(nx * ny, nw))

        # 控制体体积：u 面 (xc_i − xc_{i−1}) dy_j，v 面 dx_i (yc_j − yc_{j−1})
        hx, hy = np.diff(xc), np.diff(yc)
        self.Mu = np.concatenate([(hx[:, None] * dy[None, :]).ravel(), (dx[:, None] * hy[None, :]).ravel()])
        self.Mp = (dx[:, None] * dy[None, :]).ravel()

        # 动量面
        mo = _Faces(nw, nw)
        k, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")                   # u 控制体 x 向面，位于 xc_k
        mo.add(uid[k, j], uid[k + 1, j], [(uid[k, j], 0.5 * dy[j]), (uid[k + 1, j], 0.5 * dy[j])], dy[j] / dx[k])
        i, k = np.meshgrid(np.arange(1, nx), np.arange(ny + 1), indexing="ij")            # u 控制体 y 向面，位于 y_k
        L = np.where(k >= 1, uid[i, np.maximum(k - 1, 0)], -1)
        R = np.where(k <= ny - 1, uid[i, np.minimum(k, ny - 1)], -1)
        d = np.where(k == 0, yc[0], np.where(k == ny, 1 - yc[-1], np.concatenate([[1.0], hy, [1.0]])[k]))
        ub = np.where(k == ny, lid(x[i]) if lid is not None else 0.0, 0.0)
        mo.add(L, R, [(vid[i - 1, k], 0.5 * dx[i - 1]), (vid[i, k], 0.5 * dx[i])], hx[i - 1] / d, ub)
        i, k = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")                   # v 控制体 y 向面，位于 yc_k
        mo.add(vid[i, k], vid[i, k + 1], [(vid[i, k], 0.5 * dx[i]), (vid[i, k + 1], 0.5 * dx[i])], dx[i] / dy[k])
        k, j = np.meshgrid(np.arange(nx + 1), np.arange(1, ny), indexing="ij")            # v 控制体 x 向面，位于 x_k
        L = np.where(k >= 1, vid[np.maximum(k - 1, 0), j], -1)
        R = np.where(k <= nx - 1, vid[np.minimum(k, nx - 1), j], -1)
        d = np.where(k == 0, xc[0], np.where(k == nx, 1 - xc[-1], np.concatenate([[1.0], hx, [1.0]])[k]))
        mo.add(L, R, [(uid[k, j - 1], 0.5 * dy[j - 1]), (uid[k, j], 0.5 * dy[j])], hy[j - 1] / d)
        self.mo = mo.build(nw)

        if thermal:
            en = _Faces(nx * ny, nw)
            k, j = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")           # x 向面，位于 x_k
            L = np.where(k >= 1, tid[np.maximum(k - 1, 0), j], -1)
            R = np.where(k <= nx - 1, tid[np.minimum(k, nx - 1), j], -1)
            d = np.where(k == 0, xc[0], np.where(k == nx, 1 - xc[-1], np.concatenate([[1.0], hx, [1.0]])[k]))
            en.add(L, R, [(uid[k, j], dy[j])], dy[j] / d, np.where(k == 0, -0.5, 0.5))
            i, k = np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij")            # y 向内部面，位于 y_k
            en.add(tid[i, k - 1], tid[i, k], [(vid[i, k], dx[i])], dx[i] / hy[k - 1])
            self.en = en.build(nx * ny)
            # 浮力：v 控制体积分 T_face，T_face 为相邻两单元中心值的线性插值
            i, j = np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij")
            a = (yc[j] - y[j]) / hy[j - 1]
            vol = dx[i] * hy[j - 1]
            rows = vid[i, j].ravel()
            self.Bt = sp.csr_matrix((np.concatenate([(vol * a).ravel(), (vol * (1 - a)).ravel()]),
                                     (np.concatenate([rows, rows]), np.concatenate([tid[i, j - 1].ravel(), tid[i, j].ravel()]))),
                                    shape=(nw, nx * ny))
        self.Mpsi = (self.C.T @ sp.diags(self.Mu) @ self.C).tocsr()
        self.CKC = (self.C.T @ self.mo.K @ self.C).tocsr()                               # 雅可比中与状态无关的部分
        self.CBt = (-self.C.T @ self.Bt).tocsr() if thermal else None
        self._ds = self._ts = None

    # ---------------- 残差与雅可比 ----------------
    def res_w(self, w, T, nu):
        """面上积分动量残差（不含压力）。"""
        R = self.mo.conv(w, w) + nu * (self.mo.K @ w - self.mo.b)
        return R - self.Bt @ T if self.thermal else R

    def res_T(self, w, T, kap):
        return self.en.conv(w, T) + kap * (self.en.K @ T - self.en.b)

    def residual(self, psi, T, nu, kap=None):
        """(r_ψ, R_T)；无温度时 R_T 为 None。"""
        w = self.C @ psi
        return self.C.T @ self.res_w(w, T, nu), (self.res_T(w, T, kap) if self.thermal else None)

    def jacobian(self, psi, T, nu, kap=None):
        """块 (J_ψψ, J_ψT, J_Tψ, J_TT)，无温度时后三块为 None。"""
        w = self.C @ psi
        Jw, Jp = self.mo.dconv(w, w)
        Jpp = (self.C.T @ (Jw + Jp) @ self.C + nu * self.CKC).tocsr()
        if not self.thermal:
            return Jpp, None, None, None
        Ew, ET = self.en.dconv(w, T)
        return Jpp, self.CBt, (Ew @ self.C).tocsr(), (ET + kap * self.en.K).tocsr()

    def jac_full(self, psi, T, nu, kap=None):
        """完整雅可比（CSR），未知量顺序 [ψ, T]。"""
        Jpp, JpT, JTp, JTT = self.jacobian(psi, T, nu, kap)
        return (Jpp if not self.thermal else sp.bmat([[Jpp, JpT], [JTp, JTT]])).tocsr()

    def _lin(self, J, b):
        """Newton 线性方程；模板为随机状态下的完整雅可比。"""
        if self._ds is None:
            rng = np.random.default_rng(0)
            self._ds = _Direct(self.jac_full(rng.standard_normal(self.npsi),
                                             rng.standard_normal(self.nT) if self.thermal else None, 1.0, 1.0), 1)
        self._ds.factor(J)
        return self._ds.solve(b)[:, 0]

    def temperature(self, psi, kap, dw=None):
        """辅助场 T = S(ψ)：给定速度解线性能量方程。dw (nw, k) 为速度坐标下的方向（如 C Ψ_B，调用方算一次即可），
        给出时另返回灵敏度 −J_TT⁻¹ (∂R_T/∂w) dw (nT, k)。J_TT 每次只分解一次，温度与灵敏度共用；
        右端项列数上限取 32 的倍数，超过时重建求解器。"""
        en, w, k = self.en, self.C @ psi, 1 if dw is None else max(1, dw.shape[1])
        if self._ts is None or k > self._ts.b.shape[1]:
            rng = np.random.default_rng(0)
            tpl = en.S @ sp.diags(en.A @ rng.standard_normal(self.nw)) @ en.H + en.K
            self._ts = _Direct(tpl.tocsr(), -(-k // 32) * 32)
        self._ts.factor(en.S @ sp.diags(en.A @ w) @ en.H + kap * en.K)
        T = self._ts.solve(kap * en.b)[:, 0]
        if dw is None:
            return T
        return T, self._ts.solve(-((en.S @ sp.diags(en.H @ T) @ en.A) @ dw))

    # ---------------- 高保真求解 ----------------
    def solve(self, nu, kap=None, psi=None, T=None, tol=1e-8, maxit=30):
        """Newton（回溯线搜索）求 r_ψ = 0（与 R_T = 0）。返回 ψ、T 与状态。"""
        psi = np.zeros(self.npsi) if psi is None else psi.copy()
        if self.thermal and T is None:
            T = np.repeat(self.xc - 0.5, len(self.yc))
        n = self.npsi

        def F(q):
            rp, rT = self.residual(q[:n], q[n:] if self.thermal else None, nu, kap)
            return rp if rT is None else np.concatenate([rp, rT])

        q = psi if not self.thermal else np.concatenate([psi, T])
        f = F(q)
        for it in range(1, maxit + 1):
            dq = self._lin(self.jac_full(q[:n], q[n:] if self.thermal else None, nu, kap), -f)
            done = np.abs(dq).max() <= tol * max(1.0, np.abs(q).max())
            t, f0 = 1.0, np.linalg.norm(f)
            while not done:
                f1 = F(q + t * dq)
                if np.linalg.norm(f1) < f0 or t < 1e-3:
                    break
                t *= 0.5
            q = q + t * dq
            f = F(q) if done else f1
            if done:
                break
        st = {"iters": it, "converged": bool(done), "res": float(np.linalg.norm(f))}
        return q[:n], (q[n:] if self.thermal else None), st

    # ---------------- 输出量 ----------------
    def psi_full(self, psi):
        out = np.zeros(self.pid.shape)
        out[1:-1, 1:-1] = psi.reshape(len(self.x) - 2, len(self.y) - 2)
        return out

    def faces(self, psi):
        """u (nx+1, ny)、v (nx, ny+1)，含边界面。"""
        w = self.C @ psi
        nx, ny = len(self.dx), len(self.dy)
        u, v = np.zeros((nx + 1, ny)), np.zeros((nx, ny + 1))
        u[1:-1] = w[:(nx - 1) * ny].reshape(nx - 1, ny)
        v[:, 1:-1] = w[(nx - 1) * ny:].reshape(nx, ny - 1)
        return u, v

    def nusselt(self, T):
        """冷、热壁局部 Nu = ∂T/∂x（壁值与近壁两单元中心的二次插值）及沿 y 的平均。"""
        T = T.reshape(len(self.dx), len(self.dy))
        g = lambda tw, t0, t1, a, b: ((t0 - tw) * b ** 2 - (t1 - tw) * a ** 2) / (a * b * (b - a))
        nc = g(-0.5, T[0], T[1], self.xc[0], self.xc[1])
        nh = -g(0.5, T[-1], T[-2], 1 - self.xc[-1], 1 - self.xc[-2])
        return nc, nh, float(self.dy @ nc), float(self.dy @ nh)
