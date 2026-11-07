"""Couette–Graetz 通道的单元中心有限体积求解器（solvers/fv/求解器说明.md）。

参考域 (0, 2)×(0, 1)：x̂ ∈ [0, 1] 为入口段，[1, 2] 为受热段，物理坐标 x = 1 + L(x̂ − 1)，y = ŷ。
单元编号 k = i·ny + j（x 方向在前），不随参数变化。离散方程为单元积分形式 A T = f，A = K(L)/Pe + C(L)。
K、C、f 对 L 的导数由组装公式对物理节点坐标解析求导（dx/dL = x̂ − 1）。
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
