"""MAC 离散上的降阶闭合运算，全部在 GPU 上（torch float64，cuDSS）。

状态为内部角点流函数 ψ，基方向 Ψ_B（列）；W = C Ψ_B 为对应的面速度。
  g = Ψ_Bᵀ r_ψ = Wᵀ R_w(w, T)
  G = Ψ_Bᵀ J_eff Ψ_B = Wᵀ [ (∂R_w/∂w) W + (∂R_w/∂T) S_W ]
有温度时（cav.thermal）温度为辅助场 T = S(ψ)，S_W = −J_TT⁻¹ (∂R_T/∂w) W；无温度时没有 T 与 S_W 两项。
∂R_T/∂T = J_TT = S_e diag(A_e w) H_e + κ K_e 的非零元对面通量 A_e w 线性：values = G_T (A_e w) + κ k₀，G_T 预先算好。
J_TT 的结构固定，cuDSS 符号分析只做一次；每次 reduced 只分解一次，温度与 k 列灵敏度共用该分解，右端项列数即基方向数。
与状态无关的 C Ψ_B、A W、H W、K W、A_e W 每个基只算一次（以 Ψ_B 对象判定是否为同一个基）。
"""
import nvmath
import numpy as np
import scipy.sparse as sp
import torch
from nvmath.bindings import cudss

from solvers.mac.mac import _keys

DEV = torch.device("cuda")
_I32, _F64 = nvmath.CudaDataType.CUDA_R_32I, nvmath.CudaDataType.CUDA_R_64F


def _t(A, idx=torch.int64):
    A = A.tocsr()
    A.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=idx), torch.as_tensor(A.indices, dtype=idx),
                                   torch.as_tensor(A.data, dtype=torch.float64), size=A.shape, device=DEV)


def _g(a):
    return torch.as_tensor(a, dtype=torch.float64, device=DEV)


class _Direct:
    """GPU 上 CSR 矩阵 J（int32 下标，数值原位更新）的 cuDSS 直接法：符号分析在建立时做一次，
    factorize 按 J 当前的数值分解，solve 的右端项列数任意。右端项与解为 (k, n) 张量（即 n×k 列主序）。"""

    def __init__(self, J):
        self.n = n = J.shape[0]
        self.h, self.cfg = cudss.create(), cudss.config_create()
        self.dat = cudss.data_create(self.h)
        cudss.set_stream(self.h, torch.cuda.current_stream().cuda_stream)
        self.a = cudss.matrix_create_csr(n, n, J.values().numel(), J.crow_indices().data_ptr(), 0, J.col_indices().data_ptr(),
                                         J.values().data_ptr(), _I32, _I32, _F64, cudss.MatrixType.GENERAL,
                                         cudss.MatrixViewType.FULL, cudss.IndexBase.ZERO)
        self.z = torch.zeros(1, n, dtype=torch.float64, device=DEV)
        self._run(cudss.Phase.ANALYSIS, self.z, self.z)

    def _run(self, phase, X, B):
        x, b = (cudss.matrix_create_dn(self.n, M.shape[0], self.n, M.data_ptr(), _F64, cudss.Layout.COL_MAJOR) for M in (X, B))
        cudss.execute(self.h, phase, self.cfg, self.dat, self.a, x, b)
        cudss.matrix_destroy(x)
        cudss.matrix_destroy(b)

    def factorize(self):
        self._run(cudss.Phase.FACTORIZATION, self.z, self.z)

    def solve(self, B):
        B = B.contiguous()
        X = torch.empty_like(B)
        self._run(cudss.Phase.SOLVE, X, B)
        return X


class Closure:
    """cav：mac.Cavity。"""

    def __init__(self, cav):
        mo = cav.mo
        self.thermal = cav.thermal
        self.inertia = cav.inertia
        self.C, self.Sm, self.Am, self.Hm, self.Km = (_t(M) for M in (cav.C, mo.S, mo.A, mo.H, mo.K))
        self.bm, self.sqMu = _g(mo.b), _g(np.sqrt(cav.Mu))
        self.Mu = _g(cav.Mu)
        self._PsiB = None
        if not self.thermal:
            return
        en = cav.en
        self.Bt, self.Se, self.Ae, self.He = (_t(M) for M in (cav.Bt, en.S, en.A, en.H))
        self.be = _g(en.b)
        # J_TT 的模板结构与 G_T：S_e 的每个元 (i, f, s) 与 H_e 第 f 行的每个元 (j, h) 给出 (i, j) 处的系数 s·h·F_f
        tpl = (en.S @ sp.diags(np.random.default_rng(0).standard_normal(en.m)) @ en.H + en.K).tocsr()
        key = _keys(tpl)
        Sc, H = en.S.tocoo(), en.H.tocsr()
        H.sort_indices()
        cnt = np.diff(H.indptr)[Sc.col]
        e = np.repeat(np.arange(Sc.nnz), cnt)
        k = H.indptr[Sc.col][e] + np.arange(len(e)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.searchsorted(key, Sc.row[e].astype(np.int64) * tpl.shape[1] + H.indices[k])
        self.GT = _t(sp.csr_matrix((Sc.data[e] * H.data[k], (pos, Sc.col[e])), shape=(tpl.nnz, en.m)))
        K = en.K.tocsr()
        k0 = np.zeros(tpl.nnz)
        k0[np.searchsorted(key, _keys(K))] = K.data
        self.k0 = _g(k0)
        self.J = _t(tpl, torch.int32)
        self.ds = _Direct(self.J)

    def whiten(self, Y):
        """L Y，L = M_u^{1/2} C，‖Y‖_M = ‖L Y‖₂。Y 为 (n,) 或 (n, k)。"""
        return (self.sqMu if Y.dim() == 1 else self.sqMu[:, None]) * (self.C @ Y)

    def _temperature(self, w, kap):
        """按 w 更新并分解 J_TT，返回温度。"""
        self.J.values().copy_(self.GT @ (self.Ae @ w) + kap * self.k0)
        self.ds.factorize()
        return self.ds.solve((kap * self.be)[None])[0]

    def temperature(self, psi, kap):
        return self._temperature(self.C @ psi, kap)

    def _basis(self, PsiB):
        if PsiB is not self._PsiB:
            W = self.C @ PsiB
            self._PsiB, self.W, self.AW, self.HW, self.KW = PsiB, W, self.Am @ W, self.Hm @ W, self.Km @ W
            if self.thermal:
                self.AeW = self.Ae @ W

    def reduced(self, psi, PsiB, nu, kap=None, *, buoyancy=1.0, drag=0.0):
        """约化残差 g (k,)、约化雅可比 G (k, k) 与温度 T（无温度时为 None）。"""
        self._basis(PsiB)
        W, w = self.W, self.C @ psi
        Fw, Hw = self.Am @ w, self.Hm @ w
        Rw = self.inertia * (self.Sm @ (Fw * Hw)) + nu * (self.Km @ w - self.bm)
        JW = self.inertia * (self.Sm @ (Hw[:, None] * self.AW + Fw[:, None] * self.HW)) + nu * self.KW
        if drag != 0:
            Rw = Rw + drag * self.Mu * w
            JW = JW + drag * self.Mu[:, None] * W
        if not self.thermal:
            return W.T @ Rw, W.T @ JW, None
        T = self._temperature(w, kap)
        ST = self.ds.solve(-(self.Se @ ((self.He @ T)[:, None] * self.AeW)).T)
        return W.T @ (Rw - buoyancy * (self.Bt @ T)), W.T @ (JW - buoyancy * (self.Bt @ ST.T)), T
