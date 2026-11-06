"""MAC 离散上的降阶闭合运算，全部在 GPU 上（torch float64，cuDSS）。

状态为内部角点流函数 ψ，基方向 Ψ_B（列）；W = C Ψ_B 为对应的面速度。
  g = Ψ_Bᵀ r_ψ = Wᵀ R_w(w, T)
  G = Ψ_Bᵀ J_eff Ψ_B = Wᵀ [ (∂R_w/∂w) W + (∂R_w/∂T) S_W ]
有温度时（cav.thermal）温度为辅助场 T = S(ψ)，S_W = −J_TT⁻¹ (∂R_T/∂w) W；无温度时没有 T 与 S_W 两项。
∂R_T/∂T = J_TT = S_e diag(A_e w) H_e + κ K_e 的非零元对面通量 A_e w 线性：values = G_T (A_e w) + κ k₀，G_T 预先算好，
J_TT 的结构固定，cuDSS 符号分析只做一次。温度（1 列）与灵敏度（cols 列）各用一个求解器。
"""
import numpy as np
import nvmath.sparse.advanced as nsa
import scipy.sparse as sp
import torch

from solvers.mac.mac import _LOG, _keys

DEV = torch.device("cuda")


def _t(A, idx=torch.int64):
    A = A.tocsr()
    A.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=idx), torch.as_tensor(A.indices, dtype=idx),
                                   torch.as_tensor(A.data, dtype=torch.float64), size=A.shape, device=DEV)


def _g(a):
    return torch.as_tensor(a, dtype=torch.float64, device=DEV)


class Closure:
    """cav：mac.Cavity；cols：基方向数上限（有温度时为灵敏度右端项列数）。"""

    def __init__(self, cav, cols=256):
        mo = cav.mo
        self.thermal = cav.thermal
        self.C, self.Sm, self.Am, self.Hm, self.Km = (_t(M) for M in (cav.C, mo.S, mo.A, mo.H, mo.K))
        self.bm, self.sqMu = _g(mo.b), _g(np.sqrt(cav.Mu))
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
        n = tpl.shape[0]
        self.J = _t(tpl, torch.int32)
        self.b1, self.bk = torch.zeros(1, n, dtype=torch.float64, device=DEV), torch.zeros(cols, n, dtype=torch.float64, device=DEV)
        self.s1 = nsa.DirectSolver(self.J, self.b1.T, options={"logger": _LOG})
        self.sk = nsa.DirectSolver(self.J, self.bk.T, options={"logger": _LOG})
        self.s1.plan()
        self.sk.plan()

    def whiten(self, Y):
        """L Y，L = M_u^{1/2} C，‖Y‖_M = ‖L Y‖₂。Y 为 (n,) 或 (n, k)。"""
        return (self.sqMu if Y.dim() == 1 else self.sqMu[:, None]) * (self.C @ Y)

    def _temperature(self, w, kap):
        self.J.values().copy_(self.GT @ (self.Ae @ w) + kap * self.k0)
        self.s1.factorize()
        self.b1[0] = kap * self.be
        return self.s1.solve()[:, 0]

    def temperature(self, psi, kap):
        return self._temperature(self.C @ psi, kap)

    def reduced(self, psi, PsiB, nu, kap=None):
        """约化残差 g (k,)、约化雅可比 G (k, k) 与温度 T（无温度时为 None）。"""
        w, W = self.C @ psi, self.C @ PsiB
        Fw, Hw = self.Am @ w, self.Hm @ w
        Rw = self.Sm @ (Fw * Hw) + nu * (self.Km @ w - self.bm)
        JW = self.Sm @ (Hw[:, None] * (self.Am @ W) + Fw[:, None] * (self.Hm @ W)) + nu * (self.Km @ W)
        if not self.thermal:
            return W.T @ Rw, W.T @ JW, None
        T = self._temperature(w, kap)
        self.sk.factorize()
        k = W.shape[1]
        self.bk.zero_()
        self.bk[:k] = -(self.Se @ ((self.He @ T)[:, None] * (self.Ae @ W))).T
        ST = self.sk.solve()[:, :k]
        return W.T @ (Rw - self.Bt @ T), W.T @ (JW - self.Bt @ ST), T
