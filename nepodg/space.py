"""快照空间、灵敏度增广的基础空间与神经互补空间（方法设计 §2、§4）。

度量由算例的白化映射 L 给出：‖y‖_M = ‖L y‖₂。正交化在白化坐标中做，所得组合系数同时作用于状态坐标，
因此基同时有状态表示（用于闭合）与白化表示（用于投影与误差）。全部为 GPU 上的 float64 张量，列为向量。
"""
import torch


def _orth(X, Xw, keep):
    """Xw 的 SVD Xw = P diag(σ) Qᵀ；返回 keep(σ) 选中方向的状态基 X Q σ⁻¹、白化正交基 P 与 σ。"""
    P, s, Qh = torch.linalg.svd(Xw, full_matrices=False)
    sel = keep(s)
    return X @ (Qh[sel].T / s[sel]), P[:, sel], s


class Space:
    """M 正交基：状态表示 V 与白化表示 Vw（Vw 列正交）。"""

    def __init__(self, V, Vw):
        self.V, self.Vw = V, Vw

    def project(self, Xw):
        """白化坐标中去掉本空间上的分量。"""
        return Xw - self.Vw @ (self.Vw.T @ Xw)


class POD(Space):
    """快照空间 V：训练增量快照 U (n, Ns)，保留 σ_j/σ_1 > delta 的全部模态。"""

    def __init__(self, U, whiten, delta=1e-8):
        self.mean = U.mean(1)
        D = U - self.mean[:, None]
        Dw = whiten(D)
        V, Vw, self.sigma = _orth(D, Dw, lambda s: s / s[0] > delta)
        super().__init__(V, Vw)
        self.a = self.Vw.T @ Dw                                                   # 训练快照的 POD 系数 (n_V, Ns)


def complement(Phi, whiten, base, delta=1e-8, tau=1e-6):
    """候选方向 Phi 相对空间 base 的互补方向 Z（状态坐标）、Zw（白化坐标）与主角正弦谱 s（全部奇异值，前 n_Z 个为保留方向）。
    候选空间先取正交基（相对奇异值 > delta），去 base 分量后的奇异值 s ∈ [0, 1] 为候选空间与 span base 之间主角的正弦，保留 s > tau。
    POD 保留到 σ/σ₁ ~ 1e-8 的模态，其精度也只到该量级，tau 取在其上。"""
    Q, Qw, _ = _orth(Phi, whiten(Phi), lambda s: s / s[0] > delta)
    A, Aw = Q - base.V @ (base.Vw.T @ Qw), base.project(Qw)
    return _orth(A, Aw, lambda s: s > tau)


def sensitivity_space(pod, Sigma, whiten, delta=1e-8, tau=1e-6):
    """基础空间 W = V ⊕ S：Sigma (n, d N_s) 为训练点稳态灵敏度，去 V 分量后保留主角正弦 > tau 的方向 S。
    返回 W（前 n_V 列为 POD 模态）、n_S 与主角正弦谱。"""
    S, Sw, s = complement(Sigma, whiten, pod, delta, tau)
    return Space(torch.cat([pod.V, S], 1), torch.cat([pod.Vw, Sw], 1)), S.shape[1], s
