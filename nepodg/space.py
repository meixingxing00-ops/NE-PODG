"""完整快照空间与神经互补空间（方法设计 §2、§4）。

度量由算例的白化映射 L 给出：‖y‖_M = ‖L y‖₂。正交化在白化坐标中做，所得组合系数同时作用于状态坐标，
因此基同时有状态表示（用于闭合）与白化表示（用于投影与误差）。全部为 GPU 上的 float64 张量，列为向量。
"""
import torch


def _orth(X, Xw, keep):
    """Xw 的 SVD Xw = P diag(σ) Qᵀ；返回 keep(σ) 选中方向的状态基 X Q σ⁻¹、白化正交基 P 与 σ。"""
    P, s, Qh = torch.linalg.svd(Xw, full_matrices=False)
    sel = keep(s)
    return X @ (Qh[sel].T / s[sel]), P[:, sel], s


class POD:
    """训练增量快照 U (n, Ns)；保留 σ_j/σ_1 > delta 的全部模态。"""

    def __init__(self, U, whiten, delta=1e-8):
        self.mean = U.mean(1)
        S = U - self.mean[:, None]
        Sw = whiten(S)
        self.V, self.Vw, self.sigma = _orth(S, Sw, lambda s: s / s[0] > delta)
        self.a = self.Vw.T @ Sw                                                   # 训练快照的 POD 系数 (n_V, Ns)

    def project(self, Xw):
        """白化坐标中去掉 span V 上的分量。"""
        return Xw - self.Vw @ (self.Vw.T @ Xw)


def complement(Phi, whiten, pod, delta=1e-8, tau=1e-6):
    """神经互补方向 Z（状态坐标）、Zw（白化坐标）与主角正弦谱 s。
    候选空间先取正交基（相对奇异值 > delta），去 POD 分量后的奇异值 s ∈ [0, 1] 为候选空间与 span V 之间主角的正弦，保留 s > tau。
    POD 保留到 σ/σ₁ ~ 1e-8 的模态，其精度也只到该量级，tau 取在其上。"""
    Q, Qw, _ = _orth(Phi, whiten(Phi), lambda s: s / s[0] > delta)
    A, Aw = Q - pod.V @ (pod.Vw.T @ Qw), pod.project(Qw)
    Z, Zw, s = _orth(A, Aw, lambda s: s > tau)
    return Z, Zw, s
