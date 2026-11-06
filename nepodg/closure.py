"""物理闭合（方法设计 §5）：y(c) = y₀ + B c，g(c) = Bᵀ r_eff(y(c)) = 0，LM 求解。

算例提供 reduced(y, B, mu) -> (g, G, aux)，G = Bᵀ J_eff B；g、G 为 GPU 张量，LM 的小规模线性代数在 CPU 上做。
收敛：‖g‖ ≤ tol ‖g(c⁽⁰⁾)‖，或 LM 步长 ≤ step·max(1, ‖c‖)（初值已在解上时残差无法再相对下降）。
"""
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator


def lm(fun, c0, tol=1e-10, step=1e-12, maxit=50, maxrej=12):
    """fun(c) -> (g, G, aux)，c 为 numpy。返回 c、末次 aux 与状态。"""
    c = c0.copy()
    g, G, aux = fun(c)
    g0, lam, it, rej, done = np.linalg.norm(g), 1e-6, 0, 0, False
    while not done and it < maxit and rej < maxrej:
        A = G.T @ G
        dc = np.linalg.solve(A + lam * np.trace(A) / len(c) * np.eye(len(c)), -G.T @ g)
        if np.linalg.norm(dc) <= step * max(1.0, np.linalg.norm(c)):
            done = True
            break
        g1, G1, aux1 = fun(c + dc)
        it += 1
        if np.linalg.norm(g1) < np.linalg.norm(g):
            c, g, G, aux, lam, rej = c + dc, g1, G1, aux1, max(lam / 10, 1e-12), 0
            done = np.linalg.norm(g) <= tol * g0
        else:
            lam, rej = lam * 10, rej + 1
    return c, aux, {"converged": bool(done), "iters": it, "res": float(np.linalg.norm(g) / g0)}


def solve(reduced, y0, B, mu, c0):
    """在仿射空间 y₀ + span B 中求解；返回 y、aux 与状态。"""
    def fun(c):
        g, G, aux = reduced(y0 + B @ torch.as_tensor(c, device=B.device), B, mu)
        return g.cpu().numpy(), G.cpu().numpy(), aux
    c, aux, st = lm(fun, c0)
    return y0 + B @ torch.as_tensor(c, device=B.device), aux, st


class Init:
    """POD 系数初值：训练快照 POD 系数在参数坐标 η 上的薄板样条插值（方法设计 §10）。"""

    def __init__(self, eta, a):
        self.rbf = RBFInterpolator(np.asarray(eta), a.T.cpu().numpy(), kernel="thin_plate_spline", degree=1)

    def __call__(self, eta):
        return self.rbf(np.asarray(eta)[None])[0]
