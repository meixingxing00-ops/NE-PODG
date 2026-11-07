"""高保真解核对（solvers/fv，本算例网格）：(1) K_L、C_L、f_L 与组装的中心差分；(2) ∂T/∂η₁、∂T/∂η₂ 与解的中心差分；
(3) 温度非负与全局热量收支。工况 L ∈ {1, 10} × Pe ∈ {0.1, 316}，另加 (3.16, 10)。
用法（仓库根目录）：python -m cases.graetz.fom.checks"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import scipy.sparse.linalg as spla

from cases.graetz.case import mesh

ch = mesh()
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
srel = lambda a, b: spla.norm(a - b) / spla.norm(b)
eta = lambda L, Pe: np.array([2 * np.log10(L) - 1, (np.log10(Pe) - 0.5) / 1.5])
mu = lambda e: (10 ** ((e[0] + 1) / 2), 10 ** (1.5 * e[1] + 0.5))

print("算子对 L 的导数（中心差分 h = 1e-4 L）")
for L in (1.0, 3.16, 10.0):
    K, C, f, KL, CL, fL = ch.operators(L)
    h = 1e-4 * L
    p, m = ch.operators(L + h), ch.operators(L - h)
    print(f"  L = {L}: K_L {srel(KL, (p[0] - m[0]) / (2 * h)):.1e}  C_L {srel(CL, (p[1] - m[1]) / (2 * h)):.1e}"
          f"  f_L {rel(fL, (p[2] - m[2]) / (2 * h)):.1e}  ‖C_L‖/‖C‖ {spla.norm(CL) / spla.norm(C):.1e}")

print("灵敏度（中心差分 h = 1e-5）、非负、热量收支")
for L, Pe in ((1, 0.1), (10, 0.1), (1, 316), (10, 316), (3.16, 10)):
    T, s1, s2 = ch.solve(L, Pe, sens=True)
    e, h = eta(L, Pe), 1e-5
    d = [(ch.solve(*mu(e + h * u)) - ch.solve(*mu(e - h * u))) / (2 * h) for u in np.eye(2)]
    b = ch.balance(T, L, Pe)
    print(f"  L = {L}, Pe = {Pe}: ∂η₁ {rel(s1, d[0]):.1e}  ∂η₂ {rel(s2, d[1]):.1e}  min T/max T {T.min() / T.max():.1e}"
          f"  收支 {(b[0] - sum(b[1:])) / b[0]:.1e}（顶壁 {b[1] / b[0]:.3f}，入口 {b[2] / b[0]:.3f}，出口 {b[3] / b[0]:.3f}）")
