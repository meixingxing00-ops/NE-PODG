"""拓宽参数域试算：N6 型快照（Pe 取两端与对数中点 × L 两端）下 POD 与 SA-PODG 的闭合误差，测试为域内 40 个对数均匀随机点（种子 0）；另比较 Pe = 10³、10⁴ 与 L = 100 时 r = 1、2 两套网格的 T_av。
用法（仓库根目录）：python -m cases.graetz.explore.widen"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
from solvers.fv.fv import Channel, grid

ch = Channel(*grid(40, 1.5, 160, 2.0, 80, 2.0), scheme="luds")
w = np.sqrt(ch.Mh)


def orth(X, tol):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * s[0]] if len(s) else X


def closure(T0, B, L, Pe):
    A, f = ch.matrix(L, Pe)
    AB = A @ B
    return T0 + B @ np.linalg.solve(B.T @ AB, B.T @ (f - A @ T0))


def run(name, Lr, Pr, seed=0):
    lL, lP = np.log10(Lr), np.log10(Pr)
    tr = [(10 ** a, 10 ** b) for b in (lP[0], lP.mean(), lP[1]) for a in lL]
    snaps, sens = [], []
    for L, Pe in tr:
        T, s1, s2 = ch.solve(L, Pe, sens=True)
        snaps.append(T)
        sens += [s1, s2]
    U = np.array(snaps).T
    T0 = U.mean(1)
    V = orth(w[:, None] * (U - T0[:, None]), 1e-8)
    S = w[:, None] * np.array(sens).T
    S = S / np.linalg.norm(S, axis=0)
    Z = orth(S - V @ (V.T @ S), 1e-6)
    VS = np.hstack([V, Z])
    rng = np.random.default_rng(seed)
    q = 10 ** np.column_stack([rng.uniform(*lL, 40), rng.uniform(*lP, 40)])
    E = {"POD": [], "SA": [], "SAproj": []}
    for L, Pe in q:
        Tr = ch.solve(L, Pe)
        nr = np.linalg.norm(w * Tr)
        for k, B in (("POD", V), ("SA", VS)):
            T = closure(T0, B / w[:, None], L, Pe)
            E[k].append(np.linalg.norm(w * (T - Tr)) / nr)
        d = w * (Tr - T0)
        E["SAproj"].append(np.linalg.norm(d - VS @ (VS.T @ d)) / nr)
    hi = q[:, 1] > 10 ** (lP[1] - 0.5)
    out = " ".join(f"{k} {np.median(v):.1e}/{np.max(v):.1e}" for k, v in E.items())
    print(f"{name:28s} dim {V.shape[1]}+{Z.shape[1]}  {out}  SA 高Pe段中位 {np.median(np.array(E['SA'])[hi]):.1e}", flush=True)


run("L[1,10]  Pe[0.1,100]", (1, 10), (0.1, 100))
run("L[1,10]  Pe[0.1,1e3]", (1, 10), (0.1, 1e3))
run("L[1,10]  Pe[1,1e3]", (1, 10), (1, 1e3))
run("L[1,10]  Pe[1,1e4]", (1, 10), (1, 1e4))
run("L[1,100] Pe[0.1,100]", (1, 100), (0.1, 100))
run("L[1,100] Pe[1,1e3]", (1, 100), (1, 1e3))
# 网格是否够用：Pe = 1e3、1e4 下 T_av 的 r = 1 与 r = 2 之差
ch2 = Channel(*grid(80, 1.5, 320, 2.0, 160, 2.0), scheme="luds")
for L, Pe in ((1, 1e3), (10, 1e3), (1, 1e4), (10, 1e4), (100, 100)):
    a, b = ch.tav(ch.solve(L, Pe), Pe), ch2.tav(ch2.solve(L, Pe), Pe)
    print(f"网格 L={L} Pe={Pe:g}: T_av r1 {a:.4g} r2 {b:.4g} 相对差 {abs(a / b - 1):.1e}")
