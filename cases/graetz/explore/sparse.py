"""少快照预算试算：给定训练池点集，POD 与 SA-PODG 在全部内插、外推测试上的闭合误差。
空间：POD 为中心化快照（σ/σ₁ > 1e-8），SA-PODG 再加各点 ∂T/∂η₁、∂T/∂η₂ 去 POD 分量（τ = 1e-6）；单点时 POD 为空，仿射空间为 T_i + span S_i。
每组记录场误差 e（参考域 L²）、SA-PODG 最佳逼近误差 e_proj、T_av 误差、局部 Nu 沿受热壁的最大误差，取中位数与最大值。
结果 explore/results/sparse.csv。用法（仓库根目录）：python -m cases.graetz.explore.sparse"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.graetz.case import ROOT, cases, mesh

# 训练池下标 (i_L, j_Pe)：L = 1, 1.78, 3.16, 5.62, 10；Pe = 0.1, 0.316, 1, 3.16, 10, 31.6, 100
SETS = {
    "N1 中心 (3.16, 3.16)": [(2, 3)],
    "N1 角 (1, 0.1)": [(0, 0)],
    "N1 角 (10, 100)": [(4, 6)],
    "N1 角 (1, 100)": [(0, 6)],
    "N2 L3.16 × Pe{0.316, 31.6}": [(2, 1), (2, 5)],
    "N2 L3.16 × Pe{1, 10}": [(2, 2), (2, 4)],
    "N2 L3.16 × Pe{0.1, 100}": [(2, 0), (2, 6)],
    "N2 对角 (1, 0.1)(10, 100)": [(0, 0), (4, 6)],
    "N2 对角 (1.78, 0.316)(5.62, 31.6)": [(1, 1), (3, 5)],
    "N3 L3.16 × Pe{0.1, 3.16, 100}": [(2, 0), (2, 3), (2, 6)],
    "N3 L3.16 × Pe{0.316, 3.16, 31.6}": [(2, 1), (2, 3), (2, 5)],
    "N4 四角 L{1, 10} × Pe{0.1, 100}": [(0, 0), (4, 0), (0, 6), (4, 6)],
    "N4 L{1, 10} × Pe{0.316, 31.6}": [(0, 1), (4, 1), (0, 5), (4, 5)],
    "N4 L{1.78, 5.62} × Pe{0.316, 31.6}": [(1, 1), (3, 1), (1, 5), (3, 5)],
    "N6 L{1, 10} × Pe{0.1, 3.16, 100}": [(0, 0), (4, 0), (0, 3), (4, 3), (0, 6), (4, 6)],
}
ch = mesh()
w = np.sqrt(ch.Mh)
ld = lambda i: np.load(ROOT / "data" / f"{i}.npz")
tests = {r: [(ld(c["id"])["T"], float(c["L"]), float(c["Pe"])) for c in cases() if c["role"] == r] for r in ("interp", "extrap")}


def orth(X, tol):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * s[0]] if s.size and s[0] > 0 else U[:, :0]


def errors(T0, Bw, Tr, L, Pe):
    B = Bw / w[:, None]
    A, f = ch.matrix(L, Pe)
    T = T0 + B @ np.linalg.solve(B.T @ (A @ B), B.T @ (f - A @ T0))
    nr, d = np.linalg.norm(w * Tr), w * (Tr - T0)
    return (np.linalg.norm(w * (T - Tr)) / nr, np.linalg.norm(d - Bw @ (Bw.T @ d)) / nr,
            abs(ch.tav(T, Pe) / ch.tav(Tr, Pe) - 1), np.abs(ch.nusselt(T, Pe) / ch.nusselt(Tr, Pe) - 1).max())


out = Path(__file__).resolve().parent / "results" / "sparse.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh)
    wr.writerow(["set", "space", "role", "nV", "nZ"] + [f"{k}_{s}" for k in ("e", "e_proj", "e_Tav", "e_Nu") for s in ("med", "max")])
    for name, pts in SETS.items():
        ds = [ld(f"P{j * 5 + i + 1:03d}") for i, j in pts]
        U = np.stack([d["T"] for d in ds], 1)
        T0 = U.mean(1)
        V = orth(w[:, None] * (U - T0[:, None]), 1e-8)
        S = w[:, None] * np.stack([d[k] for d in ds for k in ("T_e1", "T_e2")], 1)
        S = S / np.linalg.norm(S, axis=0)
        Z = orth(S - V @ (V.T @ S), 1e-6)
        for sp, Bw in (("pod", V), ("sapodg", np.hstack([V, Z]))):
            if Bw.shape[1] == 0:
                continue
            for role, ts in tests.items():
                E = np.array([errors(T0, Bw, *t) for t in ts])
                wr.writerow([name, sp, role, V.shape[1], Bw.shape[1] - V.shape[1]]
                            + [f"{v:.3e}" for k in range(4) for v in (np.median(E[:, k]), E[:, k].max())])
                if sp == "sapodg":
                    print(f"{name:36s} {role:6s} e {np.median(E[:, 0]):.1e}/{E[:, 0].max():.1e}  T_av {np.median(E[:, 2]):.1e}/{E[:, 2].max():.1e}"
                          f"  Nu {np.median(E[:, 3]):.1e}/{E[:, 3].max():.1e}", flush=True)
print(f"→ {out}")
