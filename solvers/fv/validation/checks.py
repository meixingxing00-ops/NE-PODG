"""分区解析解、双向二维速度、解析参数导数和区域热量收支。

运行：D:/Anaconda/python.exe -m solvers.fv.validation.checks
"""
import csv
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from solvers.fv.fv import TransportChannel, poiseuille_fluxes, stretch, velocity_fluxes


def relative(a, b, weights=None):
    weights = 1 if weights is None else weights
    return float(np.sqrt(np.sum(weights * (a - b)**2) / np.sum(weights * b**2)))


def fixture():
    x = 5 * stretch(80, 2, "both")
    y = np.r_[-0.25 + 0.25 * stretch(16, 1.5, "hi"), stretch(48, 2, "lo")[1:]]
    solid = np.broadcast_to((y[:-1] + y[1:] < 0), (80, 64))
    xc = (x[1:] + x[:-1]) / 2
    return TransportChannel(x, y, poiseuille_fluxes(x, y), solid=solid,
                            bc={"left": ("dirichlet", 0),
                                "bottom": ("dirichlet", np.sin(np.pi * np.minimum(xc, 1) / 2)**2)})


def run():
    rows = []
    for direction in ("x", "y"):
        a = np.r_[-0.25 + 0.25 * stretch(9, 1.2, "hi"), stretch(23, 1.5, "lo")[1:]]
        b = stretch(7, 1, "both")
        x, y = (a, b) if direction == "x" else (b, a)
        c = (a[1:] + a[:-1]) / 2
        solid = np.broadcast_to(c[:, None] < 0, (32, 7)) if direction == "x" else np.broadcast_to(c < 0, (7, 32))
        ch = TransportChannel(x, y, solid=solid, bc={
            "left" if direction == "x" else "bottom": ("dirichlet", 1),
            "right" if direction == "x" else "top": ("dirichlet", 0)})
        for ratio in (0.1, 1, 100):
            T = ch.solve(17, ratio)
            heat = 1 / (0.25 / ratio + 1)
            exact1 = np.where(c < 0, 1 - heat * (c + 0.25) / ratio, heat * (1 - c))
            exact = np.broadcast_to(exact1[:, None], (32, 7)) if direction == "x" else np.broadcast_to(exact1, (7, 32))
            error = float(np.max(np.abs(T - exact.ravel())))
            assert error < 2e-11
            rows.append(dict(test=f"layered_{direction}", Pe=17, ratio=ratio, value=error))

    # 解析给定速度在两个坐标中都有正、负通量，常数场须精确保持。
    x, y = stretch(19, 1, "both"), stretch(17, 1.2, "lo")
    velocity = lambda X, Y: (Y - 0.4 + 0 * X, 0.2 - X + 0 * Y)
    fluxes = velocity_fluxes(x, y, velocity)
    for scheme in ("cds", "luds", "upwind"):
        ch = TransportChannel(x, y, fluxes, bc={side: ("dirichlet", 2.5)
                              for side in ("left", "right", "bottom", "top")}, scheme=scheme)
        T = ch.solve(7, 1)
        error = float(np.max(np.abs(T - 2.5)))
        assert error < 1e-11
        rows.append(dict(test=f"bidirectional_{scheme}", Pe=7, ratio=1, value=error))

    ch = fixture()
    T0 = ch.solve(100, 3)
    hot = ch._boundaries["bottom"][-1]
    rotated = TransportChannel(ch.y, ch.x, (ch.fy.T, ch.fx.T), solid=ch.solid.T,
                               bc={"bottom": ("dirichlet", 0), "left": ("dirichlet", hot)})
    reversed_flow = TransportChannel(-ch.x[::-1], ch.y, (-ch.fx[::-1], ch.fy[::-1]),
                                     solid=ch.solid[::-1],
                                     bc={"right": ("dirichlet", 0), "bottom": ("dirichlet", hot[::-1])})
    for name, other, expected in (("rotated_nonconstant", rotated, T0.reshape(ch.nx, ch.ny).T.ravel()),
                                  ("negative_nonconstant", reversed_flow, T0.reshape(ch.nx, ch.ny)[::-1].ravel())):
        error = relative(other.solve(100, 3), expected)
        assert error < 1e-10
        rows.append(dict(test=name, Pe=100, ratio=3, value=error))
    interface = int(np.argmin(np.abs(ch.y)))
    for Pe, ratio in ((10, 0.1), (10, 100), (1000, 0.1), (1000, 100), (100, 10**0.5)):
        T, *S = ch.solve(Pe, ratio, True)
        A, f, D = ch.matrix(Pe, ratio, True)
        for index, (name, p) in enumerate((("Pe", Pe), ("ratio", ratio))):
            h = 1e-4 * p
            plus = (Pe + h, ratio) if index == 0 else (Pe, ratio + h)
            minus = (Pe - h, ratio) if index == 0 else (Pe, ratio - h)
            Ap, fp = ch.matrix(*plus)
            Am, fm = ch.matrix(*minus)
            da, df = D[name]
            ea = float(spla.norm(da - (Ap - Am) / (2 * h)) / spla.norm(da))
            ef = relative((fp - fm) / (2 * h), df) if np.linalg.norm(df) else float(np.linalg.norm(fp - fm))
            es = relative((ch.solve(*plus) - ch.solve(*minus)) / (2 * h), S[index], ch.Mh)
            assert max(ea, ef) < 1e-6 and es < 2e-6
            for kind, value in (("matrix", ea), ("load", ef), ("sensitivity", es)):
                rows.append(dict(test=f"{kind}_{name}", Pe=Pe, ratio=ratio, value=value))
        face = ch.face_fields(T, Pe, ratio)
        tm = T.reshape(ch.nx, ch.ny)
        trace = face["ty"][:, interface]
        q = face["qy"][:, interface]
        qs = ratio / Pe * ch.dx * (tm[:, interface - 1] - trace) / (-ch.yc[interface - 1])
        qf = 1 / Pe * ch.dx * (trace - tm[:, interface]) / ch.yc[interface]
        interface_error = max(np.max(np.abs(q - qs)), np.max(np.abs(q - qf))) / np.max(np.abs(q))
        bf = ch.boundary_fluxes(T, Pe, ratio)
        Q = float(q.sum())
        # 分别加总两区域外边界和界面，检查不只依赖共享面通量的代数恒等式。
        inlet_diff = bf["left"]["diffusion"]
        solid_defect = (bf["bottom"]["diffusion"].sum() + inlet_diff[:interface].sum() + Q) / Q
        fluid_defect = (inlet_diff[interface:].sum() + bf["right"]["advection"].sum() - Q) / Q
        global_defect = ch.balance(T, Pe, ratio)["defect"] / Q
        backward = np.linalg.norm(A @ T - f) / (spla.norm(A) * np.linalg.norm(T) + np.linalg.norm(f))
        for name, value in (("interface_flux", interface_error), ("solid_balance", abs(solid_defect)),
                            ("fluid_balance", abs(fluid_defect)), ("global_balance", abs(global_defect)),
                            ("backward_error", backward)):
            assert value < 1e-9
            rows.append(dict(test=name, Pe=Pe, ratio=ratio, value=float(value)))
        assert T.min() > -1e-10 and T.max() < 1 + 1e-10
        # 任意向量也有共享温度迹，验证它并非解收敛后的偶然现象。
        rng = np.random.default_rng(0)
        arbitrary = rng.normal(size=ch.n)
        ft = ch.face_fields(arbitrary, Pe, ratio)["ty"][:, interface]
        assert np.all(np.isfinite(ft))

    # 有源、非零入流和 Neumann 载荷的守恒与导数；全部边界值固定。
    x, y = np.linspace(0, 2, 15), np.linspace(0, 1, 13)
    ch = TransportChannel(x, y, velocity_fluxes(x, y, lambda X, Y: (1 + 0 * Y, 0 * X)),
                          bc={"left": ("dirichlet", 0.7), "top": ("neumann", 0.03)}, source=0.02)
    T = ch.solve(20, 1)
    defect = abs(ch.balance(T, 20, 1)["defect"])
    assert defect < 1e-11
    rows.append(dict(test="source_neumann_balance", Pe=20, ratio=1, value=defect))

    path = Path(__file__).with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} checks passed; {path}", flush=True)


if __name__ == "__main__":
    run()
