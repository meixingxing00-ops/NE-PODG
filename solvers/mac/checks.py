"""MAC 求解器的验证（结果见 验证记录.md）。仓库根目录下运行：
  python -m solvers.mac.checks ops                     离散恒等式、对流反对称、雅可比差分核对
  python -m solvers.mac.checks lid N [s]               恒速顶盖驱动方腔 Re = 100 → 1000，对照 Ghia 等、Botella & Peyret
  python -m solvers.mac.checks nc N s [Pr]             自然对流 log10 Ra = 3 → 6，平均 Nu 对照参考值
  python -m solvers.mac.checks comsol N s DATA         与 paper-podnn 训练池 55 个 COMSOL 工况的平均 Nu 对照（DATA 为其 data 目录）
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from solvers.mac.mac import Cavity, nodes


def ops():
    rng = np.random.default_rng(0)
    for s in (0.0, 2.0):
        x, y = nodes(24, s), nodes(20, s)
        cav = Cavity(x, y, lid=lambda x: 16 * x ** 2 * (1 - x) ** 2, thermal=True)
        psi, T = rng.standard_normal(cav.npsi), rng.standard_normal(cav.nT)
        w = cav.C @ psi
        p = rng.standard_normal(cav.nT)
        G = -(cav.Dvol.T @ p) / cav.Mu
        P = p.reshape(len(x) - 1, len(y) - 1)
        Gd = np.concatenate([(np.diff(P, axis=0) / np.diff(cav.xc)[:, None]).ravel(),
                             (np.diff(P, axis=1) / np.diff(cav.yc)[None, :]).ravel()])
        Nw, NT = cav.mo.dconv(w, w)[1], cav.en.dconv(w, T)[1]
        print(f"s = {s}")
        print("  |D_vol C|_max          ", abs(cav.Dvol @ cav.C).max())
        print("  |G − 差分梯度|_max      ", abs(G - Gd).max())
        print("  |Cᵀ M_u G|_max          ", abs(cav.C.T @ (cav.Mu * G)).max())
        print("  |N_w + N_wᵀ|/|N_w|      ", abs(Nw + Nw.T).max() / abs(Nw).max())
        print("  |N_T + N_Tᵀ|/|N_T|      ", abs(NT + NT.T).max() / abs(NT).max())
        print("  |K_w − K_wᵀ|_max        ", abs(cav.mo.K - cav.mo.K.T).max())
        nu, kap, e = 0.03, 0.02, 1e-6
        J = cav.jacobian(psi, T, nu, kap)
        dp, dT = rng.standard_normal(cav.npsi), rng.standard_normal(cav.nT)
        r1, r0 = cav.residual(psi + e * dp, T + e * dT, nu, kap), cav.residual(psi - e * dp, T - e * dT, nu, kap)
        for k, an in ((0, J[0] @ dp + J[1] @ dT), (1, J[2] @ dp + J[3] @ dT)):
            print(f"  雅可比 {'r_ψ' if k == 0 else 'R_T'} 相对误差    ", np.linalg.norm((r1[k] - r0[k]) / (2 * e) - an) / np.linalg.norm(an))
        D = rng.standard_normal((cav.npsi, 3))
        Ta, Sa = cav.temperature(psi, kap, cav.C @ D)
        print("  辅助场 R_T(ψ, S(ψ)) 相对 ", np.linalg.norm(cav.res_T(w, Ta, kap)) / np.linalg.norm(kap * cav.en.b))
        Sd = np.stack([(cav.temperature(psi + e * d, kap) - cav.temperature(psi - e * d, kap)) / (2 * e) for d in D.T], 1)
        print("  灵敏度相对误差          ", np.linalg.norm(Sd - Sa) / np.linalg.norm(Sa))


def lid(n, s=0.0):
    ref = {100: {"psi": -0.103423}, 400: {"psi": -0.113909},
           1000: {"psi": -0.1189366, "umin": -0.3885698, "vmax": 0.3769447, "vmin": -0.5270771}}

    def ext(t, f, sign):
        cs, k = CubicSpline(t, f), np.argmin(sign * f)
        r = minimize_scalar(lambda z: sign * cs(z), bounds=(t[max(k - 1, 0)], t[min(k + 1, len(t) - 1)]), method="bounded")
        return float(cs(r.x))

    x = nodes(n, s)
    cav = Cavity(x, x, lid=lambda x: np.ones_like(x))
    psi = None
    for re in (100, 400, 1000):
        t0 = time.perf_counter()
        psi, _, st = cav.solve(1 / re, psi=psi)
        P, (u, v), h = cav.psi_full(psi), cav.faces(psi), n // 2
        out = {"psi": P.min(), "umin": ext(np.r_[0, cav.yc, 1], np.r_[0, u[h], 1], 1),
               "vmax": ext(np.r_[0, cav.xc, 1], np.r_[0, v[:, h], 0], -1),
               "vmin": ext(np.r_[0, cav.xc, 1], np.r_[0, v[:, h], 0], 1)}
        line = "  ".join(f"{k} {out[k]:+.6f}" + (f" ({(out[k] / ref[re][k] - 1) * 100:+.3f}%)" if k in ref[re] else "") for k in out)
        print(f"N={n} s={s} Re={re}  iters {st['iters']} conv {st['converged']}  {time.perf_counter() - t0:.1f}s  {line}")


def nc(n, s, pr=0.71):
    ref = {3.0: 1.1178, 4.0: 2.2448, 5.0: 4.5216, 6.0: 8.8252}
    x = nodes(n, s)
    cav = Cavity(x, x, thermal=True)
    psi = T = None
    for lra in np.arange(3.0, 6.01, 0.5):
        ra, t0 = 10 ** lra, time.perf_counter()
        psi, T, st = cav.solve(np.sqrt(pr / ra), 1 / np.sqrt(ra * pr), psi, T)
        _, _, nc_, nh = cav.nusselt(T)
        flux = float(cav.dy @ ((T.reshape(n, n)[0] + 0.5) / cav.xc[0]))
        r = f" ({(nc_ / ref[lra] - 1) * 100:+.3f}%)" if pr == 0.71 and lra in ref else ""
        t1 = time.perf_counter()
        aux = np.abs(cav.temperature(psi, 1 / np.sqrt(ra * pr)) - T).max()
        print(f"N={n} s={s} Pr={pr} log10Ra={lra:.1f}  iters {st['iters']} conv {st['converged']}  "
              f"{t1 - t0:.1f}s  Nu_C {nc_:.5f}{r}  Nu_H {nh:.5f}  flux {flux:.5f}  |S(ψ) − T| {aux:.1e}")


def comsol(n, s, root):
    root = Path(root)
    pool = [r for r in csv.DictReader(open(root / "cases.csv", encoding="utf-8")) if r["role"] == "pool"]
    x = nodes(n, s)
    cav = Cavity(x, x, thermal=True)
    for pr in sorted({float(r["Pr"]) for r in pool}):
        psi = T = None
        for r in sorted((r for r in pool if float(r["Pr"]) == pr), key=lambda r: float(r["Ra"])):
            ra = float(r["Ra"])
            psi, T, st = cav.solve(np.sqrt(pr / ra), 1 / np.sqrt(ra * pr), psi, T)
            nu = cav.nusselt(T)[2]
            d = np.load(root / "comsol" / r["file"])
            ref = 0.5 * (float(d["Nu_C_mean"]) + float(d["Nu_H_mean"]))
            print(f"Ra {ra:<9g} Pr {pr:<4g} MAC {nu:.5f}  COMSOL {ref:.5f}  {(nu / ref - 1) * 100:+.3f}%  conv {st['converged']}",
                  flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "ops":
        ops()
    elif cmd == "lid":
        lid(int(args[0]), *map(float, args[1:]))
    elif cmd == "nc":
        nc(int(args[0]), *map(float, args[1:]))
    else:
        comsol(int(args[0]), float(args[1]), args[2])
