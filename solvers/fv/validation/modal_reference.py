"""广义共轭 Graetz 单模态半解析参照，独立于二维 FV 离散。

参照 Pierre & Plouraboué, arXiv:1812.05351v2, (4) 和 §2.4 的分离变量方程。
横向 ODE 用 DOP853，特征根用 Brent；固体用正弦精确表达。
两端为模态 Dirichlet 条件；此验证不把它当成算例入口/出口条件的文献表值。
运行：D:/Anaconda/python.exe -m solvers.fv.validation.modal_reference
"""
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from solvers.fv.fv import TransportChannel, poiseuille_fluxes, stretch


class GraetzMode:
    def __init__(self, Pe, ratio, thickness=0.25, tol=2e-12):
        self.Pe, self.ratio, self.thickness = Pe, ratio, thickness

        def integrate(lam, dense=False):
            def ode(y, p):
                return p[1], (Pe * 6 * y * (1 - y) * lam - lam**2) * p[0]
            return solve_ivp(ode, (1, 0), [1.0, 0.0], method="DOP853", rtol=tol, atol=tol / 10,
                             dense_output=dense)

        def determinant(lam):
            p, dp = integrate(lam).y[:, -1]
            return dp - ratio * lam / np.tan(lam * thickness) * p

        a, fa = -1e-9, determinant(-1e-9)
        b = a * 1.25
        while b > -np.pi / (2 * thickness):
            fb = determinant(b)
            if fa * fb < 0:
                break
            a, fa, b = b, fb, b * 1.25
        else:
            raise RuntimeError("first downstream mode not bracketed")
        self.lam = brentq(determinant, b, a, xtol=1e-14, rtol=1e-14)
        self._solution = integrate(self.lam, True)
        self.phi0, self.dphi0 = self._solution.sol(0)
        self.match_error = abs(determinant(self.lam))

    def phi(self, y):
        y = np.asarray(y)
        fluid = self._solution.sol(np.maximum(y, 0).ravel())[0].reshape(y.shape)
        solid = self.phi0 * np.sin(self.lam * (y + self.thickness)) / np.sin(self.lam * self.thickness)
        return np.where(y >= 0, fluid, solid)

    def temperature(self, x, y):
        return np.exp(self.lam * np.asarray(x)) * self.phi(y)


def run():
    rows = []
    for Pe, ratio in ((10, 0.1), (100, 3), (1000, 100)):
        mode = GraetzMode(Pe, ratio)
        tight = GraetzMode(Pe, ratio, tol=5e-14)
        assert abs(mode.lam - tight.lam) < 1e-10
        previous = None
        for level in (1, 2, 4):
            nx, ns, nf = (v * level for v in (32, 8, 24))
            x = 3 * stretch(nx, 1, "both")
            y = np.r_[-0.25 + 0.25 * stretch(ns, 1, "hi"), stretch(nf, 1.5, "lo")[1:]]
            yc = (y[1:] + y[:-1]) / 2
            solid = np.broadcast_to(yc < 0, (nx, ns + nf))
            ch = TransportChannel(x, y, poiseuille_fluxes(x, y), solid=solid,
                                  bc={"left": ("dirichlet", mode.temperature(0, yc)),
                                      "right": ("dirichlet", mode.temperature(3, yc)),
                                      "bottom": ("dirichlet", 0)})
            T = ch.solve(Pe, ratio)
            exact = mode.temperature(ch.xc[:, None], ch.yc[None, :]).ravel()
            error = np.sqrt(ch.Mh @ (T - exact)**2 / (ch.Mh @ exact**2))
            fields = ch.face_fields(T, Pe, ratio)
            ti = mode.temperature(ch.xc, 0)
            qi = -np.exp(mode.lam * ch.xc) * mode.dphi0 / Pe
            et = np.sqrt(ch.dx @ (fields["ty"][:, ns] - ti)**2 / (ch.dx @ ti**2))
            eq = np.sqrt(ch.dx @ (fields["qy"][:, ns] / ch.dx - qi)**2 / (ch.dx @ qi**2))
            order = np.log2(previous / error) if previous is not None else np.nan
            rows.append(dict(Pe=Pe, ratio=ratio, level=level, cells=ch.n, eigenvalue=mode.lam,
                             eigenvalue_refinement=abs(mode.lam - tight.lam), matching=mode.match_error,
                             e_T=error, e_trace=et, e_flux=eq, order=order))
            print(rows[-1], flush=True)
            if previous is not None:
                assert order > 1.7
            previous = error
        assert error < 1e-3 and eq < 2e-3
    path = Path(__file__).with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run()
