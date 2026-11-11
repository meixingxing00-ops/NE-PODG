"""Validation configurations; the generic solver contains no step geometry."""
import numpy as np

from solvers.openmac.openmac import Grid, OpenMAC


def stretch(n, s=2., side='both'):
    t = np.linspace(0., 1., n + 1)
    if s == 0:
        return t
    if side == 'lo':
        return 1 + np.tanh(s * (t - 1)) / np.tanh(s)
    return .5 * (1 + np.tanh(s * (2 * t - 1)) / np.tanh(s))


def coefficients(mu, Pr=.71):
    return 2 / mu[0], 2 / (mu[0] * Pr)


def inlet_psi(x, y, mu):
    z = np.clip(y, 0., 1.)
    return 3 * z**2 - 2 * z**3


def bfs(n=16, Lin=4., Lout=30., scheme='luds', heating='temperature',
        Pr=.71, traction=None, uniform=False, extension=0.):
    s = 0. if uniform else 2.
    upstream = Lin * (stretch(max(1, int(np.ceil(Lin * n / 2))), s) - 1) if Lin else np.array([0.])
    downstream = Lout * stretch(max(2, int(np.ceil(Lout * n / 2))), 0. if uniform else 2.5, 'lo')
    x = np.r_[upstream[:-1], downstream]
    if extension:
        x = np.r_[x, x[-1] + np.linspace(0., extension, int(np.ceil(extension/(x[-1]-x[-2])))+1)[1:]]
    y = np.r_[stretch(n, s) - 1, stretch(n, s)[1:]]
    xc, yc = .5 * (x[1:] + x[:-1]), .5 * (y[1:] + y[:-1])
    active = (xc[:, None] >= 0) | (yc[None, :] >= 0)

    def tag(axis, i, j, sign):
        if axis == 0 and i == 0 and yc[j] > 0:
            return 'inlet'
        if axis == 0 and i == len(x) - 1:
            return 'outlet'
        if axis == 1 and j == 0:
            return 'heated'
        if axis == 1 and j == len(y)-1:
            return 'upper'
        return 'wall'

    def geometry(mu):
        a = np.where(y < 0, -1.5 * y**2 - .5 * y**3, 0.)
        return x, y + (mu[1] - 2) * a

    def thermal(xb, yb, label, mu):
        if label == 'inlet':
            return 'dirichlet', 0.
        if label == 'heated':
            return ('dirichlet', 1.) if heating == 'temperature' else ('neumann', coefficients(mu, Pr)[1])
        return 'neumann', 0.

    return OpenMAC(Grid(x, y, active, tag), geometry, lambda mu: coefficients(mu, Pr),
                   inlet_psi, np.array([100., 2.]), thermal=thermal,
                   traction=traction, scheme=scheme)


def channel(n=12, scheme='luds'):
    L = 2.
    x, y = np.linspace(0, L, 2 * n + 1), np.linspace(0, 1, n + 1)

    def tag(axis, i, j, sign):
        if axis == 0:
            return 'inlet' if sign < 0 else 'outlet'
        return 'wall'

    def thermal(xb, yb, label, mu):
        if label == 'outlet':
            return 'neumann', 0.
        return 'dirichlet', 1 + np.cos(np.pi * xb / L) * np.sin(np.pi * yb)

    def source(xb, yb, mu):
        nu, kappa = coefficients(mu)
        a, b = np.pi / L, np.pi
        return (-6 * yb * (1 - yb) * a * np.sin(a * xb) * np.sin(b * yb)
                + kappa * (a*a + b*b) * np.cos(a * xb) * np.sin(b * yb))

    return OpenMAC(Grid(x, y, np.ones((2 * n, n), bool), tag), lambda mu: (x, y),
                   coefficients, inlet_psi, np.array([20., 2.]), thermal=thermal,
                   force=lambda xb, yb, mu: (12 * coefficients(mu)[0], 0.),
                   source=source, scheme=scheme)
