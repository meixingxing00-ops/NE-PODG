"""共轭传热 FOM 配置与输出；只用于验证、网格研究和后续算例设计。"""
import numpy as np

from solvers.fv.fv import TransportChannel, poiseuille_fluxes, stretch

LENGTH, THICKNESS = 5.0, 0.25
NU_X = np.array([0.25, 1.25, 2.5, 4.75])


def mesh(level=2, scheme="luds"):
    """嵌套网格族 nx=80r, ns=16r, nf=48r；r 可为 0.5、1、2、4、8。"""
    nx, ns, nf = (int(v * level) for v in (80, 16, 48))
    x = LENGTH * stretch(nx, 2.0, "both")
    ys = -THICKNESS + THICKNESS * stretch(ns, 1.5, "hi")
    yf = stretch(nf, 2.0, "lo")
    y = np.concatenate([ys, yf[1:]])
    solid = np.broadcast_to(((y[:-1] + y[1:]) / 2 < 0), (nx, ns + nf))
    xc = (x[1:] + x[:-1]) / 2
    outer_wall = np.sin(np.pi * np.minimum(xc, 1.0) / 2)**2
    ch = TransportChannel(x, y, poiseuille_fluxes(x, y), solid=solid,
                          bc={"left": ("dirichlet", 0), "bottom": ("dirichlet", outer_wall)}, scheme=scheme)
    return ch


def outputs(ch, T, Pe, ratio):
    """q 从固体指向流体为正，Nu 以流体全高 H=1 定义。"""
    j = int(np.argmin(np.abs(ch.y)))
    faces = ch.face_fields(T, Pe, ratio)
    tw, q = faces["ty"][:, j], faces["qy"][:, j] / ch.dx
    # 除以精确的平均流量 1；固体通量已为 0。
    bulk = (np.asarray(T).reshape(ch.nx, ch.ny) * ch.fx[0]).sum(axis=1)
    nu = Pe * q / (tw - bulk)
    return dict(Q=float(ch.dx @ q), Tw=float(ch.dx @ tw / LENGTH),
                Tout=float(bulk[-1]), Nu=np.interp(NU_X, ch.xc, nu),
                tw=tw, q=q, bulk=bulk, nu=nu)
