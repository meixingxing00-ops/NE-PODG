"""封闭热方腔的基础研究工具：标量输出、延拓和完整雅可比的数值可逆性检查。

CSV 记录工况诊断；流场只留在内存，不保存为快照。
"""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, onenormest, splu


def audit_cases(cav, cases, names, mapping, write_rows):
    """指定工况逐点延拓并检查完整 J；只输出标量。"""
    rows, psi, T, current = [], None, None, None
    for case in cases:
        target = np.log10([float(case[k]) for k in names])
        start = target if current is None else current
        psi, T, st = advance(cav, start, target, mapping, psi, T)
        nu, kap, kw = mapping(target)
        row = dict(id=case["id"], role=case["role"], **{k: float(case[k]) for k in names},
                   **st, **scalars(cav, psi, T), **jacobian_audit(cav, psi, T, nu, kap, **kw))
        assert row["jacobian_ok"], row
        rows.append(row)
        write_rows("audit.csv", rows)
        current = target
        print(f"audit {case['id']} Nu={row['Nu_cold']:.8g} rcond={row['rcond_est']:.2e}", flush=True)
    return rows


def sensitivity_check(cav, points, names, mapping, chain, start):
    """物理对数参数的链式导数对照中心差分，chain 返回每列 q_z。"""
    rows = []
    for point in points:
        z = np.log10(point)
        psi, T, st = advance(cav, np.array([start, z[1]]), z, mapping)
        nu, kap, kw = mapping(z)
        derivatives = cav.sensitivity(psi, T, nu, kap, parameters=("nu", "kap", "buoyancy", "drag"), **kw)
        exact = chain(nu, kap, kw, derivatives)
        for k, name in enumerate(names):
            step = np.eye(2)[k] * 1e-4
            states = []
            for sign in (1, -1):
                n1, k1, kw1 = mapping(z + sign * step)
                p1, t1, status = cav.solve(n1, k1, psi, T, tol=1e-12, **kw1)
                assert status["converged"], status
                states.append(np.concatenate([p1, t1]))
            fd = (states[0] - states[1]) / 2e-4
            error, pfd = (exact[k] - fd)[:cav.npsi], fd[:cav.npsi]
            psi_error = float(np.sqrt(error @ (cav.Mpsi @ error) / (pfd @ (cav.Mpsi @ pfd))))
            t_error = float(np.linalg.norm((exact[k] - fd)[cav.npsi:]) / np.linalg.norm(fd[cav.npsi:]))
            rows.append(dict(**dict(zip(names, point)), derivative="log10_" + name,
                             psi_M_error=psi_error, T_L2_error=t_error))
            assert psi_error < 1e-5 and t_error < 1e-4, rows[-1]
    return rows


def scalars(cav, psi, T):
    nc, nh, ncm, nhm = cav.nusselt(T)
    w = cav.C @ psi
    T2 = T.reshape(len(cav.dx), len(cav.dy))
    if cav.thermal_axis == 0:
        fc = cav.dy @ ((T2[0] + 0.5) / cav.xc[0])
        fh = cav.dy @ ((0.5 - T2[-1]) / (1 - cav.xc[-1]))
    else:
        fc = cav.dx @ ((T2[:, 0] + 0.5) / cav.yc[0])
        fh = cav.dx @ ((0.5 - T2[:, -1]) / (1 - cav.yc[-1]))
    u, v = cav.faces(psi)
    h, k = len(cav.dx) // 2, len(cav.dy) // 2
    return dict(Nu_cold=ncm, Nu_hot=nhm, Nu_flux_cold=float(fc), Nu_flux_hot=float(fh),
                flux_balance=float(abs(fc - fh) / max(abs(fc), abs(fh), 1e-30)),
                Nu_local_max=float(max(nc.max(), nh.max())), psi_max=float(np.max(abs(psi))),
                psi_min=float(min(0, psi.min())), psi_positive=float(max(0, psi.max())),
                u_mid_abs=float(np.max(abs(u[h]))), v_mid_abs=float(np.max(abs(v[:, k]))),
                kinetic=float(0.5 * w @ (cav.Mu * w)), T_min=float(T.min()), T_max=float(T.max()))


def jacobian_audit(cav, psi, T, nu, kap, **coeff):
    """行列 1 范数平衡后的完整 J：SuperLU 主元、逆 1 范数估计及回代缺陷。

    rcond_est 是数值估计，不是最小奇异值的严格界，也不能证明连续参数域可逆或稳态稳定。
    """
    J = cav.jac_full(psi, T, nu, kap, **coeff)
    rows = np.asarray(abs(J).sum(1)).ravel()
    A = sp.diags(1 / rows) @ J
    cols = np.asarray(abs(A).sum(0)).ravel()
    A = (A @ sp.diags(1 / cols)).tocsc()
    lu = splu(A)
    inverse = LinearOperator(A.shape, matvec=lu.solve, matmat=lu.solve,
                             rmatvec=lambda x: lu.solve(x, trans="T"),
                             rmatmat=lambda x: lu.solve(x, trans="T"), dtype=np.float64)
    # onenormest 自带随机采样，固定局部调用的 NumPy 状态以便重现；不改变调用方随机序列。
    previous = np.random.get_state()
    try:
        np.random.seed(0)
        inverse_norm = onenormest(inverse)
    finally:
        np.random.set_state(previous)
    normA = float(abs(A).sum(0).max())
    b = np.random.default_rng(0).normal(size=A.shape[0])
    q = lu.solve(b)
    defect = np.linalg.norm(A @ q - b, 1) / (normA * np.linalg.norm(q, 1) + np.linalg.norm(b, 1))
    pivots = abs(lu.U.diagonal())
    result = dict(rcond_est=float(1 / (normA * inverse_norm)), pivot_ratio=float(pivots.min() / pivots.max()),
                  linear_backward_error=float(defect))
    result["jacobian_ok"] = bool(np.isfinite(result["rcond_est"]) and result["rcond_est"] > 1e-12 and defect < 1e-11)
    return result


def interpolate_state(source, psi, T, target):
    """网格研究用的跨网格初值。"""
    xp, yp = np.meshgrid(target.x[1:-1], target.y[1:-1], indexing="ij")
    p = RegularGridInterpolator((source.x, source.y), source.psi_full(psi))(np.stack([xp, yp], -1)).ravel()
    xt, yt = np.meshgrid(target.xc, target.yc, indexing="ij")
    t = RegularGridInterpolator((source.xc, source.yc), T.reshape(len(source.dx), len(source.dy)),
                               bounds_error=False, fill_value=None)(np.stack([xt, yt], -1)).ravel()
    return p, t


def field_errors(coarse, pc, tc, fine, pf, tf):
    """细网格流函数采样到粗网格，用粗网格 M 衡量速度场差；温度用体积 L2。"""
    pr, tr = interpolate_state(fine, pf, tf, coarse)
    d = pc - pr
    return dict(psi_M_error=float(np.sqrt(d @ (coarse.Mpsi @ d) / (pr @ (coarse.Mpsi @ pr)))),
                T_L2_error=float(np.sqrt(np.sum(coarse.Mp * (tc - tr) ** 2) / np.sum(coarse.Mp * tr ** 2))))


def advance(cav, start, target, mapping, psi=None, T=None, max_step=0.125, depth=0):
    """在用户指定的参数坐标中延拓；失败时二分步长，不改变目标工况。"""
    start, target = np.asarray(start), np.asarray(target)
    steps = max(1, int(np.ceil(np.max(abs(target - start)) / max_step)))
    current = start
    total, retries = 0, 0
    for z in np.linspace(start, target, steps + 1)[1:]:
        nu, kap, coeff = mapping(z)
        p, t, st = cav.solve(nu, kap, psi, T, tol=1e-10, maxit=35, **coeff)
        total += st["iters"]
        if not st["converged"]:
            if depth >= 4:
                raise RuntimeError(f"continuation failed at {z.tolist()}: {st}")
            p, t, nested = advance(cav, current, z, mapping, psi, T, max_step=max_step / 2, depth=depth + 1)
            total += nested["total_iters"]
            retries += 1 + nested["retries"]
            st = nested
        psi, T, current = p, t, z
    return psi, T, dict(st, total_iters=total, retries=retries)
