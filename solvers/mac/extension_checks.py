"""通用系数扩展的回归与差分验证。仓库根目录，PINNGPU：

python -X utf8 -m solvers.mac.extension_checks regression [--comsol-data PATH]
python -X utf8 -m solvers.mac.extension_checks derivatives
python -X utf8 -m solvers.mac.extension_checks iwatsu
python -X utf8 -m solvers.mac.extension_checks brinkman

只存验证标量，不生成快照。默认回归基线是扩展前的 38d1e0f。
"""
import argparse
import contextlib
import io
import json
import re
import subprocess
import types
from pathlib import Path

import numpy as np
import torch

from solvers.mac import checks
from solvers.mac.gpu import Closure
from solvers.mac.mac import Cavity, nodes

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), 1e-30))


def legacy_module(path, ref):
    source = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    mod = types.ModuleType("legacy_" + Path(path).stem)
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def regression(ref, comsol_data):
    old = legacy_module("solvers/mac/mac.py", ref)
    oldgpu = legacy_module("solvers/mac/gpu.py", ref)
    rng = np.random.default_rng(90210)
    rows = []
    for thermal in (False, True):
        for s in (0.0, 2.0):
            x, y = nodes(24, s), nodes(20, s)
            args = dict(lid=lambda x: np.ones_like(x), thermal=thermal)
            a, b = old.Cavity(x, y, **args), Cavity(x, y, **args)
            psi = rng.normal(size=a.npsi) * 0.01
            T = rng.normal(size=a.nT) if thermal else None
            nu, kap = 0.03, 0.02
            r = dict(thermal=thermal, s=s)
            r["residual"] = max(relative(v, w) for v, w in zip(a.residual(psi, T, nu, kap), b.residual(psi, T, nu, kap)) if v is not None)
            r["jacobian"] = relative(a.jac_full(psi, T, nu, kap).toarray(), b.jac_full(psi, T, nu, kap).toarray())
            r["same_state_sensitivity"] = relative(a.sensitivity(psi, T, nu, kap), b.sensitivity(psi, T, nu, kap))
            D = rng.normal(size=(a.npsi, 3))
            if thermal:
                ta, da = a.temperature(psi, kap, a.C @ D)
                tb, db = b.temperature(psi, kap, b.C @ D)
                r["temperature"], r["aux_direction"] = relative(ta, tb), relative(da, db)
            qa = a.solve(nu, kap, tol=1e-11)
            qb = b.solve(nu, kap, tol=1e-11)
            assert qa[2]["converged"] and qb[2]["converged"]
            r["steady_state"] = relative(qa[0], qb[0])
            r["steady_sensitivity"] = relative(a.sensitivity(*qa[:2], nu, kap), b.sensitivity(*qb[:2], nu, kap))
            p, basis = (torch.as_tensor(z, dtype=torch.float64, device="cuda") for z in (psi, D))
            ca, cb = oldgpu.Closure(a), Closure(b)
            ga, ja, ta = ca.reduced(p, basis, nu, kap)
            gb, jb, tb = cb.reduced(p, basis, nu, kap)
            r["gpu_residual"] = relative(ga.cpu().numpy(), gb.cpu().numpy())
            r["gpu_jacobian"] = relative(ja.cpu().numpy(), jb.cpu().numpy())
            if thermal:
                r["gpu_temperature"] = relative(ta.cpu().numpy(), tb.cpu().numpy())
            r["metric"] = relative(ca.whiten(basis).cpu().numpy(), cb.whiten(basis).cpu().numpy())
            assert max(v for k, v in r.items() if k not in ("thermal", "s")) < 3e-12, r
            rows.append(r)

    suite = [("ops", checks.ops, ()), ("lid", checks.lid, (128, 1.5)),
             ("nc", checks.nc, (128, 2.5)), ("sens", checks.sens, (64, 2.0))]
    if comsol_data:
        suite.append(("comsol", checks.comsol, (128, 2.5, comsol_data)))
    logs, errors = [], {}
    try:
        for name, fun, args in suite:
            outputs = []
            for cls in (old.Cavity, Cavity):
                checks.Cavity = cls
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    fun(*args)
                outputs.append(out.getvalue())
            logs.extend([f"--- {name}: {ref} ---\n{outputs[0]}", f"--- {name}: current ---\n{outputs[1]}"])
            # 耗时从数值比较中移除；保留所有原检查打印的物理量与误差。
            numbers = lambda text: np.array([float(z) for z in re.findall(r"(?<![\w])[-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?", re.sub(r"\d+\.\d+s", "time", text))])
            u, v = map(numbers, outputs)
            assert u.shape == v.shape
            errors[name] = float(np.max(np.abs(u - v) / np.maximum(1, np.abs(u))))
            assert errors[name] < 1e-7, (name, errors[name])
            print(f"legacy {name}: {errors[name]:.3e}", flush=True)
    finally:
        checks.Cavity = Cavity
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_checks.txt").write_text("\n".join(logs), encoding="utf-8")
    save("regression", dict(reference=ref, random_state_comparison=rows, original_checks=errors,
                            comsol_data=comsol_data, comsol_note="not run without original data" if not comsol_data else "run"))


def derivatives():
    rng, rows = np.random.default_rng(196), []
    for thermal, inertia, axis in ((False, 1.0, 0), (True, 1.0, 0), (True, 1.0, 1), (True, 0.0, 0)):
        cav = Cavity(nodes(24, 1.5), nodes(20, 1.5), lid=lambda x: np.ones_like(x), thermal=thermal,
                     inertia=inertia, thermal_axis=axis)
        nu, kap, buoyancy, drag = 0.03, 0.04, 1.3, 0.7
        kw = dict(buoyancy=buoyancy, drag=drag)
        psi = rng.normal(size=cav.npsi) * 0.01
        T = rng.normal(size=cav.nT) if thermal else None
        d = rng.normal(size=cav.npsi + (cav.nT if thermal else 0))
        def f(p, t, **coeff):
            z = cav.residual(p, t, nu, kap, **coeff)
            return np.concatenate(z) if thermal else z[0]
        h = 1e-6
        pp, tt = d[:cav.npsi], d[cav.npsi:] if thermal else None
        fd = (f(psi + h * pp, T + h * tt if thermal else None, **kw) -
              f(psi - h * pp, T - h * tt if thermal else None, **kw)) / (2 * h)
        row = dict(thermal=thermal, inertia=inertia, thermal_axis=axis,
                   state_jacobian=relative(cav.jac_full(psi, T, nu, kap, **kw) @ d, fd))
        names = ("buoyancy", "drag")
        partials = cav.parameter_derivatives(psi, T, parameters=names)
        for name in names:
            plus, minus = dict(kw), dict(kw)
            plus[name] += h
            minus[name] -= h
            fd = (f(psi, T, **plus) - f(psi, T, **minus)) / (2 * h)
            row[f"partial_{name}"] = relative(partials[name], fd)
        p, t, st = cav.solve(nu, kap, tol=1e-11, **kw)
        assert st["converged"], st
        sens = cav.sensitivity(p, t, nu, kap, parameters=names, **kw)
        J = cav.jac_full(p, t, nu, kap, **kw)
        rhs = cav.parameter_derivatives(p, t, parameters=names)
        for name in names:
            plus, minus = dict(kw), dict(kw)
            hs = 1e-4
            plus[name] += hs
            minus[name] -= hs
            qp, qm = [cav.solve(nu, kap, p, t, tol=1e-12, **a) for a in (plus, minus)]
            assert qp[2]["converged"] and qm[2]["converged"]
            fp = np.concatenate(qp[:2]) if thermal else qp[0]
            fm = np.concatenate(qm[:2]) if thermal else qm[0]
            fd = (fp - fm) / (2 * hs)
            s = sens[name]
            M = cav.Mpsi
            error = s[:cav.npsi] - fd[:cav.npsi]
            denom = fd[:cav.npsi] @ (M @ fd[:cav.npsi])
            row[f"sensitivity_{name}_psi_M"] = float(np.sqrt(error @ (M @ error) / denom)) if denom > 1e-28 else float(np.linalg.norm(error))
            if thermal:
                row[f"sensitivity_{name}_T"] = relative(s[cav.npsi:], fd[cav.npsi:])
            row[f"linear_defect_{name}"] = float(np.linalg.norm(J @ s + rhs[name]) / max(np.linalg.norm(rhs[name]), 1e-30))
        D = rng.normal(size=(cav.npsi, 3)) * 0.01
        cl = Closure(cav)
        tp, tb = [torch.as_tensor(z, dtype=torch.float64, device="cuda") for z in (p, D)]
        g, G, Tg = cl.reduced(tp, tb, nu, kap, **kw)
        if thermal:
            ta, Sa = cav.temperature(p, kap, cav.C @ D)
            j = cav.jacobian(p, ta, nu, kap, **kw)
            jv = j[0] @ D + j[1] @ Sa
            ra = cav.residual(p, ta, nu, kap, **kw)[0]
            row["gpu_temperature"] = relative(Tg.cpu().numpy(), ta)
        else:
            ra, jv = cav.residual(p, None, nu, **kw)[0], cav.jac_full(p, None, nu, **kw) @ D
        row["gpu_residual_absolute"] = float(np.linalg.norm(g.cpu().numpy() - D.T @ ra))
        row["gpu_jacobian"] = relative(G.cpu().numpy(), D.T @ jv)
        h = 1e-5
        gd = [(cl.reduced(tp + h * tb[:, k], tb, nu, kap, **kw)[0] -
               cl.reduced(tp - h * tb[:, k], tb, nu, kap, **kw)[0]).cpu().numpy() / (2 * h) for k in range(3)]
        row["gpu_total_jacobian_fd"] = relative(G.cpu().numpy(), np.stack(gd, 1))
        assert row["state_jacobian"] < 1e-7 and row["gpu_jacobian"] < 1e-10
        assert row["gpu_total_jacobian_fd"] < 1e-6
        for name in names:
            assert row[f"partial_{name}"] < 1e-6
            assert row[f"sensitivity_{name}_psi_M"] < 1e-5
            assert row[f"linear_defect_{name}"] < 1e-9
        rows.append(row)
    save("derivatives", rows)


def iwatsu():
    # 文献热边界：上热下冷，侧壁绝热；恒速顶盖向 +x；Pr=.71，Ri=Gr/Re²。
    reference = {100: (1.94, 1.34, 1.02), 400: (3.84, 3.62, 1.22), 1000: (6.33, 6.29, 1.77)}
    rows = []
    for n in (64, 128, 256):
        x = nodes(n, 2.0)
        cav = Cavity(x, x, lid=lambda x: np.ones_like(x), thermal=True, thermal_axis=1)
        seed_p = seed_t = None
        last_re = 20
        for re in reference:
            # 先在 Gr=100 沿 Re 延拓，随后固定 Re 扫描 Gr。
            for rr in np.geomspace(last_re, re, max(2, int(np.ceil(np.log10(re / last_re) / 0.125)) + 1)):
                seed_p, seed_t, st = cav.solve(1 / rr, 1 / (rr * 0.71), seed_p, seed_t,
                                              buoyancy=100 / rr ** 2, tol=1e-10, maxit=60)
                assert st["converged"], (n, rr, st)
            p, t, last_re = seed_p.copy(), seed_t.copy(), re
            for lg in np.arange(2.0, 6.01, 0.125):
                gr = 10 ** lg
                p, t, st = cav.solve(1 / re, 1 / (re * 0.71), p, t, buoyancy=gr / re ** 2, tol=1e-10, maxit=60)
                assert st["converged"], (n, re, gr, st)
                if lg not in (2.0, 4.0, 6.0):
                    continue
                ref = reference[re][(int(lg) - 2) // 2]
                _, _, nc, nh = cav.nusselt(t)
                row = dict(N=n, s=2.0, Re=re, Gr=gr, Ri=gr / re ** 2, Nu_cold=nc, Nu_hot=nh,
                           reference=ref, error_percent=100 * (nh / ref - 1), **st)
                rows.append(row)
                save("iwatsu", rows)
                print(row, flush=True)
    save("iwatsu", rows)


def brinkman():
    """Lauriat & Prasad (1987), Table 3, A=1, transport omitted, viscosity ratio=1.

    Their Ra*=Ra*Da. A horizontal reflection changes left-hot to right-hot
    without changing mean Nu. Only scalar verification results are saved.
    """
    from solvers.mac.study import advance, interpolate_state, scalars

    refs = {100: (3.02, 2.84, 2.41, 1.70), 500: (8.41, 7.35, 5.42, 3.30)}

    def mapping(z):
        ra, da = 10.0 ** np.asarray(z)
        nu = np.sqrt(.71 / ra)
        return nu, 1 / np.sqrt(.71 * ra), dict(drag=nu / da)

    rows, states = [], {}
    for n in (64, 128, 256):
        x = nodes(n, 2.5)
        cav = Cavity(x, x, thermal=True, inertia=0)
        for rad, reference in refs.items():
            for da, ref in zip((1e-5, 1e-4, 1e-3, 1e-2), reference):
                ra = rad / da
                z = np.log10([ra, da])
                nu, kap, kw = mapping(z)
                if n == 64:
                    p, t, st = advance(cav, [3, z[1]], z, mapping)
                else:
                    p, t = interpolate_state(*states[n // 2, rad, da], cav)
                    p, t, st = cav.solve(nu, kap, p, t, tol=1e-10, **kw)
                assert st["converged"], (n, rad, da, st)
                states[n, rad, da] = cav, p, t
                row = dict(N=n, s=2.5, Ra=ra, Da=da, Ra_D=rad, reference=ref,
                           **scalars(cav, p, t), **st)
                row["error_percent"] = 100 * (row["Nu_cold"] / ref - 1)
                rows.append(row)
                print(f"Brinkman N={n} Ra_D={rad} Da={da:g} Nu={row['Nu_cold']:.9g}", flush=True)
        save("brinkman", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("regression", "derivatives", "iwatsu", "brinkman"))
    parser.add_argument("--reference", default="38d1e0f")
    parser.add_argument("--comsol-data")
    args = parser.parse_args()
    if args.task == "regression":
        regression(args.reference, args.comsol_data)
    else:
        globals()[args.task]()
