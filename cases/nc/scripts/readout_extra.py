"""N12：当前读出方向与额外互补空间的余量捕获，六个网络、127 个测试工况。
复用现行互补空间与秩阈值，只做投影；输出 results/n12/readout_extra*.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.readout_extra
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cases.nc import case
from nepodg import candidates, space

MODELS = ("vm256", "v1m256", "v2m256", "r1m256", "r2m256", "r3m256")
OUT = case.ROOT / "results" / "n12" / "readout_extra.csv"


@torch.no_grad()
def run(out=OUT, models=MODELS, limit=None):
    g = case.NC()
    rows = case.cases()
    ids = [c["id"] for c in rows if c["n12"] == "1"]
    pod = space.POD(torch.stack([g.load(i)[0] for i in ids], 1), g.whiten)
    base, nS, _ = space.sensitivity_space(pod, g.sens(ids).reshape(g.n, -1), g.whiten)
    tests = [c for c in rows if c["role"] in ("interp", "extrap")][:limit]
    ref = {}
    for c in tests:
        q, _, mu = g.load(c["id"])
        rho = base.project(g.whiten(q - pod.mean)[:, None])[:, 0]
        ref[c["id"]] = (mu, rho, torch.linalg.norm(g.whiten(q - g.lift)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records, t0 = [], time.perf_counter()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = None
        for name in models:
            trained, seed, m, layers, width = candidates.kind(name)
            if trained:
                net = candidates.load(case.MODELS / name / "n12.pt", g.X.device)
            else:
                torch.manual_seed(seed)
                net = candidates.Net(m=m, layers=layers, width=width).to(g.X.device).eval()
            for j, c in enumerate(tests, 1):
                mu, rho, scale = ref[c["id"]]
                eta = case.eta(mu)
                Phi = candidates.directions(net, g.X, eta, g.admissible)
                _, Zw, angles = space.complement(Phi, g.whiten, base)
                s = torch.linalg.svdvals(g.whiten(Phi).cpu())
                sr = s / s[0]
                e = torch.as_tensor(eta, dtype=torch.float32, device=g.X.device)
                qhat = Phi @ net.readout(e).double()
                qhatw = g.whiten(qhat)
                a, b = Zw.T @ rho, Zw.T @ qhatw
                out_norm, rho2 = torch.linalg.norm(b), torch.dot(rho, rho)
                nout = int(out_norm > 0)
                extra = a.clone()
                captured_out = torch.zeros((), dtype=torch.float64, device=g.X.device)
                if nout:
                    direction = b / out_norm
                    captured_out = torch.dot(direction, a)
                    extra = a - direction * captured_out
                full = float(torch.dot(a, a) / rho2)
                gout = float(captured_out ** 2 / rho2)
                gextra = float(torch.dot(extra, extra) / rho2)
                row = dict(model=name, group="trained" if trained else "random", seed=seed,
                           case_id=c["id"], role=c["role"], Ra=mu[0], Pr=mu[1], m=m,
                           nV=pod.V.shape[1], nS=nS, nZ=Zw.shape[1], n_out=nout, n_extra=Zw.shape[1]-nout,
                           n_candidate=int((sr > 1e-8).sum()), n_eff_1e4=int((sr > 1e-4).sum()),
                           n_eff_1e6=int((sr > 1e-6).sum()), s_min=float(angles[Zw.shape[1]-1]),
                           eproj_W=float(rho2.sqrt()/scale), out_norm=float(out_norm),
                           out_relative_norm=float(out_norm/torch.linalg.norm(qhatw)),
                           Gamma_full=full, Gamma_out=gout, Gamma_extra=gextra,
                           extra_fraction=gextra/full if full > 0 else np.nan,
                           decomposition_error=abs(full-gout-gextra))
                records.append(row)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if j % 25 == 0 or j == len(tests):
                    fh.flush()
                    print(f"{name} {j}/{len(tests)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    data = pd.DataFrame(records)
    d = data[data.eproj_W > 1e-9]
    pooled = d.assign(model="all", seed=-1)
    summary = []
    for (group, model, seed, role), x in pd.concat([d, pooled]).groupby(["group", "model", "seed", "role"]):
        row = dict(group=group, model=model, seed=seed, role=role, n=len(x),
                   nZ_min=x.nZ.min(), nZ_med=x.nZ.median(), nZ_max=x.nZ.max(),
                   n_candidate_min=x.n_candidate.min(), n_candidate_max=x.n_candidate.max(),
                   n_eff_1e4_med=x.n_eff_1e4.median(), n_eff_1e6_med=x.n_eff_1e6.median(),
                   n_out_zero=int((x.n_out == 0).sum()), out_relative_norm_min=x.out_relative_norm.min(),
                   decomposition_error_max=x.decomposition_error.max())
        for col in ("Gamma_full", "Gamma_out", "Gamma_extra", "extra_fraction"):
            for label, q in (("q25", .25), ("med", .5), ("q75", .75)):
                row[f"{col}_{label}"] = x[col].quantile(q)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary[["group", "model", "role", "n", "nZ_min", "nZ_max", "Gamma_full_med", "Gamma_out_med", "Gamma_extra_med", "extra_fraction_med"]].to_string(index=False), flush=True)
    print(f"decomposition error max={data.decomposition_error.max():.3e}; wrote {len(data)} rows: {out}", flush=True)


if __name__ == "__main__":
    run()
