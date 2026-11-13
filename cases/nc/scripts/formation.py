"""N12、vm256 的训练空间诊断：五个检查点，127 个测试工况，仅做白化坐标中的空间投影。

候选空间按 M-SVD 的前 r=8、16、32 个方向截取，再去 W 分量，另计完整空间。
汇总采用 eproj_W>1e-9；固定维数的共同样本要求各检查点 sigma_r/sigma_1>1e-4 且 nZ=r。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.formation
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
from nepodg import candidates, metrics, space

STEPS = (0, 500, 2000, 10000, 30000)
RANKS = (8, 16, 32)
OUT = case.ROOT / "results" / "n12" / "formation_vm256.csv"


@torch.no_grad()
def run(out=OUT, steps=STEPS, limit=None):
    g = case.NC()
    rows = case.cases()
    ids = [c["id"] for c in rows if c["n12"] == "1"]
    pod = space.POD(torch.stack([g.load(i)[0] for i in ids], 1), g.whiten)
    base, nS, _ = space.sensitivity_space(pod, g.sens(ids).reshape(g.n, -1), g.whiten)
    W = space.Space(None, metrics.basis(base.V, g.whiten))
    tests = [c for c in rows if c["role"] in ("interp", "extrap")][:limit]
    ref = {}
    for c in tests:
        psi, _, mu = g.load(c["id"])
        rho = W.project(g.whiten(psi - pod.mean)[:, None])[:, 0]
        scale = torch.linalg.norm(g.whiten(psi - g.lift))
        ref[c["id"]] = (mu, rho, scale)
    ckpt = case.MODELS / "vm256" / "n12_ckpt"
    losses = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(ckpt / "loss.csv"))}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    records = []
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = None
        for step in steps:
            net = candidates.load(ckpt / f"{step}.pt", g.X.device)
            for j, c in enumerate(tests, 1):
                mu, rho, scale = ref[c["id"]]
                Phi = candidates.directions(net, g.X, case.eta(mu), g.admissible)
                _, Qw, s = space._orth(Phi, g.whiten(Phi), lambda s: s / s[0] > 1e-8)
                sr = s / s[0]
                rho2 = torch.dot(rho, rho)
                row = dict(step=step, case_id=c["id"], role=c["role"], Ra=mu[0], Pr=mu[1],
                           loss=losses.get(step, np.nan), m=net.cfg["m"], nV=pod.V.shape[1], nS=nS,
                           n_eff_1e4=int((sr > 1e-4).sum()), n_eff_1e6=int((sr > 1e-6).sum()),
                           n_candidate=Qw.shape[1], eproj_W=float(rho2.sqrt() / scale))
                for r in (*RANKS, None):
                    tag = f"r{r}" if r is not None else "full"
                    Aw = W.project(Qw[:, :r])
                    _, Zw, _ = space._orth(Aw, Aw, lambda s: s > 1e-6)
                    coeff = Zw.T @ rho
                    row[f"nZ_{tag}"] = Zw.shape[1]
                    row[f"Gamma_{tag}"] = float(torch.dot(coeff, coeff) / rho2)
                    row[f"eproj_{tag}"] = float(torch.linalg.norm(rho - Zw @ coeff) / scale)
                    if r is not None:
                        row[f"sigma_rel_{tag}"] = float(sr[r - 1])
                records.append(row)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if j % 25 == 0 or j == len(tests):
                    fh.flush()
                    print(f"step={step:5d} {j:3d}/{len(tests)} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    data = pd.DataFrame(records)
    summary = []
    for role in data.role.unique():
        d = data[(data.role == role) & (data.eproj_W > 1e-9)]
        for tag in ("r8", "r16", "r32", "full"):
            selected = d if tag == "full" else d[(d[f"sigma_rel_{tag}"] > 1e-4) & (d[f"nZ_{tag}"] == int(tag[1:]))]
            common = set.intersection(*(set(selected[selected.step == st].case_id) for st in steps))
            for step in steps:
                x = d[d.step == step]
                a = selected[selected.step == step]
                b = a[a.case_id.isin(common)]
                summary.append(dict(role=role, rank=tag, step=step, loss=x.loss.iloc[0], n=len(x),
                                    n_reliable=len(a), n_common=len(b), Gamma_med=a[f"Gamma_{tag}"].median(),
                                    Gamma_common_med=b[f"Gamma_{tag}"].median(), n_eff_1e4_med=x.n_eff_1e4.median(),
                                    n_eff_1e6_med=x.n_eff_1e6.median(), nZ_med=x[f"nZ_{tag}"].median()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {len(data)} rows: {out}", flush=True)


if __name__ == "__main__":
    run()
