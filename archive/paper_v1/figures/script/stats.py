"""论文跨算例统计：六个算例默认预算下 POD-G、SA-PODG、NE-PODG 的配对结果（方法设计 §7）。
口径：末次相对残差 res ≤ 1e-6 为解出（线性闭合均解出）；SA/NE 增益、Γ、χ_B/χ_W 与逐工况分解只用两者都解出的工况，Γ、χ 与分解另要求
两者的 e_proj 都大于 1e-9（此下 e_proj 不可靠）。逐工况分解 ln(e_W/e_B) = −½ln(1−Γ) − ln(χ_B/χ_W)，记最佳逼近项绝对值大于闭合项的工况数。
输出 paper/figures/data/：paired.csv（逐工况主场误差，nc 另含未训练网络 VS_r1m256）、summary.csv（每个算例与测试集一行）、
outputs.csv（各输出量误差的中位数 / 最大值）。用法（仓库根目录）：python paper/figures/script/stats.py"""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
DEFAULT = {"nc": 12, "graetz": 4, "conjugate": 4, "mixed": 4, "porous": 4, "bfs": 4}
METHODS = {"pod": "pod", "sa": "sapodg", "ne": "VS_vm256"}
COMMON = {"id", "nV", "nS", "nZ", "e", "e_proj", "e_real", "e_proj_pod", "converged", "iters", "res", "time", "time_basis",
          "s_min", "s_med", "n_s01", "n_s001"}
EPS, RES = 1e-9, 1e-6
out = root / "paper" / "figures" / "data"
out.mkdir(parents=True, exist_ok=True)
paired, summary, outputs = [], [], []
for case, b in DEFAULT.items():
    for role in ("interp", "extrap"):
        d = {k: pd.read_csv(root / "cases" / case / "results" / f"n{b}" / f"{m}_{role}.csv").set_index("id")
             for k, m in METHODS.items()}
        if case == "nc":
            d["r1"] = pd.read_csv(root / "cases" / "nc" / "results" / f"n{b}" / f"VS_r1m256_{role}.csv").set_index("id")
        ids = d["sa"].index
        p = pd.DataFrame({"case": case, "role": role, "id": ids})
        for k, t in d.items():
            t = t.loc[ids]
            p[f"e_{k}"], p[f"ep_{k}"] = t.e.values, t.e_proj.values
            p[f"solved_{k}"] = (t.res.values <= RES) if "res" in t else True
            p[f"time_{k}"] = t.time.values
        paired.append(p)
        ok = p.solved_sa & p.solved_ne
        q = p[ok]
        u = q[(q.ep_sa > EPS) & (q.ep_ne > EPS)]
        gam = 1 - (u.ep_ne / u.ep_sa) ** 2
        chi = (u.e_ne / u.ep_ne) / (u.e_sa / u.ep_sa)
        dom = (np.abs(-0.5 * np.log(1 - gam)) > np.abs(np.log(chi))).sum()
        ratio = q.e_sa / q.e_ne
        t = d["ne"].loc[ids]
        row = dict(case=case, role=role, n=len(p), budget=f"N{b}", nV=int(t.nV.iloc[0]), nS=int(t.nS.iloc[0]),
                   nZ_med=float(t.nZ.median()), unsolved_pod=int((~p.solved_pod).sum()), unsolved_sa=int((~p.solved_sa).sum()),
                   unsolved_ne=int((~p.solved_ne).sum()), n_pair=len(q), n_mech=len(u))
        for k in ("pod", "sa", "ne"):
            s = p[p[f"solved_{k}"]]
            row[f"e_{k}_med"], row[f"e_{k}_max"] = s[f"e_{k}"].median(), s[f"e_{k}"].max()
            row[f"time_{k}_med"] = p[f"time_{k}"].median()
        row.update(ratio_med=ratio.median(), ratio_min=ratio.min(), n_ne_worse=int((ratio < 1).sum()),
                   gamma_med=gam.median(), gamma_min=gam.min(), chi_med=chi.median(), chi_min=chi.min(), chi_max=chi.max(),
                   n_gamma_dominant=int(dom))
        summary.append(row)
        for k in ("pod", "sa", "ne"):
            t = d[k].loc[ids]
            t = t[p[f"solved_{k}"].values]
            for col in [c for c in t.columns if c not in COMMON and c.startswith("e_")]:
                outputs.append(dict(case=case, role=role, method=k, quantity=col, median=t[col].median(), max=t[col].max()))
pd.concat(paired).to_csv(out / "paired.csv", index=False)
S = pd.DataFrame(summary)
S.to_csv(out / "summary.csv", index=False)
pd.DataFrame(outputs).to_csv(out / "outputs.csv", index=False)
pct = lambda x: f"{100 * x:.3g}"
for r in S.itertuples():
    print(f"{r.case:9s} {r.role:6s} n={r.n:3d} SA {pct(r.e_sa_med)}/{pct(r.e_sa_max)} NE {pct(r.e_ne_med)}/{pct(r.e_ne_max)} "
          f"ratio {r.ratio_med:.3g} (min {r.ratio_min:.3g}, NE worse {r.n_ne_worse}) Γ {r.gamma_med:.3g} χ {r.chi_med:.3g} "
          f"Γ-dominant {r.n_gamma_dominant}/{r.n_mech} unsolved SA {r.unsolved_sa} NE {r.unsolved_ne} "
          f"time SA/NE {1e3 * r.time_sa_med:.3g}/{1e3 * r.time_ne_med:.3g} ms")
print(f"→ {out}")
