"""train_traj.py 的结果按步数与区段汇总。用法（仓库根目录）：python archive/cases/nc/mechanism/traj_summary.py B 结果.csv"""
import sys, numpy as np, pandas as pd
R = "cases/nc/results"
B, f = sys.argv[1], sys.argv[2]
d = pd.read_csv(f)
sa = pd.concat([pd.read_csv(f"{R}/n{B}/sapodg_{r}.csv") for r in ("interp", "extrap")]).set_index("id")
vs = pd.concat([pd.read_csv(f"{R}/n{B}/VS_v_{r}.csv") for r in ("interp", "extrap")]).set_index("id")
d["gain"] = 1 / np.sqrt(1 - d.Gamma)                      # e_proj,W / e_proj,B
d["e_sa"] = d.id.map(sa.e); d["e_vs"] = d.id.map(vs.e)
REG = [("I lo", lambda x: x.id.str.startswith("I") & (x.Ra > 1e3) & (x.Ra <= 3.16e4)), ("I hi", lambda x: x.id.str.startswith("I") & (x.Ra > 3.16e4)),
       ("E4.22", lambda x: (x.Ra / 4.22e5 - 1).abs() < 1e-2), ("E5.62", lambda x: (x.Ra / 5.62e5 - 1).abs() < 1e-2), ("E1e6", lambda x: (x.Ra / 1e6 - 1).abs() < 1e-2)]
print("step loss | n>1e-2 n>1e-4 n>1e-6 n_s01 (中位) | Γ 中位 (I lo, I hi, E4.22, E5.62, E1e6) | e_projW/e_projB 中位 (同)")
for st, x in d.groupby("step"):
    print(f"{st:6d} {x.loss.iloc[0]:.2e} | {x.n2.median():.0f} {x.n4.median():.0f} {x.n6.median():.0f} {x.n_s01.median():.0f} | "
          + " ".join(f"{x[s(x)].Gamma.median():.4f}" for _, s in REG) + " | " + " ".join(f"{x[s(x)].gain.median():.2f}" for _, s in REG))
print("\n闭合：e / e_SA 中位 (同区段) | e/e_proj 中位 | e/e_VS_v(提交) 中位")
for st, x in d[d.e.notna()].groupby("step"):
    print(f"{st:6d} " + " ".join(f"{(x[s(x)].e / x[s(x)].e_sa).median():.3f}" for _, s in REG) + f" | {(x.e / x.e_proj).median():.2f} | {(x.e / x.e_vs).median():.4f}")
