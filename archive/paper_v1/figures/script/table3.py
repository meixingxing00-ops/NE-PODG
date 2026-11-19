"""论文 Table 3：六个算例默认预算下三种方法代表性传热量的相对误差（%，中位数 / 最大值），内插与外推。
数据 paper/figures/data/outputs.csv（stats.py，只含该空间内解出的工况）。输出 paper/figures/data/table3.tex（tabular，由正文 \\input）。
用法（仓库根目录）：python paper/figures/script/table3.py"""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[3]
O = pd.read_csv(root / "paper" / "figures" / "data" / "outputs.csv")
S = pd.read_csv(root / "paper" / "figures" / "data" / "summary.csv")
ROWS = [("nc", "Natural convection", [("e_Nu", r"$\overline{Nu}$")]),
        ("graetz", "Couette--Graetz", [("e_Tav", r"$T_{\mathrm{av}}$"), ("e_Nu", r"local $Nu$")]),
        ("conjugate", "Conjugate channel", [("e_Q", r"$Q$"), ("e_Nu", r"local $Nu$")]),
        ("mixed", "Mixed convection", [("e_Nu", r"$\overline{Nu}$")]),
        ("porous", "Porous cavity", [("e_Nu", r"$\overline{Nu}$")]),
        ("bfs", "Backward-facing step", [("e_xr", r"$x_r$"), ("e_Nuavg", r"$\overline{Nu}$")])]


def fmt(x):
    x = 100 * x
    if x == 0:
        return "0"
    if x < 1e-3 or x >= 1e3:
        m, e = f"{x:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:#.3g}".rstrip(".")


METH = (("pod", "POD-G"), ("sa", "SA-PODG"), ("ne", "NE-PODG"))
lines = [r"\begin{tabular}{lllcccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Interpolation} & \multicolumn{2}{c}{Extrapolation} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Problem & Output & Method & median & max & median & max \\"]
for case, name, qs in ROWS:
    lines.append(r"\midrule")
    for j, (q, lab) in enumerate(qs):
        if j:
            lines.append(r"\cmidrule(lr){2-7}")
        for k, (m, mlab) in enumerate(METH):
            cells = []
            for role in ("interp", "extrap"):
                r = O[(O.case == case) & (O.role == role) & (O.method == m) & (O.quantity == q)].iloc[0]
                cells += [fmt(r["median"]), fmt(r["max"])]
            head = (name if j == 0 and k == 0 else "") + " & " + (lab if k == 0 else "")
            lines.append(f"{head} & {mlab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
out = root / "paper" / "figures" / "data" / "table3.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
bad = S[(S.unsolved_sa > 0) | (S.unsolved_ne > 0) | (S.unsolved_pod > 0)]
print("\n".join(lines))
print("unsolved:", bad[["case", "role", "unsolved_pod", "unsolved_sa", "unsolved_ne"]].to_dict("records"))
print(f"→ {out}")
