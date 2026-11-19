"""论文 Table 5：方腔自然对流四档快照预算下三种方法的温度误差 e_T 与平均 Nu 误差的中位数（%），内插与外推；
N12、N50 另给 NE-PODG 在三个训练种子（1234、1、2）间的范围。数据 cases/nc/results/n<B>/。
输出 paper/figures/data/table5.tex（tabular，由正文 \\input）。用法（仓库根目录）：python paper/figures/script/table5.py"""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[3]
res = root / "cases" / "nc" / "results"
load = lambda b, m, r: pd.read_csv(res / f"n{b}" / f"{m}_{r}.csv")


def fmt(x):
    x = 100 * x
    if x < 1e-3 or x >= 1e3:
        m, e = f"{x:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:#.3g}".rstrip(".")


lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Interpolation} & \multicolumn{3}{c}{Extrapolation} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Budget & Error & POD-G & SA-PODG & NE-PODG & POD-G & SA-PODG & NE-PODG \\"]
for b in (12, 20, 30, 50):
    lines.append(r"\midrule")
    for j, (q, lab) in enumerate((("e_T", r"$e_T$"), ("e_Nu", r"$\overline{Nu}$"))):
        cells = [fmt(load(b, m, r)[q].median()) for r in ("interp", "extrap") for m in ("pod", "sapodg", "VS_vm256")]
        lines.append(f"{rf'$N_{{{b}}}$' if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
lines += [r"\midrule", r"\multicolumn{8}{l}{NE-PODG, seeds 1234 / 1 / 2 (min--max of the medians)} \\"]
for b in (12, 50):
    for j, (q, lab) in enumerate((("e", r"$e$"), ("e_T", r"$e_T$"), ("e_Nu", r"$\overline{Nu}$"))):
        cells = []
        for r in ("interp", "extrap"):
            v = [load(b, f, r)[q].median() for f in ("VS_vm256", "VS_v1m256", "VS_v2m256")]
            cells.append(rf"\multicolumn{{3}}{{c}}{{{fmt(min(v))}--{fmt(max(v))}}}")
        lines.append(f"{rf'$N_{{{b}}}$' if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
out = root / "paper" / "figures" / "data" / "table5.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"→ {out}")
