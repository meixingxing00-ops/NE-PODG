"""论文 Fig. 6：graetz（E014）、conjugate（E007）、bfs（E017）的场与壁面传热，均为 N4 下 SA-PODG 误差最接近外推中位数的外推工况。
每行一个算例：SA-PODG 与 NE-PODG 的有符号误差（温度 (T − T*)/max|T*|；bfs 为速度大小 (|U| − |U*|)/max|U*|），叠加参考解的等温线或流线，
同一行两图色标相同；末行：graetz 受热壁局部 Nu 与 conjugate 界面热流 q 的相对误差，灰线为参考值（右轴）；bfs 受热壁剪切 τ_w 沿 x/S 的分布，
小图为主再附着点附近的放大，圆点为各方法的再附着位置。
数据 cases/<算例>/results/n4/fields_<编号>.npz（各算例 scripts/fields.py）。
用法（仓库根目录）：python paper/figures/script/fig6_fields.py [输出路径，默认 paper/figures/Fig_06.pdf；.png 用于预览]"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                            "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
                            "ytick.direction": "in", "pdf.fonttype": 42})
cmap = plt.get_cmap("RdBu_r")
div = matplotlib.colors.ListedColormap(cmap(np.linspace(0.08, 0.92, 256)))
SA, NE = cmap(0.12), cmap(0.88)
root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper" / "figures" / "Fig_06.pdf"
D = {c: np.load(root / "cases" / c / "results" / "n4" / f"fields_{i}.npz")
     for c, i in (("graetz", "E014"), ("conjugate", "E007"), ("bfs", "E017"))}

mm = 1 / 25.4
fig = plt.figure(figsize=(190 * mm, 132 * mm))
gs = fig.add_gridspec(4, 3, height_ratios=[20.5, 21.5, 13.5, 33], width_ratios=[1, 1, 0.035], left=0.055, right=0.94,
                      bottom=0.1, top=0.965, hspace=0.38, wspace=0.08)
rows = []
# graetz：温度，x ∈ [0, 1 + L]
d = D["graetz"]
rows.append(dict(name=rf"Couette–Graetz, $L={d['L']:.3g}$, $Pe={d['Pe']:.3g}$", x=d["x"], y=d["y"], ref=d["T_fom"],
                 err={k: (d[f"T_{k}"] - d["T_fom"]) / np.abs(d["T_fom"]).max() for k in ("sapodg", "VS_vm256")},
                 lines=(d["T_fom"], np.linspace(0.1, 0.9, 5) * d["T_fom"].max(), "k"), xlim=None,
                 label=r"$(T-T^\star)/\max|T^\star|$"))
d = D["conjugate"]
rows.append(dict(name=rf"Conjugate channel, $Pe={d['Pe']:g}$, $k_s/k_f={d['ratio']:g}$", x=d["x"], y=d["y"], ref=d["T_fom"],
                 err={k: (d[f"T_{k}"] - d["T_fom"]) / np.abs(d["T_fom"]).max() for k in ("sapodg", "VS_vm256")},
                 lines=(d["T_fom"], np.linspace(0.1, 0.9, 5) * d["T_fom"].max(), "k"), xlim=None, iface=True,
                 label=r"$(T-T^\star)/\max|T^\star|$"))
d = D["bfs"]
pr = d["psi_fom"]
rows.append(dict(name=rf"Backward-facing step, $Re={d['Re']:.3g}$, $ER={d['ER']:.3g}$", x=d["x"], y=d["y"], ref=d["U_fom"],
                 err={k: (d[f"U_{k}"] - d["U_fom"]) / np.nanmax(d["U_fom"]) for k in ("sapodg", "VS_vm256")},
                 lines=(pr, np.r_[np.nanmin(pr) * np.array([0.6, 0.2]), np.linspace(0.1, 0.9, 5) * np.nanmax(pr)], "k"),
                 xlim=(-1, 12), label=r"$(|\mathbf{U}|-|\mathbf{U}^\star|)/\max|\mathbf{U}^\star|$"))
for r, row in enumerate(rows):
    lim = np.nanmax(np.abs(row["err"]["sapodg"]))
    norm = matplotlib.colors.Normalize(-lim, lim)
    for j, (k, lab) in enumerate((("sapodg", "SA-PODG"), ("VS_vm256", "NE-PODG"))):
        ax = fig.add_subplot(gs[r, j])
        X, Y = row["x"], row["y"]
        im = ax.pcolormesh(X, Y, row["err"][k].T, cmap=div, norm=norm, shading="flat", rasterized=True)
        F, lev, col = row["lines"]
        if F.shape == (len(X), len(Y)):
            ax.contour(X, Y, F.T, levels=np.sort(lev), colors=col, linewidths=0.45)
        else:
            xc, yc = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
            ax.contour(xc, yc, F.T, levels=np.sort(lev), colors=col, linewidths=0.45)
        if row.get("iface"):
            ax.axhline(0, color="0.3", lw=0.6, ls="--")
        if row["xlim"]:
            ax.set_xlim(*row["xlim"])
        ax.set_aspect("equal")
        ax.tick_params(length=2, pad=1.5)
        e = np.sqrt(np.nanmean(row["err"][k] ** 2))
        ax.set_title(f"{lab}" + (f" — {row['name']}" if j == 0 else ""), fontsize=7.5, loc="left", pad=2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[r, 2]))
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_fontsize(6.5)
    cb.set_label(row["label"], fontsize=7)
# 末行：沿壁面的相对误差
sub = gs[3, :].subgridspec(1, 3, wspace=0.62)
curves = [("graetz", r"$x$", "xw", "Nu", r"local $Nu$"), ("conjugate", r"$x$", "xc", "q", r"interface $q$")]
for j, (c, xl, xk, qk, ql) in enumerate(curves):
    d = D[c]
    ax = fig.add_subplot(sub[0, j])
    x = d[xk]
    ref = d[f"{qk}_fom"]
    m = np.isfinite(ref) & (np.abs(ref) > 0)
    for k, col, ls in (("sapodg", SA, "-"), ("VS_vm256", NE, "-")):
        ax.plot(x[m], np.abs(d[f"{qk}_{k}"][m] / ref[m] - 1), color=col, lw=0.9, ls=ls)
    ax.set_yscale("log")
    ax.set_xlabel(xl, labelpad=1)
    ax.set_ylabel(f"relative error of {ql}", labelpad=1, fontsize=7)
    ax.tick_params(which="both", length=2, pad=1.5)
    tw = ax.twinx()
    tw.plot(x[m], ref[m], color="0.6", lw=0.8)
    tw.set_ylabel(ql, labelpad=1, fontsize=7, color="0.45")
    tw.tick_params(length=2, pad=1.5, labelsize=6.5, colors="0.45")
    ax.set_zorder(tw.get_zorder() + 1)
    ax.patch.set_visible(False)
# bfs：受热壁（下壁）剪切与主再附着点；x 以台阶高度 S 无量纲化
d = D["bfs"]
S = float(d["ER"]) - 1
xs = d["xs"] / S


def reattach(tau):
    """与 openmac recirculation 相同的规则：最小剪切下游第一个由负到正的零点。"""
    i0 = int(np.argmin(tau))
    k = i0 + int(np.flatnonzero((tau[i0:-1] < 0) & (tau[i0 + 1:] >= 0))[0])
    return xs[k] - tau[k] * (xs[k + 1] - xs[k]) / (tau[k + 1] - tau[k])


ax = fig.add_subplot(sub[0, 2])
m = (xs > 0) & (xs < 20)
ins = ax.inset_axes([0.5, 0.12, 0.45, 0.42])
for k, col, ls, lab in (("fom", "k", "-", "HF"), ("sapodg", SA, "-", "SA"), ("VS_vm256", NE, "--", "NE")):
    tau = d[f"shear_{k}"]
    ax.plot(xs[m], tau[m], color=col, lw=0.9, ls=ls)
    xr = reattach(tau)
    zm = (xs > xr - 0.6) & (xs < xr + 0.6)
    ins.plot(xs[zm], tau[zm], color=col, lw=0.9, ls=ls)
    ins.plot(xr, 0, "o", ms=3, mfc="white" if k == "VS_vm256" else col, mec=col, mew=0.8, zorder=3)
    print(k, f"x_r/S {xr:.3f}")
ax.axhline(0, color="0.6", lw=0.5, zorder=0)
ins.axhline(0, color="0.6", lw=0.5, zorder=0)
ins.set_xlim(9.0, 10.2)
ins.set_ylim(-2.5e-3, 2.5e-3)
ins.tick_params(length=1.5, pad=1, labelsize=5.5)
ins.set_yticks([])
ins.set_title(r"near $x_r$", fontsize=6, pad=1)
ax.indicate_inset_zoom(ins, edgecolor="0.5", lw=0.5)
ax.set_xlabel(r"$x/S$", labelpad=1)
ax.set_ylabel(r"wall shear $\tau_w$", labelpad=1, fontsize=7)
ax.tick_params(length=2, pad=1.5)
fig.legend(handles=[plt.Line2D([], [], color="k", lw=0.9, label="high-fidelity"),
                    plt.Line2D([], [], color=SA, lw=0.9, label="SA-PODG"), plt.Line2D([], [], color=NE, lw=0.9, label="NE-PODG"),
                    plt.Line2D([], [], color="0.6", lw=0.8, label="high-fidelity value (right axis)")],
           loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, fontsize=7)
fig.savefig(out, dpi=400)
print(f"→ {out}")
