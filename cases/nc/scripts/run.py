"""方腔自然对流的降阶求解（nepodg.experiment.run，空间与公共列见该模块）。温度由闭合在全网格上求得；算例输出列为温度误差 e_T（体积 L²）
与平均 Nu 的相对误差 e_Nu。结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.run B role [space]"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.nc import case
from nepodg import closure, experiment

g = case.NC()
Mp = g.cav.Mp


def rom(y0, B, mu, c0):
    return closure.solve(g.reduced, y0, B, mu, c0)


def outputs(y, T, ref, mu):
    T, T_r = T.cpu().numpy(), ref[1]
    nu, nu_r = g.cav.nusselt(T)[2], g.cav.nusselt(T_r)[2]
    return dict(e_T=np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2)), e_Nu=abs(nu / nu_r - 1))


experiment.run(case, g, ("Ra", "Pr"), rom, outputs)
