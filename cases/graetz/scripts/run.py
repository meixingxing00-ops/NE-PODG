"""Couette–Graetz 的降阶求解（nepodg.experiment.run，空间与公共列见该模块）。闭合为线性 Galerkin 直接求解（case.Graetz.solve）；
算例输出列为 T_av 的相对误差 e_Tav 与局部 Nu 沿受热壁的最大相对误差 e_Nu。role 另可取 pool（训练池）。
结果 results/n<B>/<space>_<role>.csv。用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.run B role [space]"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.graetz import case
from nepodg import experiment

g = case.Graetz()


def rom(y0, B, mu, c0):
    return g.solve(y0, B, mu), None, None


def outputs(T, _, ref, mu):
    T_r = ref[0]
    return dict(e_Tav=abs(g.tav(T, mu) / g.tav(T_r, mu) - 1), e_Nu=np.abs(g.nusselt(T, mu) / g.nusselt(T_r, mu) - 1).max())


experiment.run(case, g, ("L", "Pe"), rom, outputs)
