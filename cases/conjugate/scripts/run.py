"""通道共轭传热的降阶求解（nepodg.experiment.run，空间与公共列见该模块）。闭合为线性 Galerkin 直接求解（case.Conjugate.solve），
主场误差 e 为温度增量 u 的参考域 L²；算例输出列为温度场误差 e_T、界面总热流 Q、平均界面温度 Tw、出口混合温度 Tout 的相对误差与
四个位置局部 Nu 的最大相对误差 e_Nu。role 另可取 pool（训练池）。结果 results/n<B>/<space>_<role>.csv。
用法（仓库根目录，PINNGPU）：python -m cases.conjugate.scripts.run B role [space]"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.conjugate import case
from nepodg import experiment

g = case.Conjugate()
Mh = g.ch.Mh
rel = lambda a, b: abs(a / b - 1)


def rom(y0, B, mu, c0):
    return g.solve(y0, B, mu), None, None


def outputs(u, _, ref, mu):
    u_r = ref[0]
    T, T_r = g.temperature(u), g.temperature(u_r)
    o, o_r = g.outputs(u, mu), g.outputs(u_r, mu)
    return dict(e_T=np.sqrt(Mh @ (T - T_r) ** 2 / (Mh @ T_r ** 2)), e_Q=rel(o["Q"], o_r["Q"]), e_Tw=rel(o["Tw"], o_r["Tw"]),
                e_Tout=rel(o["Tout"], o_r["Tout"]), e_Nu=np.abs(o["Nu"] / o_r["Nu"] - 1).max())


experiment.run(case, g, ("Pe", "ratio"), rom, outputs)
