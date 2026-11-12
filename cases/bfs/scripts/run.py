"""后台阶的降阶求解（nepodg.experiment.run，空间与公共列见该模块）。降阶解求出后在全网格上解一次温度；主场误差 e 为参考域速度 L²，
算例输出列为温度误差 e_T、主再附着长度 x_r、受热壁最大 Nu 与平均 Nu 的相对误差，以及 x/S = 1、2、5、10、20 处局部 Nu 的最大相对误差 e_Nu。
结果 results/n<B>/<space>_<role>.csv。用法（仓库根目录，PINNGPU）：python -m cases.bfs.scripts.run B role [space]"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

from cases.bfs import case
from nepodg import closure, experiment

g = case.BFS()
Mp = g.m.Mphat
NUX = ["Nu_1S", "Nu_2S", "Nu_5S", "Nu_10S", "Nu_20S"]
rel = lambda a, b: abs(a / b - 1)


def rom(y0, B, mu, c0):
    q, _, st = closure.solve(g.reduced, y0, B, mu, c0)
    return q, g.temperature(q, mu), st


def outputs(q, T, ref, mu):
    q_r, T_r = ref[0], ref[1]
    o, o_r = g.outputs(q, T, mu), g.outputs(q_r, T_r, mu)
    return dict(e_T=np.sqrt(Mp @ (T - T_r) ** 2 / (Mp @ T_r ** 2)), e_xr=rel(o["xr_S"], o_r["xr_S"]),
                e_Numax=rel(o["Nu_max"], o_r["Nu_max"]), e_Nuavg=rel(o["Nu_avg"], o_r["Nu_avg"]),
                e_Nu=max(rel(o[k], o_r[k]) for k in NUX))


experiment.run(case, g, ("Re", "ER"), rom, outputs)
