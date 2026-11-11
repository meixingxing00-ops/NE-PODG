"""后台阶算例的高保真设置（实例化设计 §1、§5）：入口段 4h、出口距台阶 40h、每入口高度 48 个单元，
台阶下游下壁定温 T=1，入口 T=0，其余壁绝热；Pr=0.71。参数 μ=(Re, ER)，Re=2U_b h/ν。"""
import numpy as np

from solvers.openmac.benchmarks import bfs

N, LIN, LOUT, PR = 48, 4., 40., .71


def model(n=N):
    return bfs(n, Lin=LIN, Lout=LOUT, Pr=PR)


def eta(mu):
    """训练区 Re∈[100,400]、ER∈[1.5,2] 映到 [-1,1]²。"""
    return np.array([(np.log10(mu[0]) - np.log10(200)) / np.log10(2), 4 * (mu[1] - 1.75)])


def deta(mu):
    """dμ/dη（对角）：dRe/dη₁ = Re·ln2，dER/dη₂ = 1/4。"""
    return np.array([mu[0] * np.log(2), .25])
