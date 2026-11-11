"""后台阶的实例化（实例化设计 §6–7）：主场为自由角点流函数的增量 q = y − ℓ_S（ℓ_S 为 ER = 1.75 的离散 Stokes 解，与参数无关），
度量为参考域速度 L²；温度为被动标量，不进入闭合，降阶解求出后在全网格上解一次温度。
误差按方法设计 §7 以容许增量归一：‖q*‖_M（lift = 0）。
闭合：g = Bᵀ r_y(ℓ_S + q)，G = Bᵀ J_yy B，r_y 与 J_yy 由 openmac 在 CPU 上组装。"""
import csv
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from cases.bfs.fom.config import LIN, LOUT, eta, model
from cases.bfs.fom.grid_study import outputs

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"                    # <网络名>/n<B>.pt
DEV = torch.device("cuda")
SEED, M = 1234, 128
DELTA = 0.5                                 # 边界因子的衰减长度：入口段半高


def kind(name):
    """网络名 → (是否训练, 种子, m)。v[种子][m<m>]：训练的网络；r<种子>[m<m>]：同结构随机初始化的未训练网络。"""
    t, seed, m = re.fullmatch(r"([vr])(\d*)(?:m(\d+))?", name).groups()
    return t == "v", int(seed or SEED), int(m or M)


def _distance(x, y, a, b):
    """点 (x, y) 到线段 ab 的距离。"""
    d = np.subtract(b, a)
    t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(x - a[0] - t * d[0], y - a[1] - t * d[1])


def stokes_lift(m):
    """ER = 1.75 的离散 Stokes 解（对流项置零，解与 Re 无关），自由角点。"""
    mu, z = np.array([200., 1.75]), np.zeros(m.grid.ny_state)
    return spla.spsolve(m.flow_jacobian(z, mu, convection=0.).tocsc(), -m.flow_residual(z, mu, convection=0.))


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class BFS:
    def __init__(self):
        self.m = m = model()
        g = m.grid
        self.n = g.ny_state
        self.free = torch.as_tensor(g.free, device=DEV)
        L = (sp.diags(np.sqrt(m.Muhat)) @ m.Chat).tocsr()
        self.L = torch.sparse_csr_tensor(torch.as_tensor(L.indptr), torch.as_tensor(L.indices),
                                         torch.as_tensor(L.data), size=L.shape, device=DEV)
        self.ell_np = stokes_lift(m)
        self.ell = torch.as_tensor(self.ell_np, device=DEV)
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)
        # 网络采样点为参考域全部角点（[i, j] 顺序），输入为网格指标坐标 (2i/nx − 1, 2j/ny − 1)：入口与台阶附近的加密区在输入中展开
        I, J = np.meshgrid(np.arange(g.nx + 1), np.arange(g.ny + 1), indexing="ij")
        self.X = torch.as_tensor(np.stack([2 * I.ravel() / g.nx - 1, 2 * J.ravel() / g.ny - 1], 1), dtype=torch.float32,
                                 device=DEV)
        # 边界因子 b：在入口、上壁与下壁折线处按 d² 趋于零，出口不加
        x, y = g.xv.ravel(), g.yv.ravel()
        low = np.min([_distance(x, y, a, b) for a, b in (((-LIN, 0), (0, 0)), ((0, 0), (0, -1)), ((0, -1), (LOUT, -1)))], 0)
        self.bf = torch.as_tensor((np.tanh(low / DELTA) * np.tanh((1 - y) / DELTA) * np.tanh((x + LIN) / DELTA)) ** 2,
                                  device=DEV)

    def whiten(self, Y):
        return self.L @ Y if Y.dim() == 2 else (self.L @ Y[:, None])[:, 0]

    def admissible(self, F):
        """角点上的原始场 F (P,) 或 (P, k) → 自由角点的流函数增量 [b F]_free（固定角点增量为零，出口角点保留）。"""
        return ((self.bf if F.dim() == 1 else self.bf[:, None]) * F)[self.free]

    def state(self, q):
        return self.ell_np + (q.cpu().numpy() if torch.is_tensor(q) else q)

    def reduced(self, q, B, mu):
        y = self.state(q)
        Bn = B.cpu().numpy()
        mu = np.asarray(mu, float)
        r, J = self.m.flow_residual(y, mu), self.m.flow_jacobian(y, mu)
        return torch.as_tensor(Bn.T @ r, device=DEV), torch.as_tensor(Bn.T @ (J @ Bn), device=DEV), None

    def temperature(self, q, mu):
        return self.m.temperature(self.state(q), np.asarray(mu, float))

    def outputs(self, q, T, mu):
        return outputs(self.m, self.state(q), T, np.asarray(mu, float))

    @staticmethod
    def load(cid):
        """快照：流函数增量 q（GPU）、温度、参数 (Re, ER)。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["q"], device=DEV), d["T"], (float(d["Re"]), float(d["ER"]))

    @staticmethod
    def sens(ids):
        """训练点 ∂q/∂η，(n, len(ids), 2) 的 GPU 张量；快照文件中的 q_e1、q_e2 已对 η 求得。"""
        d = [np.load(ROOT / "data" / f"{i}.npz") for i in ids]
        return torch.as_tensor(np.stack([np.stack([x["q_e1"], x["q_e2"]], 1) for x in d], 1), device=DEV)
