"""Couette–Graetz 的实例化（实例化设计）：主场为全部单元温度 T（提升为 0，任意单元向量都容许），度量为参考域 L²，无辅助场。
向 nepodg 提供 whiten、闭合 solve、网络采样点 X 与容许映射 admissible、参数坐标 eta 与快照读取。
闭合为线性 Galerkin：T_r = T₀ + B c，(Bᵀ A B) c = Bᵀ(f − A T₀)，直接求解。"""
import csv
from pathlib import Path

import numpy as np
import torch

from solvers.fv.fv import Channel, grid

ROOT = Path(__file__).resolve().parent
DEV = torch.device("cuda")


def mesh():
    """入口段 40、受热段 160、y 向 80 个单元（16000），tanh 加密参数 1.5、2.0、2.0；LUDS 对流格式。"""
    return Channel(*grid(40, 1.5, 160, 2.0, 80, 2.0), scheme="luds")


def eta(mu):
    """参数坐标：训练区 L ∈ [1, 10]、Pe ∈ [0.1, 100] 映到 [−1, 1]²。"""
    return np.array([2 * np.log10(mu[0]) - 1, (np.log10(mu[1]) - 0.5) / 1.5])


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class Graetz:
    def __init__(self):
        self.ch = ch = mesh()
        self.n = ch.n
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)
        self.w = torch.as_tensor(np.sqrt(ch.Mh), device=DEV)
        # 网络采样点为全部单元中心，参考坐标缩放到 [−1, 1]
        X, Y = np.meshgrid(ch.xhc - 1, 2 * ch.yc - 1, indexing="ij")
        self.X = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=DEV)

    def whiten(self, Y):
        return self.w * Y if Y.dim() == 1 else self.w[:, None] * Y

    @staticmethod
    def admissible(F):
        return F

    def solve(self, T0, B, mu):
        """仿射空间 T₀ + span B 中的 Galerkin 解（GPU 张量）。"""
        A, f = self.ch.matrix(*mu)
        Bn, t0 = B.cpu().numpy(), T0.cpu().numpy()
        c = np.linalg.solve(Bn.T @ (A @ Bn), Bn.T @ (f - A @ t0))
        return T0 + B @ torch.as_tensor(c, device=DEV)

    def tav(self, T, mu):
        return self.ch.tav(T.cpu().numpy() if torch.is_tensor(T) else T, mu[1])

    @staticmethod
    def load(cid):
        """快照：单元温度（GPU）、参数 (L, Pe)。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["T"], device=DEV), (float(d["L"]), float(d["Pe"]))

    @staticmethod
    def sens_directions(budget):
        """SA-PODG 候选方向：预算 N<budget> 全部训练点的 ∂T/∂η₁、∂T/∂η₂，(n, 2 N_B)，与查询参数无关。"""
        cols = []
        for r in cases():
            if r[f"n{budget}"] == "1":
                d = np.load(ROOT / "data" / f"{r['id']}.npz")
                cols += [d["T_e1"], d["T_e2"]]
        return torch.as_tensor(np.stack(cols, 1), device=DEV)
