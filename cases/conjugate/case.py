"""通道共轭传热的实例化（实例化设计 §7）：主场为全部单元的温度增量 u = T − ℓ_h，ℓ_h(x_i, y_j) = g(x_i) 与参数无关，
任意单元向量都容许；度量为参考域分区 L²（单元面积），无辅助场。
闭合为线性 Galerkin：u_r = u₀ + B c，(Bᵀ A B) c = Bᵀ(f − A ℓ_h − A u₀)，直接求解。"""
import csv
from pathlib import Path

import numpy as np
import torch

from cases.conjugate.fom.config import LENGTH, THICKNESS, mesh, outputs

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"                    # <网络名>/n<B>.pt
DEV = torch.device("cuda")


def eta(mu):
    """参数坐标：训练区 Pe ∈ [10, 1000]、ρ ∈ [0.1, 100] 映到 [−1, 1]²。"""
    return np.array([np.log10(mu[0]) - 2, (np.log10(mu[1]) - 0.5) / 1.5])


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class Conjugate:
    def __init__(self):
        self.ch = ch = mesh()
        self.n = ch.n
        g = np.sin(np.pi * np.minimum(ch.xc, 1.0) / 2) ** 2
        self.ell = np.repeat(g, ch.ny)                                     # k = i·ny + j
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)   # 主场已是增量
        self.w = torch.as_tensor(np.sqrt(ch.Mh), device=DEV)
        X, Y = np.meshgrid(2 * ch.xc / LENGTH - 1, 2 * (ch.yc + THICKNESS) / (1 + THICKNESS) - 1, indexing="ij")
        self.X = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=DEV)

    def whiten(self, Y):
        return self.w * Y if Y.dim() == 1 else self.w[:, None] * Y

    @staticmethod
    def admissible(F):
        return F

    def solve(self, u0, B, mu):
        """仿射空间 u₀ + span B 中的 Galerkin 解（GPU 张量）。"""
        A, f = self.ch.matrix(*mu)
        Bn, u = B.cpu().numpy(), u0.cpu().numpy()
        c = np.linalg.solve(Bn.T @ (A @ Bn), Bn.T @ (f - A @ (self.ell + u)))
        return u0 + B @ torch.as_tensor(c, device=DEV)

    def temperature(self, u):
        return (u.cpu().numpy() if torch.is_tensor(u) else u) + self.ell

    def outputs(self, u, mu):
        """界面总热流 Q、平均界面温度 Tw、出口混合温度 Tout、x = 0.25、1.25、2.5、4.75 处的局部 Nu。"""
        return outputs(self.ch, self.temperature(u), *mu)

    @staticmethod
    def load(cid):
        """快照：温度增量 u（GPU）、参数 (Pe, ρ)。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["u"], device=DEV), (float(d["Pe"]), float(d["ratio"]))

    @staticmethod
    def sens(ids):
        """训练点 ∂u/∂η，(n, len(ids), 2) 的 GPU 张量；快照文件中的 u_e1、u_e2 已对 η 求得。"""
        d = [np.load(ROOT / "data" / f"{i}.npz") for i in ids]
        return torch.as_tensor(np.stack([np.stack([x["u_e1"], x["u_e2"]], 1) for x in d], 1), device=DEV)
