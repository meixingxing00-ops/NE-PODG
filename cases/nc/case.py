"""方腔自然对流的实例化（实例化设计）：主场为内部角点流函数 ψ（容许坐标，提升为 0），度量为速度 L²，温度为辅助场。
向 nepodg 提供 whiten、reduced、网络采样点 X 与容许映射 admissible、参数坐标 eta 与快照读取。"""
import csv
from pathlib import Path

import numpy as np
import torch

from solvers.mac.gpu import DEV, Closure
from solvers.mac.mac import Cavity, nodes

ROOT = Path(__file__).resolve().parent
N, S = 128, 2.0


def nu_kap(mu):
    ra, pr = mu
    return np.sqrt(pr / ra), 1 / np.sqrt(ra * pr)


def eta(mu):
    """参数坐标：训练区映到 [−1, 1]²。"""
    return np.array([(np.log10(mu[0]) - 3.25) / 2.25, np.log10(mu[1])])


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class NC:
    def __init__(self, cols=256):
        x = nodes(N, S)
        self.cav = Cavity(x, x, thermal=True)
        self.op = Closure(self.cav, cols)
        self.n = self.cav.npsi
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)
        # 网络采样点为全部角点（[i, j] 顺序），坐标缩放到 [−1, 1]；边界因子 B = 256 x²(1−x)² y²(1−y)²
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.X = torch.as_tensor(np.stack([2 * X.ravel() - 1, 2 * Y.ravel() - 1], 1), dtype=torch.float32, device=DEV)
        self.bf = torch.as_tensor((256 * (X * (1 - X) * Y * (1 - Y)) ** 2).ravel(), dtype=torch.float64, device=DEV)

    def whiten(self, Y):
        return self.op.whiten(Y)

    def admissible(self, F):
        """角点上的原始场 F (P,) 或 (P, k) → 内部角点流函数 B·F（容许坐标）。"""
        G = (self.bf if F.dim() == 1 else self.bf[:, None]) * F
        return G.reshape(N + 1, N + 1, *F.shape[1:])[1:-1, 1:-1].reshape(self.n, *F.shape[1:])

    def reduced(self, y, B, mu):
        return self.op.reduced(y, B, *nu_kap(mu))

    @staticmethod
    def load(cid):
        """快照：内部角点 ψ（GPU）、温度、参数。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["psi"][1:-1, 1:-1].ravel(), device=DEV), d["T"].ravel(), (float(d["Ra"]), float(d["Pr"]))
