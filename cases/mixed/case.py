"""顶盖驱动混合对流的实例化（实例化设计 §6–7）：主场为内部角点流函数增量 y = ψ − ψ_L，ψ_L 为恒速顶盖、无热场、无动量输运、
ν = 1 的离散 Stokes 解，与参数无关；度量为速度 L²，温度为辅助场。参数 μ = (Re, Ri)，ν = 1/Re，κ = 1/(Re Pr)，浮力系数 Ri。
误差按方法设计 §7 以容许增量归一：‖y*‖_M（lift = 0）。"""
import csv
from pathlib import Path

import numpy as np
import torch

from solvers.mac.gpu import DEV, Closure
from solvers.mac.mac import Cavity, nodes

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"                    # <网络名>/n<B>.pt
N, S, PR = 128, 2.0, 0.71
LID = lambda x: np.ones_like(x)


def coefficients(mu):
    """(ν, κ, 浮力系数)。"""
    re_, ri = mu
    return 1 / re_, 1 / (re_ * PR), ri


def eta(mu):
    """参数坐标：训练区 Re ∈ [20, 100]、Ri ∈ [0.1, 5] 映到 [−1, 1]²。"""
    return np.array([2 * (np.log10(mu[0]) - np.log10(20)) / np.log10(5) - 1,
                     2 * (np.log10(mu[1]) + 1) / np.log10(50) - 1])


def cavity():
    x = nodes(N, S)
    return Cavity(x, x, lid=LID, thermal=True)


def stokes_lift():
    """恒速顶盖、无热场、无动量输运、ν = 1 的离散 Stokes 解（内部角点）。"""
    x = nodes(N, S)
    psi, _, st = Cavity(x, x, lid=LID, thermal=False, inertia=0.0).solve(1.0, tol=1e-12)
    assert st["converged"], st
    return psi


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class Mixed:
    def __init__(self):
        self.cav = cavity()
        self.op = Closure(self.cav)
        self.n = self.cav.npsi
        self.psi_L = torch.as_tensor(stokes_lift(), device=DEV)
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)
        x = self.cav.x
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.X = torch.as_tensor(np.stack([2 * X.ravel() - 1, 2 * Y.ravel() - 1], 1), dtype=torch.float32, device=DEV)
        self.bf = torch.as_tensor((256 * (X * (1 - X) * Y * (1 - Y)) ** 2).ravel(), dtype=torch.float64, device=DEV)

    def whiten(self, Y):
        return self.op.whiten(Y)

    def admissible(self, F):
        """角点上的原始场 F (P,) 或 (P, k) → 内部角点流函数增量 B·F。"""
        G = (self.bf if F.dim() == 1 else self.bf[:, None]) * F
        return G.reshape(N + 1, N + 1, *F.shape[1:])[1:-1, 1:-1].reshape(self.n, *F.shape[1:])

    def reduced(self, y, B, mu):
        nu, kap, ri = coefficients(mu)
        return self.op.reduced(self.psi_L + y, B, nu, kap, buoyancy=ri)

    @staticmethod
    def load(cid):
        """快照：流函数增量 y = ψ − ψ_L（GPU）、单元温度、参数 (Re, Ri)。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["y"], device=DEV), d["T"], (float(d["Re"]), float(d["Ri"]))

    @staticmethod
    def sens(ids):
        """训练点 ∂y/∂η = ∂ψ/∂η，(n, len(ids), 2) 的 GPU 张量；快照文件中的 psi_e1、psi_e2 已对 η 求得。"""
        d = [np.load(ROOT / "data" / f"{i}.npz") for i in ids]
        return torch.as_tensor(np.stack([np.stack([x["psi_e1"], x["psi_e2"]], 1) for x in d], 1), device=DEV)
