"""多孔方腔的实例化（实例化设计 §6–7）：主场为内部角点流函数 ψ（提升为 0），度量为速度 L²，温度为辅助场；
Brinkman–Darcy 模型无动量输运（inertia=0），阻力系数 d = ν/Da。参数 μ = (Ra*, Da)，Ra = Ra*/Da。"""
import csv
import re
from pathlib import Path

import numpy as np
import torch

from solvers.mac.gpu import DEV, Closure
from solvers.mac.mac import Cavity, nodes

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"                    # <网络名>/n<B>.pt
N, S, PR = 128, 2.5, 0.71
SEED, M = 1234, 128


def kind(name):
    """网络名 → (是否训练, 种子, m)。v[种子][m<m>]：训练的网络；r<种子>[m<m>]：同结构随机初始化的未训练网络。"""
    t, seed, m = re.fullmatch(r"([vr])(\d*)(?:m(\d+))?", name).groups()
    return t == "v", int(seed or SEED), int(m or M)


def coefficients(mu):
    """(ν, κ, d)：ν = √(Pr/Ra)，κ = 1/√(Ra Pr)，d = ν/Da，Ra = Ra*/Da。"""
    ras, da = mu
    ra = ras / da
    nu = np.sqrt(PR / ra)
    return nu, 1 / np.sqrt(ra * PR), nu / da


def eta(mu):
    """参数坐标：训练区 Ra* ∈ [31.6, 1000]、Da ∈ [1e-4, 1e-2] 映到 [−1, 1]²。"""
    return np.array([(np.log10(mu[0]) - 2.25) / 0.75, np.log10(mu[1]) + 3])


def cavity():
    x = nodes(N, S)
    return Cavity(x, x, thermal=True, inertia=0.0)


def cases():
    return list(csv.DictReader(open(ROOT / "cases.csv", encoding="utf-8")))


class Porous:
    def __init__(self):
        self.cav = cavity()
        self.op = Closure(self.cav)
        self.n = self.cav.npsi
        self.lift = torch.zeros(self.n, dtype=torch.float64, device=DEV)
        x = self.cav.x
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.X = torch.as_tensor(np.stack([2 * X.ravel() - 1, 2 * Y.ravel() - 1], 1), dtype=torch.float32, device=DEV)
        self.bf = torch.as_tensor((256 * (X * (1 - X) * Y * (1 - Y)) ** 2).ravel(), dtype=torch.float64, device=DEV)

    def whiten(self, Y):
        return self.op.whiten(Y)

    def admissible(self, F):
        """角点上的原始场 F (P,) 或 (P, k) → 内部角点流函数 B·F。"""
        G = (self.bf if F.dim() == 1 else self.bf[:, None]) * F
        return G.reshape(N + 1, N + 1, *F.shape[1:])[1:-1, 1:-1].reshape(self.n, *F.shape[1:])

    def reduced(self, y, B, mu):
        nu, kap, d = coefficients(mu)
        return self.op.reduced(y, B, nu, kap, drag=d)

    @staticmethod
    def load(cid):
        """快照：内部角点 ψ（GPU）、单元温度、参数 (Ra*, Da)。"""
        d = np.load(ROOT / "data" / f"{cid}.npz")
        return torch.as_tensor(d["psi"], device=DEV), d["T"], (float(d["Ras"]), float(d["Da"]))

    @staticmethod
    def sens(ids):
        """训练点 ∂ψ/∂η，(n, len(ids), 2) 的 GPU 张量；快照文件中的 psi_e1、psi_e2 已对 η 求得。"""
        d = [np.load(ROOT / "data" / f"{i}.npz") for i in ids]
        return torch.as_tensor(np.stack([np.stack([x["psi_e1"], x["psi_e2"]], 1) for x in d], 1), device=DEV)
