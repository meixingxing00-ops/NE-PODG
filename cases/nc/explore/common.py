"""探索用公共部分：预算 N_B 的训练快照、POD、SA 空间，任意正交基上的去投影与最佳逼近误差。
基统一为 (B, Bw)：状态坐标与白化坐标，Bw 列正交。"""
import numpy as np
import torch

from cases.nc.case import NC, cases, eta
from cases.nc.case import MODELS, SENS as SA_DATA
from nepodg import candidates, space
from solvers.mac.gpu import DEV

SEG = [("interp", lambda r: 1e3 < r <= 3.17e4, "内插 10³<Ra≤3.16×10⁴"),
       ("interp", lambda r: r > 3.17e4, "内插 Ra>3.16×10⁴"),
       ("extrap", lambda r: abs(r - 4.22e5) < 1e3, "外推 4.22×10⁵"),
       ("extrap", lambda r: abs(r - 5.62e5) < 1e3, "外推 5.62×10⁵"),
       ("extrap", lambda r: r > 9e5, "外推 1.00×10⁶")]
# 不在 cases.csv 预算列中的预算：训练快照编号
IDS = {6: ["P001", "P007", "P010", "P041", "P047", "P050"]}                 # Ra{10, 10⁴, 3.16×10⁵} × Pr{0.1, 10}


def orth(X, Xw, delta=1e-8):
    return space._orth(X, Xw, lambda s: s / s[0] > delta)[:2]


def extend(base, Phi, whiten, tau=1e-6, delta=1e-8):
    """候选方向 Phi 去掉 span base 的分量后正交化；返回 (Z, Zw, s)，s 降序（主角正弦）。"""
    B, Bw = base
    Q, Qw = orth(Phi, whiten(Phi), delta)
    A, Aw = Q - B @ (Bw.T @ Qw), Qw - Bw @ (Bw.T @ Qw)
    Z, Zw, s = space._orth(A, Aw, lambda s: s > tau)
    return Z, Zw, s[:Z.shape[1]]


def extend_k(base, Phi, whiten, k):
    """Phi 原始列去 span base 分量后，按白化能量取前 k 个方向。"""
    B, Bw = base
    Pw = whiten(Phi)
    A, Aw = Phi - B @ (Bw.T @ Pw), Pw - Bw @ (Bw.T @ Pw)
    return space._orth(A, Aw, lambda s: (torch.arange(len(s), device=s.device) < k) & (s / s[0] > 1e-8))[:2]


def cat(*bases):
    return torch.cat([b[0] for b in bases], 1), torch.cat([b[1] for b in bases], 1)


def eproj(dw, scale, Bw):
    """白化目标 dw = L(u − ū) 在 span Bw 中的相对最佳逼近误差。"""
    return float(torch.linalg.norm(dw - Bw @ (Bw.T @ dw)) / scale)


class Budget:
    def __init__(self, nc, budget):
        self.nc, self.budget = nc, budget
        self.rows = [c for c in cases() if (c["id"] in IDS[budget] if budget in IDS else c[f"n{budget}"] == "1")]
        self.snaps = [nc.load(c["id"]) for c in self.rows]
        self.U = torch.stack([s[0] for s in self.snaps], 1)
        self.etas = [eta(s[2]) for s in self.snaps]
        self.pod = space.POD(self.U, nc.whiten)
        self.Vb = (self.pod.V, self.pod.Vw)
        d = [np.load(SA_DATA / f"{c['id']}.npz") for c in self.rows]               # 各训练点 ∂ψ/∂log10 Ra、∂ψ/∂log10 Pr
        self.Sraw = torch.as_tensor(np.stack([v for x in d for v in (x["psi_Ra"], x["psi_Pr"])], 1), device=DEV)
        Z, Zw, _ = extend(self.Vb, self.Sraw, nc.whiten)
        self.VS = cat(self.Vb, (Z, Zw))

    def net(self):
        """主线只用值监督的网络 cases/nc/models/archive/centered/v/n<B>.pt。"""
        return candidates.load(MODELS / "archive" / "centered" / "v" / f"n{self.budget}.pt", self.nc.X.device)


def poly(nc, deg):
    """𝓑·P_a(x)P_b(y)，Legendre，a + b ≤ deg，坐标为网络采样点（[−1, 1]²），经容许映射。"""
    from numpy.polynomial import legendre
    X = nc.X.double().cpu().numpy()
    Lx = [legendre.legval(X[:, 0], np.eye(deg + 1)[a]) for a in range(deg + 1)]
    Ly = [legendre.legval(X[:, 1], np.eye(deg + 1)[a]) for a in range(deg + 1)]
    F = np.stack([Lx[a] * Ly[b] for a in range(deg + 1) for b in range(deg + 1 - a)], 1)
    return nc.admissible(torch.as_tensor(F, device=nc.lift.device))


def tests(role=None):
    return [c for c in cases() if c["role"] in ((role,) if role else ("interp", "extrap"))]


def target(nc, mean, cid):
    psi, T, mu = nc.load(cid)
    return nc.whiten(psi - mean), float(torch.linalg.norm(nc.whiten(psi))), mu


def seg_table(rows, cols, fmt=lambda v: f"{v * 100:.3g}"):
    """rows：dict 列表（含 role、Ra 与 cols 中的键），返回按 SEG 分段的中位数 markdown 行。"""
    out = []
    for role, f, name in SEG:
        sub = [r for r in rows if r["role"] == role and f(float(r["Ra"]))]
        out.append(f"| {name} | {len(sub)} | " + " | ".join(fmt(np.median([r[c] for r in sub])) for c in cols) + " |")
    return "\n".join(out)
