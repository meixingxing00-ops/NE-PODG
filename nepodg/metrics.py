"""误差评价（方法设计 §7）：主场 M 范数相对误差 e、仿射空间 y₀ + span B 中的最佳逼近误差 e_proj 与 e_real，
e² = e_proj² + e_real²。白化坐标中计算，Bw 为 span B 的白化正交基，由 basis 从闭合所用的状态基得到。"""
import torch


def basis(B, whiten):
    """状态基 B 张成空间的白化正交基：whiten(B) 的 QR（在 CPU 上算）。状态基与 space 给出的白化基在 σ/σ₁ ~ 1e-8 的方向上
    相差舍入的放大量，e_proj 须按闭合实际所用的 span B 计算。"""
    Bw = whiten(B)
    return torch.linalg.qr(Bw.cpu())[0].to(Bw.device)


def errors(y, y_ref, y0, Bw, whiten, lift):
    d = whiten(y_ref - y0)
    scale = torch.linalg.norm(whiten(y_ref - lift))
    e = torch.linalg.norm(whiten(y - y_ref)) / scale
    ep = torch.linalg.norm(d - Bw @ (Bw.T @ d)) / scale
    return float(e), float(ep), float(torch.sqrt(torch.clamp(e ** 2 - ep ** 2, min=0)))
