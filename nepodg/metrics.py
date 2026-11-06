"""误差评价（方法设计 §7）：主场 M 范数相对误差 e、仿射空间 y₀ + span B 中的最佳逼近误差 e_proj 与 e_real，
e² = e_proj² + e_real²。白化坐标中计算，Bw 为 span B 的白化正交基。"""
import torch


def errors(y, y_ref, y0, Bw, whiten, lift):
    d = whiten(y_ref - y0)
    scale = torch.linalg.norm(whiten(y_ref - lift))
    e = torch.linalg.norm(whiten(y - y_ref)) / scale
    ep = torch.linalg.norm(d - Bw @ (Bw.T @ d)) / scale
    return float(e), float(ep), float(torch.sqrt(torch.clamp(e ** 2 - ep ** 2, min=0)))
