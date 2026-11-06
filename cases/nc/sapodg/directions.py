"""SA-PODG 的候选方向：快照预算 nB 的全部训练快照处，流函数对 log10 Ra、log10 Pr 的稳态灵敏度（sensitivities.py 生成），
共 2·N_B 列，与查询参数无关。经 nepodg.space.complement 去 POD 分量、正交化后作为增广方向。"""
from pathlib import Path

import numpy as np
import torch

from cases.nc.case import cases
from solvers.mac.gpu import DEV

DATA = Path(__file__).resolve().parent / "data"


def directions(budget):
    """(n, 2 N_B) 的 GPU float64 张量。"""
    cols = []
    for r in cases():
        if r[f"n{budget}"] == "1":
            d = np.load(DATA / f"{r['id']}.npz")
            cols += [d["psi_Ra"], d["psi_Pr"]]
    return torch.as_tensor(np.stack(cols, 1), device=DEV)
