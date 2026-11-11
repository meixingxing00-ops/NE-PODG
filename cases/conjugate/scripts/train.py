"""训练候选方向网络：快照预算 N<B> 的训练快照（方法设计 §3）。网络名 kind 见 case.kind（v、v<种子>、可加 m<m>），
默认 v（种子 1234，m = 128），存 cases/conjugate/models/<kind>/n<B>.pt；损失曲线（每 steps/100 步的平均损失）另存为同目录的 n<B>_loss.csv。
用法（仓库根目录，PINNGPU）：python -m cases.conjugate.scripts.train B [kind] [steps]"""
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch

from cases.conjugate.case import MODELS, Conjugate, cases, eta, kind
from nepodg import candidates

STEPS, LR, BATCH = 30000, 1e-3, 8
NET = dict(layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "v"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
_, seed, m = kind(name)
g = Conjugate()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
snaps = [g.load(i) for i in ids]
U = torch.stack([s[0] for s in snaps], 1)
torch.manual_seed(seed)
net = candidates.Net(m=m, **NET).to(g.X.device)
t0 = time.perf_counter()
hist = candidates.train(net, g.X, [eta(s[1]) for s in snaps], U, g.admissible, g.whiten, steps, LR, BATCH, seed,
                        log=max(1, steps // 100))
print(f"n{budget} {name}：{len(ids)} 个训练快照，m = {m}，{steps} 步，{time.perf_counter() - t0:.0f} s，末段损失 {hist[-1]:.3e}")
(MODELS / name).mkdir(parents=True, exist_ok=True)
candidates.save(MODELS / name / f"n{budget}.pt", net, hist)
with open(MODELS / name / f"n{budget}_loss.csv", "w") as fh:     # 每 steps/100 步的平均损失
    fh.write("step,loss\n")
    fh.writelines(f"{(k + 1) * max(1, steps // 100)},{a:.6e}\n" for k, a in enumerate(hist))
