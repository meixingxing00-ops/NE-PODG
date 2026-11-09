"""训练候选方向网络：快照预算 N<B> 的训练快照（方法设计 §3）。种子默认 1234，存 cases/nc/models/v/n<B>.pt；
给定其他种子 s 时存 models/v<s>/n<B>.pt。损失曲线（每 steps/100 步的平均损失）另存为同目录的 n<B>_loss.csv。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.train B [seed] [steps]"""
import sys
import time

import torch

from cases.nc.case import MODELS, NC, cases, eta
from nepodg import candidates

STEPS, LR, BATCH, SEED = 30000, 1e-3, 8, 1234
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
kind = "v" if seed == SEED else f"v{seed}"
nc = NC()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
snaps = [nc.load(i) for i in ids]
U = torch.stack([s[0] for s in snaps], 1)
torch.manual_seed(seed)
net = candidates.Net(**NET).to(nc.X.device)
t0 = time.perf_counter()
hist = candidates.train(net, nc.X, [eta(s[2]) for s in snaps], U, nc.admissible, nc.whiten, steps, LR, BATCH, seed,
                        log=max(1, steps // 100))
print(f"n{budget} {kind}：{len(ids)} 个训练快照，{steps} 步，{time.perf_counter() - t0:.0f} s，末段损失 {hist[-1]:.3e}")
(MODELS / kind).mkdir(parents=True, exist_ok=True)
candidates.save(MODELS / kind / f"n{budget}.pt", net, hist)
with open(MODELS / kind / f"n{budget}_loss.csv", "w") as fh:     # 每 steps/100 步的平均损失
    fh.write("step,loss\n")
    fh.writelines(f"{(k + 1) * max(1, steps // 100)},{a:.6e}\n" for k, a in enumerate(hist))
