"""训练候选方向网络：快照预算 nB（12、20、30、50）的训练快照，模型存 cases/nc/models/n<B>.pt。
用法（仓库根目录，PINNGPU）：python -m cases.nc.scripts.train B [steps]"""
import sys
import time

import torch

from cases.nc.case import NC, ROOT, cases, eta
from nepodg import candidates

STEPS, LR, BATCH, SEED = 100000, 1e-3, 8, 1234
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS
nc = NC()
train = [c for c in cases() if c[f"n{budget}"] == "1"]
snaps = [nc.load(c["id"]) for c in train]
U = torch.stack([s[0] for s in snaps], 1)
mean = U.mean(1)
torch.manual_seed(SEED)
net = candidates.Net(**NET).to(nc.X.device)
t0 = time.perf_counter()
hist = candidates.train(net, nc.X, [eta(s[2]) for s in snaps], U, mean, nc.admissible, nc.whiten,
                        steps, LR, BATCH, SEED, log=max(1, steps // 100))
print(f"n{budget}：{len(train)} 个训练快照，{steps} 步，{time.perf_counter() - t0:.0f} s，末段损失 {hist[-1]:.3e}")
(ROOT / "models").mkdir(exist_ok=True)
candidates.save(ROOT / "models" / f"n{budget}.pt", net, mean, hist)
