"""训练候选方向网络：快照预算 N<B> 的训练快照与灵敏度（方法设计 §3、实例化设计 §8）。kind：
  v（默认）  快照值监督，存 cases/graetz/models/v/n<B>.pt
  d          快照值与灵敏度共同监督（λ = 1，消融），存 cases/graetz/models/d/n<B>.pt
kind 后接整数（如 d2、v2）时以该数为种子，存 models/<kind>/n<B>.pt；默认种子 1234。
损失曲线（每 steps/100 步的平均值项、导数项）另存为 models/<kind>/n<B>_loss.csv。
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.train B [kind] [steps]"""
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch

from cases.graetz.case import MODELS, Graetz, cases, eta
from nepodg import candidates

STEPS, LR, BATCH, SEED, LAM = 30000, 1e-3, 8, 1234, 1.0
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "v"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
seed = int(kind[1:]) if len(kind) > 1 else SEED
g = Graetz()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
snaps = [g.load(i) for i in ids]
U = torch.stack([s[0] for s in snaps], 1)
torch.manual_seed(seed)
net = candidates.Net(**NET).to(g.X.device)
t0 = time.perf_counter()
hist = candidates.train(net, g.X, [eta(s[1]) for s in snaps], U, g.admissible, g.whiten, steps, LR, BATCH, seed,
                        log=max(1, steps // 100), S=g.sens(ids) if kind[0] == "d" else None, lam=LAM)
print(f"n{budget} {kind}：{len(ids)} 个训练快照，{steps} 步，{time.perf_counter() - t0:.0f} s，"
      f"末段损失（值、导数）{hist[-1][0]:.3e} {hist[-1][1]:.3e}")
(MODELS / kind).mkdir(parents=True, exist_ok=True)
candidates.save(MODELS / kind / f"n{budget}.pt", net, hist)
with open(MODELS / kind / f"n{budget}_loss.csv", "w") as fh:     # 每 steps/100 步的平均损失
    fh.write("step,value,deriv\n")
    fh.writelines(f"{(k + 1) * max(1, steps // 100)},{a:.6e},{b:.6e}\n" for k, (a, b) in enumerate(hist))
